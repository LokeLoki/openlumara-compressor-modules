"""context_compressor — automatic context compression for OpenLumara.

A drop-in module that replaces OpenLumara's default "cut the oldest
messages" behavior with a structure-preserving fold:

  * The chat is split into three zones: a small protected HEAD, a
    token-budgeted protected TAIL (recent conversation kept verbatim),
    and the MIDDLE in between — the only part that gets summarized.
  * When usage crosses the threshold, the middle is summarized by a
    quiet side-channel LLM call into a structured handoff (preset
    selectable: engineering or narrative), enriched by code with
    machine-stamped counters, verbatim user rows, and a regex-harvested
    anchor index, then hard-capped.
  * The result is spliced in as OpenLumara's standard SUMMARIZATION_
    CUTOFF marker + summary pair — fully reversible (delete messages
    above the dotted line to revert; full history stays on disk).

Guards (all module-local, no core changes needed):
  * cooldown after every fold, success or failure
  * failure breaker: N consecutive errors suspend auto-compaction
  * anti-thrash: folds that leave the prompt still over threshold trip
    a block, so incompressible sessions stop re-firing the LLM

Requires: a current-structure OpenLumara install. No core patches.
Settings: summary_preset (Default/Narrative), threshold, tail_ratio,
protect_first_n / protect_last_n, summary_timeout, cap.
"""
import asyncio
import hashlib
import json
import re
import time
import urllib.request

import core

# ---------------------------------------------------------------- constants
# Ported from -agent/agent/context_compressor.py
MAX_TAIL_MESSAGE_FLOOR = 8       # _MAX_TAIL_MESSAGE_FLOOR — count floor when the tail is budget-limited
SUMMARY_RATIO = 0.20             # _SUMMARY_RATIO — summary budget vs content tokens
MIN_SUMMARY_TOKENS = 2000        # _MIN_SUMMARY_TOKENS
SUMMARY_TOKENS_CEILING = 10000   # _SUMMARY_TOKENS_CEILING
CONTENT_MAX = 6000               # _CONTENT_MAX — chars of message body the summarizer sees
CONTENT_HEAD = 4000              # _CONTENT_HEAD
CONTENT_TAIL = 1500              # _CONTENT_TAIL
TOOL_ARGS_MAX = 1500             # _TOOL_ARGS_MAX
SUMMARY_INPUT_MAX_CHARS = 160000 # _SUMMARY_INPUT_MAX_CHARS — aggregate cap on serialized turns
MIN_COMPRESS_MESSAGES = 10       # sanity: don't compact tiny chats
MIN_COMPRESS_CHARS = 4000        # sanity: don't spend an LLM call on < ~1k tokens


# Token-GATED (not turn-cadence) reclaim of old bulky tool output, running
# long before the full-compression threshold. rationale: on
# large-window models should_compress (~50% of the window) rarely fires, so
# old tool outputs otherwise ride in history and are re-sent verbatim on
# every subsequent turn — pure wasted prefill on a local model. Two
# hysteresis gates keep prompt-cache breaks episodic instead of firing every
# tool iteration: (a) a commit only happens when the prune reclaims at least
# proactive_prune_min_reclaim_tokens, (b) after a commit the prune disarms
# until the live context regrows a full trigger-sized runway.

SUMMARIZATION_CUTOFF = {"_metadata": {"signal": "SUMMARIZATION_CUTOFF"}}
DUPLICATE_PLACEHOLDER = "[Duplicate tool output — same content as a more recent call]"

# ---- full-compaction anti-thrashing ( / #14695, ported) ----
# When the incompressible prompt floor (system prompt + tool schemas) already
# meets the compaction threshold, every pass shrinks messages by a healthy
# margin yet leaves the live prompt over the line — so the next turn fires
# again, forever. On a local model that's a re-prefill loop: each fire rewrites
# the transcript ahead of the recent ~1.5k tokens and busts the KV checkpoint.
# Two consecutive "ineffective" verdicts (post-compaction prompt still over
# threshold) block automatic full compaction for anti_thrash_recovery_seconds,
# then allow ONE recovery probe. The deterministic prune/micro passes keep
# running throughout (they don't need the LLM and can still reclaim space).
_FEASIBILITY_SKIP_DEFAULT = 0.10       # pre-LLM skip: middle below this fraction of threshold
_ANTI_THRASH_RECOVERY_DEFAULT = 300    # seconds of block before a recovery probe

# The stored compaction summary is wrapped so a local model reads it as a
# checkpoint, not a live instruction set — otherwise it re-executes summarized
# work ("finish the PR", "run the tests"…) as if it were a fresh ask. The end
# marker closes the summary and points the model at the message to actually
# answer. Kept short on purpose: local models follow a tight, concrete note
# better than ~40-line defensive preamble, and less prompt bloat.
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into "
    "the summary below. Treat it as background reference, NOT as active "
    "instructions: do not re-run, 'finish', or wrap up work it describes. "
    "Respond only to the latest user message that appears AFTER this summary. "
    "If none appears, do nothing and wait. Persistent memory in the system "
    "prompt is always authoritative."
)
SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
)


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)

def strip_summary_wrapper(text):
    """Remove the SUMMARY_PREFIX / SUMMARY_END_MARKER wrapper so iterative
    updates and other consumers see the raw summary body, not the framing."""
    if not text:
        return ""
    t = text
    if t.startswith(SUMMARY_PREFIX):
        t = t[len(SUMMARY_PREFIX):].lstrip()
    end = t.find(SUMMARY_END_MARKER)
    if end != -1:
        t = t[:end].rstrip()
    return t.strip()

# Secret redaction ( before any summary crosses the
# persistence boundary; summaries are re-injected into every later prompt)
_REDACT_PATTERNS = [
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}"), "[REDACTED]"),
    (re.compile(r"\bsk-or-[A-Za-z0-9\-_]{10,}"), "[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bgho_[A-Za-z0-9]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret[_-]?key|password|passwd)\b(\s*[:=]\s*)(['\"]?)([^\s'\",}{]{6,})"),
     lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED]"),
    (re.compile(r"(?i)\b(from|userinfo://)\s*[^\s:]+:[^\s@]+@"), "[REDACTED]@"),
]


def redact_text(text: str) -> str:
    if not text:
        return ""
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...[truncated]"


def _truncate_head_tail(text: str, max_chars: int = CONTENT_MAX,
                        head: int = CONTENT_HEAD, tail: int = CONTENT_TAIL) -> str:
    if len(text) <= max_chars:
        return text
    return text[:head].rstrip() + "\n...[truncated]...\n" + text[-tail:].lstrip()


# base64 inline images are NOT 1 token per 4 chars — a 2.4 MB data-URI is
# ~600k raw chars but the vision encoder sees ~1.5k tokens. Count them at a
# fixed flat cost (matches core/context.py, which strips historical images
# from the API prompt entirely and only sends the newest one).
_DATA_URI_RE = re.compile(r"data:[A-Za-z0-9/+.\\-]+;base64,[A-Za-z0-9+/=]+")
IMAGE_TOKEN_ESTIMATE = 1500

# ---- head/tail hardening (ported from v3.5) ----
TAIL_MAX_CONTEXT_FRACTION = 0.20   # protected tail may never exceed this share of the window
TAIL_SOFT_CEILING_OVERRUN = 1.5    # whole-row overrun allowed before in-tail pressure demotion
PRESSURE_KEEP_RECENT_MESSAGES = 3  # newest rows kept verbatim even under pressure
TAIL_KEEP_TOOL_ROUNDS = 6          # tool results older than the newest N rounds are demote-eligible
MAX_KEEP_TOOL_IMAGES = 3           # newest image-bearing tool results kept verbatim


def _msg_tokens(msg) -> int:
    """Rough token estimate, same chars/4 convention as core/context.py,
    but with base64 image payloads counted at a fixed flat cost instead of
    raw length (a multi-MB data-URI must not read as hundreds of thousands
    of tokens — that breaks every token-budget walk and gate that uses
    this estimator)."""
    if not isinstance(msg, dict):
        try:
            return len(json.dumps(msg, default=str)) // 4
        except Exception:
            return 0
    content = msg.get("content")
    image_count = 0
    if isinstance(content, str):
        image_count = len(_DATA_URI_RE.findall(content))
    elif isinstance(content, list):
        image_count = sum(1 for p in content
                          if isinstance(p, dict) and p.get("type") == "image_url")
    text_msg = msg
    if isinstance(content, str) and image_count:
        text_msg = {**msg, "content": _DATA_URI_RE.sub("[image]", content)}
    elif isinstance(content, list) and image_count:
        # strip the raw base64 out of image_url parts too — a data-URI in a
        # content-part must not be counted at chars/4 either
        def _strip_part(p):
            if isinstance(p, dict) and p.get("type") == "image_url":
                return {"type": "image_url", "image_url": {"url": "[image]"}}
            return p
        text_msg = {**msg, "content": [_strip_part(p) for p in content]}
    try:
        return len(json.dumps(text_msg, default=str)) // 4 + image_count * IMAGE_TOKEN_ESTIMATE
    except Exception:
        return 0


def last_cutoff_index(messages):
    """Index of the most recent SUMMARIZATION_CUTOFF marker, or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("_metadata", {}).get("signal") == "SUMMARIZATION_CUTOFF":
            return i
    return -1


def previous_summary(messages, cutoff_idx):
    """The assistant summary stored right after a cutoff marker (iterative update input)."""
    if cutoff_idx < 0 or cutoff_idx + 1 >= len(messages):
        return ""
    nxt = messages[cutoff_idx + 1]
    if nxt.get("role") == "assistant":
        c = nxt.get("content")
        if isinstance(c, str):
            return c
    return ""


def _tool_call_id(tc):
    """Stable id for one assistant tool_call entry (chat-completions shape)."""
    if not isinstance(tc, dict):
        return ""
    return tc.get("id") or ""


def _sanitize_tool_pairs(messages):
    """Fix orphaned tool_call / tool_result pairs after a compression splice.

    Two failure modes the splice can create, both of which make the next API
    call 400 ("No tool call found for function call output with call_id …" /
    "tool_call_id … does not match"):

    1. A tool *result* whose parent assistant tool_call was summarized away —
       drop the orphaned result row.
    2. An assistant *tool_call* whose result row was dropped — strip the call
       from the assistant message (removing it entirely if it was the only
       one, and giving the message visible content so it isn't an empty turn).

    The trailing non-tool assistant message is treated as a possibly in-flight
    request (its results may not be persisted yet) and is preserved verbatim
    rather than stripped. Returns a new list; never raises on a malformed row.
    """
    surviving = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = _tool_call_id(tc)
                if cid:
                    surviving.add(cid)

    result_ids = set()
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id") or ""
            if cid:
                result_ids.add(cid)

    out = list(messages)

    orphaned_results = result_ids - surviving
    if orphaned_results:
        out = [m for m in out if not (
            m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)]

    missing_results = surviving - result_ids
    if missing_results:
        # the last non-tool message is the live request — never strip its calls
        idx = len(out) - 1
        while idx >= 0 and out[idx].get("role") == "tool":
            idx -= 1
        live = out[idx] if (idx >= 0 and out[idx].get("role") == "assistant") else None
        for i, m in enumerate(out):
            if m.get("role") != "assistant" or m is live:
                continue
            tcs = m.get("tool_calls")
            if not tcs:
                continue
            kept = [tc for tc in tcs if _tool_call_id(tc) not in missing_results]
            if len(kept) == len(tcs):
                continue
            if kept:
                out[i] = {**m, "tool_calls": kept}
            else:
                nm = {k: v for k, v in m.items() if k != "tool_calls"}
                if not (nm.get("content") or "").strip() if isinstance(nm.get("content"), str) else not nm.get("content"):
                    nm["content"] = "(tool call removed)"
                out[i] = nm

    return out


def _merge_adjacent_user_turns(messages):
    """Merge consecutive plain-text user turns the splice left adjacent.

    A compaction that summarizes away an assistant turn can leave two real
    user messages back-to-back, which breaks llama.cpp's role alternation.
    Joining them (\\n\\n, exactly what the API repair pass does) keeps every
    byte the user typed. Metadata-carrying messages (cutoff / summary markers)
    and non-string (multimodal) content are left alone.
    """
    merged = []
    for m in messages:
        prev = merged[-1] if merged else None
        if (isinstance(m, dict) and isinstance(prev, dict)
                and m.get("role") == "user" and prev.get("role") == "user"
                and not m.get("_metadata") and not prev.get("_metadata")
                and isinstance(m.get("content"), str)
                and isinstance(prev.get("content"), str)):
            a, b = prev["content"], m["content"]
            prev["content"] = (a + "\n\n" + b) if (a and b) else (a or b)
            continue
        merged.append(m)
    return merged


def _resplit_point(messages, tail_anchor, tail_index):
    """Re-resolve the summary splice point on a post-sanitize message list.

    Sanitizing can drop rows ahead of the tail, shifting the original
    tail_start index. The splice must land at the TAIL boundary (never right
    after a pre-existing cutoff — that would leave the compressible middle
    after the new cutoff and thus still visible, so nothing would be
    compressed). The tail-boundary message (captured by identity before
    sanitizing) is re-located by identity; if sanitize removed it (an orphaned
    tool result at the boundary), fall back to the original index clamped to
    the shrunken list.
    """
    if tail_anchor is not None:
        try:
            for i, m in enumerate(messages):
                if m is tail_anchor:
                    return i
        except Exception:
            pass
    return min(tail_index, len(messages))


def serialize_turns(turns):
    """Serialize messages into labeled text for the summarizer (, simplified)."""
    call_index = {}
    for m in turns:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("id") or ""
                    if cid:
                        call_index[cid] = tc

    parts = []
    for m in turns:
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else (
            json.dumps(content, default=str) if content is not None else "")
        text = redact_text(text)
        if role == "user":
            parts.append("USER:\n" + (_truncate(text, CONTENT_MAX) or "(empty)"))
        elif role == "assistant":
            if m.get("tool_calls"):
                lines = []
                for tc in m.get("tool_calls"):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {}) or {}
                    args = str(fn.get("arguments") or "")
                    args = redact_text(_truncate(args, TOOL_ARGS_MAX))
                    lines.append(f"TOOL_CALL: {fn.get('name', '?')}({args})")
                if text.strip():
                    lines.append(_truncate_head_tail(text))
                parts.append("ASSISTANT:\n" + "\n".join(l for l in lines if l))
            else:
                parts.append("ASSISTANT:\n" + (_truncate_head_tail(text) or "(empty)"))
        elif role == "tool":
            name = "?"
            cid = m.get("tool_call_id") or ""
            tc = call_index.get(cid)
            if tc:
                name = (tc.get("function", {}) or {}).get("name", "?")
            parts.append(f"TOOL_RESULT ({name}):\n" + (_truncate_head_tail(text) or "(empty)"))
        else:
            parts.append(f"{str(role).upper()}:\n" + _truncate(text, CONTENT_MAX))

    serialized = "\n\n".join(parts)
    if len(serialized) > SUMMARY_INPUT_MAX_CHARS:
        # aggregate cap: keep the tail (most recent = most relevant)
        serialized = "...[earlier turns truncated to fit summarizer budget]...\n" + \
                     serialized[-SUMMARY_INPUT_MAX_CHARS + 100:]
    return serialized


# ------------------------------------------------------- tool-result helpers
# Ported from _summarize_tool_result (OpenLumara tool names added)
def summarize_tool_result(tool_name, tool_args, tool_content):
    """One-line informative summary of a tool call + result. Never raises."""
    try:
        try:
            args = json.loads(tool_args) if tool_args else {}
        except (ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        content = tool_content or ""
        content_len = len(content)
        line_count = content.count("\n") + 1 if content.strip() else 0

        if tool_name in ("terminal", "run_command"):
            cmd = str(args.get("command") or "")
            if len(cmd) > 80:
                cmd = cmd[:77] + "..."
            exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
            exit_code = exit_match.group(1) if exit_match else "?"
            return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

        if tool_name == "read_file":
            return f"[read_file] read {args.get('path', '?')} from line {args.get('offset', 1)} ({content_len:,} chars)"

        if tool_name == "write_file":
            return f"[write_file] wrote to {args.get('path', '?')} ({content_len:,} chars)"

        if tool_name == "search_files":
            match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
            count = match_count.group(1) if match_count else "?"
            return f"[search_files] {args.get('target', 'content')} search for '{args.get('pattern', '?')}' in {args.get('path', '.')} -> {count} matches"

        if tool_name == "patch":
            return f"[patch] {args.get('mode', 'replace')} in {args.get('path', '?')} ({content_len:,} chars result)"

        if tool_name == "web_search":
            return f"[web_search] query='{args.get('query', '?')}' ({content_len:,} chars result)"

        if tool_name == "web_extract":
            urls = args.get("urls", [])
            first = urls[0] if isinstance(urls, list) and urls else "?"
            return f"[web_extract] {first} ({content_len:,} chars result)"

        if tool_name in ("browser_exec", "computer_use", "vision_analyze"):
            return f"[{tool_name}] {content_len:,} chars result"

        return f"[{tool_name}] {content_len:,} chars result"
    except Exception:
        _len = len(tool_content) if isinstance(tool_content, str) else 0
        return f"[{tool_name}] ({_len:,} chars result)"


def _looks_already_pruned(content) -> bool:
    if not isinstance(content, str) or not content:
        return True
    if content == DUPLICATE_PLACEHOLDER or content.startswith("[Duplicate tool output"):
        return True
    # 1-line demotion summaries: "[tool] ... (N chars)"
    if content.startswith("[") and " chars" in content and len(content) < 400:
        return True
    return False


def truncate_tool_call_args_json(args, head_chars=200):
    """Shrink long string values inside a tool-call arguments JSON blob while
    preserving JSON validity (, ported).

    The function.arguments field is a JSON-encoded string the provider
    validates strictly — slicing it raw produces unterminated strings that
    400 on every subsequent turn. Parse, shrink string leaves, re-serialize.
    Non-JSON arguments (some backends) are returned unchanged.
    """
    if not isinstance(args, str):
        return args
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj):
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    # ensure_ascii=False preserves CJK/emoji instead of bloating with \uXXXX
    return json.dumps(shrunken, ensure_ascii=False)


def _retire_stale_tool_result_images(messages, keep_newest=MAX_KEEP_TOOL_IMAGES):
    """Replace image payloads on older tool results with text placeholders
    . Keeps the newest
    keep_newest image-bearing tool rows; user uploads untouched.
    Mutates the list rows in place; returns rows rewritten. The text passes
    below skip multimodal rows entirely, so without this a screenshot
    payload inside the protected tail rides every request forever."""
    seen = 0
    rewritten = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        c = m.get("content")
        n_imgs = 0
        if isinstance(c, str):
            n_imgs = len(_DATA_URI_RE.findall(c))
        elif isinstance(c, list):
            n_imgs = sum(1 for p in c
                         if isinstance(p, dict) and p.get("type") == "image_url")
        if not n_imgs:
            continue
        seen += 1
        if seen <= max(keep_newest, 0):
            continue
        if isinstance(c, str):
            m["content"] = _DATA_URI_RE.sub("[image]", c)
        elif isinstance(c, list):
            kept = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    kept.append({"type": "image_url", "image_url": {"url": "[image]"}})
                else:
                    kept.append(p)
            m["content"] = kept
        rewritten += 1
    return rewritten


# --------------------------------------------------------------- prompt build
SUMMARIZER_PREAMBLE = (
    "You are a summarization agent creating a context checkpoint. "
    "Treat the conversation turns below as source material for a compact record of prior work. "
    "Produce only the structured summary; do not add a greeting, preamble, or prefix. "
    "NEVER include API keys, tokens, passwords, secrets, credentials, or connection strings in the summary — "
    "replace any that appear with [REDACTED]. Note that credentials were present, but do not preserve their values."
)

PRESETS = {
    # Summary content presets: the machinery (trigger, splice, guards) is preset-agnostic;
    # a preset only shapes the summary text - sections, merge rules, style directives,
    # anchor patterns, and cap eviction order. Adding one = adding a dict entry.
    "Default Compression": {
        "label": "Engineering / agentic work",
        "sections": """## Goal
[The user's overall objective for this conversation, and the active task at the moment of compaction.
If the last exchange was fully resolved, write: None.]

## User Context & Preferences
[User's personality, communication style, preferences, and relationship context revealed in the conversation.
Include things like: what they value, how they like to be addressed, their expertise level, recurring topics
they care about, and any personal context that would help maintain continuity. Do not invent preferences —
only include what was actually expressed or clearly implied.]

## Constraints & Preferences
[Runtime, configuration, and task-specific constraints. Do not invent.]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — working directory, modified/created files, running processes/servers, test status,
anything needed to pick up where this leaves off.]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages. If none, write: None.]

## Key Decisions
[Important decisions made and WHY. If none, write: None.]

## Resolved Questions
[Questions asked and their answers. If none, write: None.]

## Relevant Files
[Files read, modified, or created — with brief note on each. If none, write: None.]

## Critical Context
[Specific values, error messages, configuration details, or data that would be lost without explicit preservation.
NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]""",
        "merge_rules": (
            "ONE HOME PER FACT: each event, error, fix, and decision appears in EXACTLY ONE section - "
            "if Errors & Fixes explains a fix, Key Decisions and Resolved Questions must NOT restate it. "
            "TRUTH-REVISION RULE: when the new turns correct, retract, or supersede a claim stated as fact in "
            "earlier turns, record ONLY the final position as fact; if the prior state is still informative, "
            "tag it (superseded: <old>) - a retracted claim must never survive as current fact, and a user "
            "correction outranks any earlier assistant assertion. COUNTER LINE: if the previous summary "
            "contains a line starting with 'Live session counters (machine-stamped', carry that line through "
            "VERBATIM - its numbers are machine-stamped at fold time and outrank anything inferable from the "
            "turns; never re-estimate, edit, or duplicate it. Keep Completed Actions to the last 15-20 items "
            "or only actions still affecting current work; drop Relevant Files entries no longer referenced. "
            "Continue the Completed Actions numbering. Move in-progress items into Completed Actions when "
            "done. Update Active State to reflect current state. "
            "Carry forward any '## User Messages (verbatim, newest first)' and '## Anchor Index' sections "
            "from the previous summary under the same headings. Remove information only if it is clearly obsolete."
        ),
        "style_directives": (
            "Be CONCRETE - include file paths, command outputs, error messages, line numbers, and specific "
            "values. Avoid vague descriptions like \"made some changes\" - say exactly what changed.\n"
            "PRIORITIZE CONVERSATION FLOW AND PROBLEM-SOLVING NARRATIVE: structure the summary around "
            "what's happening and why, not just what files exist or were read. If debugging or fixing something, "
            "emphasize: the problem -> investigation -> fix -> verification. If exploring or creating, emphasize: "
            "the direction being pursued, what was learned, and where things are heading. Preserve the REASONING "
            "behind decisions and the user's intent, not just the outcomes. Technical details (file paths, "
            "commands, error messages, code snippets) are important for continuity but serve the narrative - "
            "include what's needed to continue the work, not an exhaustive inventory."
        ),
        "anchor_patterns": (
                r"[A-Za-z]:\\[^\s\"<>|:*?]+",
                r"\w+\.(?:py|js|ts|json|ya?ml|toml|md|txt|bat|ps1|log|cfg|ini)\b",
                r"https?://\S+",
                r"\b\w+(?:Error|Exception)\b.{0,120}",),
        "anchor_min_count": 1,
        "counter_heading": "## Active State",
        "verbatim_heading": "## User Messages (verbatim, newest first)",
        "anchor_heading": "## Anchor Index (exact identifiers from the folded region)",
        "eviction_order": ("## Relevant Files",
            "## Resolved Questions",
            "## Key Decisions",
            "## Completed Actions",
            "## Anchor Index (exact identifiers from the folded region)",
            "## Blocked",),
    },
    "Narrative Compression": {
        "label": "Narrative / roleplay storytelling",
        "sections": """## People & Relationships
[Each named person or character: current situation, location, relationships, and what they currently KNOW versus what has been revealed only to the reader. Note who is present now.]

## Open Threads & Promises
[Unresolved plot lines, promises made, foreshadowing planted, open topics and commitments - most recent first. Mark anything the user explicitly asked to keep in mind.]

## World & Circumstances
[The standing facts both sides of the conversation happen in: for stories, setting, system/magic rules, invented terms, factions; for conversation, the user life circumstances - routines, work, people, constraints. Contradictions resolve to the newest canon.]

## Timeline
[Progression in order: key story events, or life events as the user told them. Fragments are fine.]

## Voice & Tone
[The register this relationship speaks in: for stories, POV, tense, prose style, recurring motifs; for conversation, the established rapport - warm, direct, playful - and any style instructions the user gave.]

## Where Things Stand
[The moment of the cut: for stories, location, who is present, the last exchange, what is mid-motion; for conversation, the current mood and what is mid-discussion. If between scenes or topics, write: Between scenes.]""",
        "merge_rules": ("ONE HOME PER FACT: each event, reveal, promise, and standing fact appears in EXACTLY ONE section - if Open Threads & Promises records a promise, Timeline must NOT restate it as an open item. TRUTH-REVISION RULE: when an earlier belief turns out false - a plot reveal, or a changed opinion, plan, or circumstance - record the CURRENT state; tag the outdated belief (superseded: <old>) only if the difference still changes what happens next. A reveal is not an error - do not treat contradictions as bugs. COUNTER LINE: if the previous summary contains a line starting with 'Live session counters (machine-stamped', carry that line through VERBATIM - its numbers are machine-stamped at fold time and outrank anything inferable from the turns; never re-estimate, edit, or duplicate it. Keep Open Threads to items still open at the cut; finished threads move into People & Relationships or drop out. Update Where Things Stand to the moment of the cut. Carry forward any '## User Messages (verbatim, newest first)' and '## Anchor Index' sections from the previous summary under the same headings. Remove information only if it is clearly obsolete."),
        "style_directives": ("PRIORITIZE HUMAN CONTINUITY: structure the summary around the people, what they know and feel, what is in motion, and what has been promised - whether this is a story or a companionship. Quote exact names, places, invented terms, and signature phrases VERBATIM - paraphrasing a name breaks the continuity that follows. Preserve tone and the emotional trajectory of recent exchanges. Omit meta-commentary about the conversation or storytelling process itself."),
        "anchor_patterns": (
            r"\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b",
        ),
        "anchor_min_count": 2,
        "counter_heading": "## Where Things Stand",
        "verbatim_heading": "## User Messages (verbatim, newest first)",
        "anchor_heading": "## Anchor Index (recurring names and terms)",
        "eviction_order": (
            "## Timeline",
            "## World & Circumstances",
            "## Anchor Index (recurring names and terms)",
            "## Open Threads & Promises",
        ),
    },
}
DEFAULT_PRESET = "Default Compression"

# Legacy names that may exist in stored config values; resolved on lookup.
_PRESET_ALIASES = {
    "engineering": "default compression",
    "narrative": "narrative compression",
}



def build_summary_prompt(turns, previous_summary=None, summary_budget=4000, today=None, preset=None):
    preset = preset or PRESETS[DEFAULT_PRESET]
    content = serialize_turns(turns)
    tail = (
        f"\nTarget ~{summary_budget} tokens. " + preset["style_directives"]
    )
    if today:
        tail += (
            f"\nTEMPORAL ANCHORING: The current date is {today}. When an action has already been carried out, "
            "phrase it as a completed, dated, past-tense fact rather than an open instruction. "
            "Never leave a finished action worded as if it still needs doing."
        )
    tail += "\nWrite only the summary body. Do not include any preamble or prefix."

    if previous_summary:
        prev = redact_text(previous_summary.strip())
        if len(prev) > 30000:
            prev = prev[:30000] + "\n...[previous summary truncated]"
        return (
            f"{SUMMARIZER_PREAMBLE}\n\n"
            "You are updating a context compaction summary. A previous compaction produced the summary below. "
            "New conversation turns have occurred since then and need to be incorporated.\n\n"
            f"PREVIOUS SUMMARY:\n{prev}\n\n"
            f"NEW TURNS TO INCORPORATE:\n{content}\n\n"
            "Update the summary using the exact structure below. MERGE, never nest: rewrite the sections from "
            "the previous summary plus the new turns - do NOT copy previous-summary text verbatim into any "
            "section, and do NOT re-log actions the previous summary already recorded; update those entries "
            "instead. " + preset["merge_rules"] + "\n\n"
            f"{preset['sections']}{tail}"
        )
    return (
        f"{SUMMARIZER_PREAMBLE}\n\n"
        "Create a structured checkpoint summary for the conversation after earlier turns are compacted. "
        "The summary should preserve enough detail for continuity without re-reading the original turns.\n\n"
        f"TURNS TO SUMMARIZE:\n{content}\n\n"
        f"Use this exact structure:\n\n{preset['sections']}{tail}"
    )


# ------------------------------------------------------------------- the module
class ContextCompressor(core.module.Module):
    """Automatically compacts the conversation when it grows past a size
    threshold, keeping a recent tail verbatim. Runs on its own; /compact
    forces it immediately."""

    # defaults: conservative values tuned for large local windows
    settings = {
        # Summary preset: selects the handoff template, style directives, and anchor
        # harvest patterns. Built-ins: "Default Compression", "Narrative Compression".
        # Declared as a select so the webui renders a dropdown; option descriptions
        # show beneath the control for the active preset.
        "summary_preset": {
            "type": "select",
            "description": "Handoff style used when the context folds. Default Compression is tuned for coding/debug/research sessions; Narrative Compression for storytelling and roleplay.",
            "default": DEFAULT_PRESET,
            "options": {
                "Default Compression": "For technical sessions - coding, debugging, research. Keeps files, decisions, and errors.",
                "Narrative Compression": "For personal sessions - roleplay, storytelling, relationships. Keeps people, threads, and promises.",
            },
        },
        # Full-compaction TRIGGER. Fires when the API's reported usage
        # (system prompt + conversation + current tool output) reaches
        # threshold * max_context. 0.75 -> ~108k on a 144k window:
        # conversational chats with a big system prompt and big tool
        # reads stay well under it, while genuinely long chats still get
        # their middle summarized long before the window fills. (0.35
        # -> ~50k proved too aggressive: a 25k chat with a ~13k system
        # prompt + one big file read already crossed it.)
        "threshold": 0.75,
        # Protected live-tail budget = max_context * tail_ratio, INDEPENDENT
        # of the trigger threshold. This is the recent conversation the user
        # keeps verbatim (their explicit preference). 0.20 -> ~29k on a 144k
        # window, matching the tail size the old target_ratio-0.26 produced.
        "tail_ratio": 0.20,
        "protect_first_n": 5,
        "protect_last_n": 60,
        # ---- head/tail hardening ( ports) ----
        # Verbatim early-row protection decays after the first fold: rows
        # stranded behind the cutoff are neither sent nor summarized.
        # Set false to restore permanent protect_first_n.
        "decay_head_protection": True,
        # Min chars for a tool result to be pressure-demoted inside the
        # protected tail when it exceeds the soft ceiling.
        "pressure_demote_min_chars": 1500,
        # ---- deterministic summary hardening ( ports) ----
        # Post-generation enrichment: machine-stamped counter line, verbatim
        # user rows, and a regex-harvested anchor index, appended by code.
        # Hard cap on the assembled handoff in chars; iterative updates can
        # snowball across folds. 0 disables.
        "handoff_cap_chars": 40000,
        # hygiene_hard_message_limit (400 by default):
        # a SECOND compaction trigger — message count.
        # OR-logic: _needs_compress = (tokens >= threshold or
        # msg_count >= 400). Keep api.max_messages at/above this so
        # OpenLumara's silent read-time front-chop only kicks in as a
        # last resort, behind compaction (ordering).
        "max_messages": 400,
        # LLM rolling summary (the true micro-compaction): every
        
        # middle is absorbed into a rolling "## Context so far" marker via a
        # small LLM call, and the absorbed span is spliced out. User turns are
        # never absorbed. Defrags the marker when it grows baggy. Fail-open:
        # 3 consecutive failures on one exchange skip it and advance the
        # cursor ( pattern).
        # Token-GATED (not turn-cadence) reclaim of old bulky tool output,
        # running long before the full-compression threshold. Deterministic,
        # zero LLM calls. Two hysteresis gates keep context rewrites
        # episodic: a commit only happens when it reclaims at least
        
        # disarms until the live context regrows a full trigger-sized runway.
        
        "summary_timeout": 300,
        # Temperature for the summarizer LLM call (fold + rolling merge).
        # 0.4 = quiet-confidence band for structured handoffs;
        # 0.1 if you ever want terser/loyaler lists.
        "summary_temperature": 0.4,
        # Reasoning effort for the summarizer call (Maya: none|low|high|max).
        # low = light thinking pass; max burns completion budget before the summary.
        "summary_reasoning_effort": "low",
        "cooldown_seconds": 300,
        "max_consecutive_failures": 3,
        # ---- anti-thrashing ----
        # Pre-LLM feasibility skip: if the compressible middle is below this
        # fraction of the threshold AND a prior ineffective verdict exists,
        # skip the LLM call — such a small middle can't clear the threshold,
        # so deterministic pruning is the only thing that can still help.
        # 0 disables the skip.
        "feasibility_skip_fraction": _FEASIBILITY_SKIP_DEFAULT,
        # Once two consecutive post-compaction verdicts are "ineffective"
        # (the live prompt is still over the threshold), automatic full
        # compaction blocks for this long, then allows ONE recovery probe.
        "anti_thrash_recovery_seconds": _ANTI_THRASH_RECOVERY_DEFAULT,
        # Rolling-absorb gate (Fix #6): don't spend an LLM merge on an
        # exchange smaller than this many tokens — the merge itself costs
        # roughly that much, so it saves nothing and just busts the KV
        # checkpoint. 0 disables the gate.
        "push_notices": True,
    }

    unsafe = False  # settings visible in the WebUI settings page

    def __init__(self, manager, is_user_module=False, channel=None):
        super().__init__(manager, is_user_module, channel)
        self._turns_since_cutoff = {}   # chat_id -> assistant turns since newest cutoff
        self._failures = {}             # chat_id -> consecutive summary failures
        self._bail_until = {}           # chat_id -> epoch s
        self._stats = {}                # chat_id -> counters
        # anti-thrashing state ( /
        # _anti_thrash_recovery_deadline): per-chat verdicts that a full
        # compaction left the live prompt still over threshold, plus the
        # block deadline after the breaker trips.
        self._ineffective = {}          # chat_id -> consecutive ineffective verdicts
        self._thrash_until = {}         # chat_id -> epoch s, auto-compaction blocked until
        # chat_id -> True when the previous turn just compacted and we owe one
        # post-compaction effectiveness verdict (judged on the next turn's real
        # prompt reading, which is the first one built from the compressed context).
        self._verify_compaction = {}

    # ---------------- helpers ----------------
    def _chat_key(self):
        try:
            ch = self.channel.context.chat
            return ch.get("id") or "current"
        except Exception:
            return "current"

    def _log(self, msg):
        self.manager.log("context_compressor", msg)

    def _threshold_tokens(self):
        max_context = int(core.config.get("api", "max_context", default=8192))
        return int(max_context * float(self.config.get("threshold", default=0.75)))

    def _prompt_usage(self, chat):
        """Prompt-only token count for the compaction trigger.

        The threshold is a fraction of the model's *prompt* window, so the
        comparison must be prompt-only. api.py now reports the API's
        prompt_tokens separately (stored as chat["prompt_token_usage"]);
        when that's absent (API without a prompt/completion split, or the
        first turn before any API-reported usage) we fall back to the
        legacy total so behaviour degrades to the old path, never to zero.
        """
        try:
            prompt = int(chat.get("prompt_token_usage", 0) or 0)
            if prompt > 0:
                return prompt
        except Exception:
            pass
        try:
            return int(chat.get("token_usage", 0) or 0)
        except Exception:
            return 0

    def _tail_budget(self):
        """Token budget for the protected live tail: max_context * tail_ratio.

        Decoupled from the compaction threshold on purpose: the threshold is
        the TRIGGER for a full compaction (lowered so it actually fires on
        big-window models), while the tail is the recent conversation the
        user wants kept verbatim — it must not shrink when the threshold
        does."""
        max_context = int(core.config.get("api", "max_context", default=8192))
        return int(max_context * float(self.config.get("tail_ratio", default=0.20)))

    def _summary_budget(self, middle_tokens):
        return max(MIN_SUMMARY_TOKENS,
                   min(int(middle_tokens * SUMMARY_RATIO), SUMMARY_TOKENS_CEILING))

    def _tail_boundary(self, messages):
        """Token-budgeted tail walk (protect_tail_tokens logic).
        Returns the index where the protected tail starts."""
        n = len(messages)
        budget = self._tail_budget()
        min_protect = min(int(self.config.get("protect_last_n", default=60)),
                          n, MAX_TAIL_MESSAGE_FLOOR)
        accumulated = 0
        boundary = n
        for i in range(n - 1, -1, -1):
            t = _msg_tokens(messages[i])
            if accumulated + t > budget and (n - i) >= min_protect:
                boundary = i
                break
            accumulated += t
            boundary = i
        protected = max(n - boundary, min_protect)
        return n - protected

    def _pressure_demote_tail(self, messages, tail_start):
        """Demote bulky tool results inside the protected tail when the tail
        alone exceeds the soft ceiling ( +
        _pressure_demote_tail, #61932). The newest PRESSURE_KEEP_RECENT_MESSAGES
        rows stay verbatim, and tool results in the newest TAIL_KEEP_TOOL_ROUNDS
        rounds are never touched. In-place content swap: row count, roles, and
        object identity are preserved so the tail-anchor relocation in run_full
        still resolves. Returns the number of rows demoted."""
        tail = messages[tail_start:]
        if len(tail) <= PRESSURE_KEEP_RECENT_MESSAGES:
            return 0
        tail_budget = self._tail_budget()
        soft = int(tail_budget * TAIL_SOFT_CEILING_OVERRUN)
        try:
            ctx_len = int(core.config.get("api", "max_context", default=8192))
        except Exception:
            ctx_len = 0
        if ctx_len > 0:
            soft = min(soft, int(ctx_len * TAIL_MAX_CONTEXT_FRACTION))
        soft = max(soft, tail_budget)
        total = sum(_msg_tokens(m) for m in tail)
        if total <= soft:
            return 0
        # newest-first round numbering: an assistant row with tool_calls starts
        # a round; round_of[j] = k means row j belongs to the k-th newest round
        round_of = {}
        r = 0
        for j in range(len(tail) - 1, -1, -1):
            if tail[j].get("role") == "assistant" and tail[j].get("tool_calls"):
                r += 1
            round_of[j] = r
        call_index = {}
        for m in tail:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id") or ""
                        if cid:
                            fn = tc.get("function", {}) or {}
                            call_index[cid] = (fn.get("name", "unknown"), str(fn.get("arguments", "")))
        min_chars = max(0, int(self.config.get("pressure_demote_min_chars", default=1500) or 0))
        demote_end = len(tail) - min(PRESSURE_KEEP_RECENT_MESSAGES, len(tail))
        demoted = 0
        for j in range(0, demote_end):
            if total <= soft:
                break
            m = tail[j]
            if m.get("role") != "tool":
                continue
            if round_of.get(j, 0) <= TAIL_KEEP_TOOL_ROUNDS:
                continue  # inside the newest rounds - keep verbatim
            c = m.get("content")
            if isinstance(c, str):
                if _looks_already_pruned(c) or len(c) < min_chars:
                    continue
                cid = m.get("tool_call_id") or ""
                name, args = call_index.get(cid, ("unknown", ""))
                m["content"] = summarize_tool_result(name, args, c)
                total = sum(_msg_tokens(x) for x in tail)
                demoted += 1
            elif isinstance(c, list):
                kept = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        kept.append({"type": "image_url", "image_url": {"url": "[image]"}})
                    else:
                        kept.append(p)
                m["content"] = kept
                total = sum(_msg_tokens(x) for x in tail)
                demoted += 1
        return demoted

    def _preset(self):
        raw = str(self.config.get("summary_preset", default=DEFAULT_PRESET) or DEFAULT_PRESET)
        name = _PRESET_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        index = {k.strip().lower(): v for k, v in PRESETS.items()}
        return index.get(name, PRESETS[DEFAULT_PRESET])

    def _enrich_summary(self, summary, middle, n_disk):
        """Append machine-stamped counters, verbatim user rows, and an anchor index ( ports). All code-owned; the summarizer never gets a vote on these."""
        preset = self._preset()
        marker = "Live session counters (machine-stamped"
        try:
            usage = int(self._prompt_usage(self.channel.context.chat) or 0)
        except Exception:
            usage = 0
        counter = ("- " + marker + "): ~" + format(usage, ",") + " prompt tokens, "
                   + format(n_disk, ",") + " messages on disk at fold time. This line is "
                   "current truth; never replace it with numbers copied from an earlier handoff.")
        # idempotent: drop any previously stamped line, then stamp fresh
        lines = [ln for ln in summary.split("\n") if marker not in ln]
        idxs = [i for i, ln in enumerate(lines) if ln.strip() == preset["counter_heading"]]
        if idxs:
            lines.insert(idxs[0] + 1, counter)
            summary = "\n".join(lines)
        else:
            summary = summary.rstrip() + "\n\n" + preset["counter_heading"] + "\n" + counter + "\n"
        # verbatim user rows from the folded region, newest first, char-budgeted
        rows = []
        used = 0
        try:
            for m in reversed(middle):
                if m.get("role") != "user":
                    continue
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                if not isinstance(c, str) or not c.strip():
                    continue
                c = c.strip()
                if len(c) > 4000:
                    c = c[:4000].rstrip() + " ...[truncated; full text in on-disk history]"
                rows.append("> " + c.replace("\n", "\n> "))
                used += len(c)
                if used >= 24000:
                    break
        except Exception:
            rows = []
        uhead = preset["verbatim_heading"]
        if rows:
            slines = summary.split("\n")
            hidx = [i for i, ln in enumerate(slines) if ln.strip() == uhead]
            if hidx:
                # merge under the existing heading: new rows first, cap total size
                s2 = hidx[0] + 1
                e2 = s2
                while e2 < len(slines) and not slines[e2].startswith("## "):
                    e2 += 1
                merged = rows + slines[s2:e2]
                total = sum(len(r) for r in merged)
                while total > 24000 and len(merged) > 1:
                    total -= len(merged[-1])
                    merged.pop()
                slines[s2:e2] = merged
                summary = "\n".join(slines)
            else:
                summary = summary.rstrip() + "\n\n" + uhead + "\n" + "\n".join(rows) + "\n"
        # anchor index: regex-harvested exact identifiers, LLM-free
        anchors = []
        try:
            chunks = [m.get("content") for m in middle if isinstance(m.get("content"), str)]
            text = "\n".join(chunks)
            pats = preset["anchor_patterns"]
            min_count = int(preset.get("anchor_min_count", 1))
            have = set()
            for pat in pats:
                matches = [mm.group(0).strip().rstrip(".,;:") for mm in re.finditer(pat, text)]
                matches = [mv for mv in matches if mv]
                if min_count <= 1:
                    picked = matches[:10]
                else:
                    freq = {}
                    order = []
                    for mv in matches:
                        freq[mv] = freq.get(mv, 0) + 1
                        if mv not in order:
                            order.append(mv)
                    picked = [mv for mv in order if freq[mv] >= min_count][:10]
                for mv in picked:
                    if mv not in have:
                        have.add(mv)
                        anchors.append(mv)
        except Exception:
            anchors = []
        ahead = preset["anchor_heading"]
        if anchors:
            slines = summary.split("\n")
            hidx = [i for i, ln in enumerate(slines) if ln.strip() == ahead]
            if hidx:
                s2 = hidx[0] + 1
                e2 = s2
                while e2 < len(slines) and not slines[e2].startswith("## "):
                    e2 += 1
                old = [ln for ln in slines[s2:e2] if ln.startswith("- ")]
                have2 = set(old)
                new = ["- " + a for a in anchors if ("- " + a) not in have2]
                slines[s2:e2] = old + new
                summary = "\n".join(slines)
            else:
                joined = "\n".join("- " + a for a in anchors[:24])
                summary = summary.rstrip() + "\n\n" + ahead + "\n" + joined + "\n"
        return summary

    def _cap_summary_handoff(self, summary, ceiling_chars):
        """Evict whole low-priority sections until the assembled handoff fits the cap ( port). Never touches Goal, Active State, Critical Context, or the verbatim user block."""
        if ceiling_chars <= 0 or len(summary) <= ceiling_chars:
            return summary
        preset = self._preset()
        evict_order = preset["eviction_order"]
        for head in evict_order:
            if len(summary) <= ceiling_chars:
                break
            lines = summary.split("\n")
            idxs = [i for i, ln in enumerate(lines) if ln.strip() == head]
            if not idxs:
                continue
            s = idxs[0]
            e = s + 1
            while e < len(lines) and not lines[e].startswith("## "):
                e += 1
            lines[s:e] = [head, "[dropped at fold: handoff cap; full text in on-disk history]"]
            summary = "\n".join(lines)
        return summary

    def _stats_for(self, key):
        if key not in self._stats:
            self._stats[key] = {"compactions": 0, "failures": 0}
        return self._stats[key]

    # ---------------- core hook ----------------
    async def on_assistant_message(self, content: str):
        """Called by stock core after each assistant message is persisted.
        Checks the fold threshold and, when crossed, runs the full fold.

        Trigger (OR-logic):
          * token trigger:  usage >= threshold * max_context
          * count trigger:  message count >= max_messages
        Guarded by cooldown, the failure breaker, and the anti-thrash
        block - all module-local, so no core patches are needed.
        """
        try:
            ctx = self.channel.context
            messages = await ctx.chat.messages.get()
        except Exception as e:
            self._log(f"could not read chat messages: {e}")
            return

        key = self._chat_key()

        # Anti-thrash verdict: the previous turn just compacted, so the
        # prompt tokens reported for THIS turn are the first real reading
        # of the compressed context. Still over threshold => the
        # incompressible floor (system prompt + tool schemas) is at or
        # above the line; record the strike, two in a row trips the block.
        if self._verify_compaction.pop(key, False):
            usage_now = self._prompt_usage(ctx.chat)
            thr_now = self._threshold_tokens()
            if thr_now > 0 and usage_now > 0:
                if usage_now >= thr_now:
                    n = self._ineffective.get(key, 0) + 1
                    self._ineffective[key] = n
                    self._log(f"compaction did not clear threshold ({usage_now:,} >= "
                              f"{thr_now:,}) - ineffective strike {n}. "
                              f"incompressible prompt floor may already exceed it.")
                    if n >= 2:
                        secs = int(self.config.get("anti_thrash_recovery_seconds",
                                                   default=_ANTI_THRASH_RECOVERY_DEFAULT))
                        self._thrash_until[key] = time.time() + secs
                        self._log(f"anti-thrash breaker tripped - auto full-compaction "
                                  f"blocked for {secs}s")
                else:
                    if self._ineffective.get(key, 0):
                        self._log(f"compaction cleared threshold ({usage_now:,} < "
                                  f"{thr_now:,}) - effectiveness strikes reset")
                    self._ineffective[key] = 0

        # threshold check (every turn; cheap integer compares)
        usage = self._prompt_usage(ctx.chat)
        thr = self._threshold_tokens()
        msg_limit = int(self.config.get("max_messages", default=400))
        count_trigger = msg_limit > 0 and len(messages) >= msg_limit
        if thr <= 0 and not count_trigger:
            return
        if not count_trigger and usage < thr:
            return

        # token-trigger guard: the API-reported usage can be stale right
        # after a fold; re-estimate against the live context build before
        # spending a summarizer call.
        if not count_trigger and usage >= thr:
            try:
                est = await ctx.get_total_tokens()
                if est < thr:
                    self._log(f"usage {usage:,} was over threshold but live context "
                              f"({est:,} tokens) is back under - skipping summarizer call")
                    return
            except Exception:
                pass  # estimation failed - proceed with the API-reported number

        await self.run_full(key)

    async def _rewrite_messages(self, new_list, expected_old_len=None):
        """Replace the head of the message list (micro-compact rewrite).
        expected_old_len guards against a race: if the user sent new messages
        after we read the list, only our original span is rewritten."""
        data = self.channel.context.chat.messages.data
        if expected_old_len is None or len(data) <= expected_old_len:
            data.load([dict(m) for m in new_list])
        else:
            # messages were appended while we were working — keep the newer ones
            newer = [dict(m) for m in data[expected_old_len:]]
            data.load([dict(m) for m in new_list[:expected_old_len]] + newer)
        await self.channel.context.chat.messages.save()

    async def _push(self, text):
        try:
            await self.channel.push(text)
        except Exception:
            pass

    # ---------------- full compaction ----------------
    async def run_full(self, key=None, force=False):
        """Compress the conversation between a protected head and a
        token-budgeted protected tail. Fail-open on any error."""
        key = key or self._chat_key()

        now = time.time()
        if not force and self._bail_until.get(key, 0) > now:
            return
        if not force and self._failures.get(key, 0) >= int(self.config.get("max_consecutive_failures", default=3)):
            return
        # Anti-thrashing block : after two consecutive
        # "ineffective" compactions, automatic compaction is blocked so an
        # incompressible session (system prompt + tool schemas already over
        # threshold) stops re-firing the LLM every turn. A manual
        # /compact now (force=True) bypasses it; after the window elapses we
        # allow ONE recovery probe (counters drop to 1 strike).
        if not force:
            t_until = self._thrash_until.get(key, 0)
            if t_until > now:
                self._log(f"full compaction blocked (anti-thrash) for "
                          f"{int(t_until - now)}s more — deterministic passes still run")
                return
            if t_until > 0:  # window elapsed → one recovery probe, then re-block if ineffective
                self._ineffective[key] = 1
                self._thrash_until.pop(key, None)
                self._log("anti-thrash recovery probe allowed after cooldown")

        try:
            ctx = self.channel.context
            messages = await ctx.chat.messages.get()
        except Exception as e:
            self._log(f"run_full: could not read messages: {e}")
            return

        cutoff = last_cutoff_index(messages)
        if cutoff >= 0 and cutoff + 1 < len(messages) and \
           messages[cutoff + 1].get("role") == "assistant":
            # a summary already sits at the head; the head IS that summary
            head_end = cutoff + 2
        else:
            # Head decay (, #11996): verbatim
            # early-row protection only makes sense BEFORE the first fold.
            # After the cutoff lands behind them those rows are neither sent
            # (context.get() drops pre-cutoff rows) nor summarized - silently
            # stranded on disk. Absorb them into the first summary instead.
            # Set decay_head_protection to false to keep the old behavior.
            if self.config.get("decay_head_protection", default=True):
                head_end = 0
            else:
                head_end = min(int(self.config.get("protect_first_n", default=5)), len(messages))

        # Retire stale base64 tool images beyond the newest MAX_KEEP_TOOL_IMAGES
        # before the tail walk counts tokens, so retired payloads don't count.
        _retire_stale_tool_result_images(messages)

        tail_start = self._tail_boundary(messages)
        if tail_start <= head_end:
            tail_start = min(len(messages), head_end + 1)
        # Boundary alignment : the
        # compressed region must never start inside a tool-result run or end
        # mid tool-call group, or the splice orphans a tool row and the next
        # API call 400s ("No tool call found for function call output ...").
        while head_end < len(messages) and messages[head_end].get("role") == "tool":
            head_end += 1
        check = tail_start - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            tail_start = check  # pull the whole assistant+results group into the summarized region
        if tail_start <= head_end:
            tail_start = min(len(messages), head_end + 1)
        middle = messages[head_end:tail_start]

        # In-tail pressure demotion ( +
        # _pressure_demote_tail): the protected tail may overrun its token
        # budget by whole rows up to a hard share of the window; beyond that,
        # bulky tool results older than the newest TAIL_KEEP_TOOL_ROUNDS rounds
        # are demoted to one-liners. In-place, so the tail-anchor relocation
        # below still resolves.
        demoted = self._pressure_demote_tail(messages, tail_start)
        if demoted:
            self._log("tail pressure: demoted " + str(demoted) +
                      " bulky tool result(s) inside the protected tail")

        # capture the tail-boundary message by identity so _resplit_point can
        # re-locate it after sanitize drops rows (shifting indices)
        tail_anchor = messages[tail_start] if tail_start < len(messages) else None

        if len(middle) < MIN_COMPRESS_MESSAGES:
            return
        middle_chars = sum(len(str(m.get("content") or "")) for m in middle)
        if middle_chars < MIN_COMPRESS_CHARS:
            return

        # Pre-LLM feasibility skip : if the compressible
        # middle is tiny relative to the threshold AND a prior compaction
        # proved incompressible (strike on record), the LLM call can't clear
        # the threshold anyway — skip it and let the deterministic passes
        # carry. Saves a whole re-prefill on a local model.
        thr = self._threshold_tokens()
        skip_frac = float(self.config.get("feasibility_skip_fraction", default=_FEASIBILITY_SKIP_DEFAULT))
        if thr > 0 and skip_frac > 0 and self._ineffective.get(key, 0) >= 1:
            middle_tokens = sum(_msg_tokens(m) for m in middle)
            if middle_tokens < thr * skip_frac:
                self._log(f"feasibility skip: middle ~{middle_tokens:,} tokens "
                          f"(< {skip_frac:.0%} of {thr:,} threshold) with a prior "
                          f"ineffective verdict — skipping summarizer call")
                return

        # build prompt
        try:
            import datetime
            today = datetime.date.today().isoformat()
        except Exception:
            today = None
        prev = previous_summary(messages, cutoff) if cutoff >= 0 else ""
        middle_tokens = sum(_msg_tokens(m) for m in middle)
        prompt = build_summary_prompt(middle, previous_summary=strip_summary_wrapper(prev) or None,
                                      summary_budget=self._summary_budget(middle_tokens),
                                      today=today,
                                      preset=self._preset())

        # output cap scales with the real middle (raises the old flat 4000-token fallback for huge middles)
        summary = await self._summarize(
            prompt,
            max_tokens=min(int(self._summary_budget(middle_tokens)) * 2,
                           int(core.config.get("api", "max_output_tokens", default=8192) or 8192)))
        if not summary:
            f = self._failures.get(key, 0) + 1
            self._failures[key] = f
            st = self._stats_for(key)
            st["failures"] += 1
            self._log(f"summary generation failed ({f} consecutive). original messages kept.")
            if f >= int(self.config.get("max_consecutive_failures", default=3)):
                self._bail_until[key] = now + int(self.config.get("cooldown_seconds", default=300))
                self._log(f"compaction suspended for {int(self.config.get('cooldown_seconds', default=300))}s (failure bail-out)")
            return

        # success: splice the cutoff marker + summary in at tail_start.
        # CRITICAL: it must be inserted BEFORE the protected tail, not
        # appended. context.get() keeps only messages after the newest
        # cutoff — appending would put the "protected" tail BEFORE the
        # cutoff and silently drop the recent conversation from the model's
        # context (it would see only the summary + last message).
        # Splicing at tail_start means: everything older than the tail is
        # summarized, the tail (and anything after it) stays live & verbatim.
        # The full history remains on disk either side, so it's reversible
        # (delete above the dotted line in the WebUI) — same as /compress.
        # Deterministic enrichment : machine-stamped counters,
        # verbatim user rows, anchor index - then the hard handoff cap.
        # Code-owned facts never depend on the summarizer remembering them.
        try:
            summary = self._enrich_summary(summary, middle, len(messages))
        except Exception as e:
            self._log("summary enrichment failed (kept raw summary): " + str(e))
        cap_chars = int(self.config.get("handoff_cap_chars", default=40000) or 0)
        if cap_chars > 0 and len(summary) > cap_chars:
            trimmed = self._cap_summary_handoff(summary, cap_chars)
            self._log("handoff cap: " + format(len(summary), ",") + " -> "
                      + format(len(trimmed), ",") + " chars")
            summary = trimmed
        self._failures[key] = 0
        self._bail_until.pop(key, None)
        # Sanitize the surviving list ( +
        # _merge_adjacent_user_turns): the splice can orphan a tool result
        # (its parent assistant tool_call was summarized away) or leave two
        # plain user turns adjacent. A stray tool row makes the next API call
        # 400, and adjacent user turns break llama.cpp's alternation.
        sanitized = _sanitize_tool_pairs(messages)
        # Re-resolve the splice point against the sanitized list (sanitizing can
        # remove rows ahead of the tail, shifting tail_start).
        tail_start = _resplit_point(sanitized, tail_anchor, tail_start)
        # Merge adjacent user turns in the LIVE TAIL only — that's the part
        # sent to the model after the cutoff, where role alternation matters.
        # The head/middle sit before the cutoff as on-disk history and stay
        # byte-for-byte intact (reversibility), so we don't rewrite them.
        sanitized = (sanitized[:tail_start]
                     + _merge_adjacent_user_turns(sanitized[tail_start:]))
        wrapped = f"{SUMMARY_PREFIX}\n\n{summary.strip()}\n\n{SUMMARY_END_MARKER}"
        sanitized[tail_start:tail_start] = [
            SUMMARIZATION_CUTOFF,
            {"role": "assistant", "content": wrapped,
             "_metadata": {"auto_compaction": True}},
        ]
        await self._rewrite_messages(sanitized, expected_old_len=len(messages))
        # arm the post-compaction effectiveness verdict for the NEXT turn.
        self._verify_compaction[key] = True
        self._turns_since_cutoff[key] = 0
        st = self._stats_for(key)
        st["compactions"] += 1
        reclaimed = middle_chars // 4
        self._log(f"compacted {len(middle)} messages (~{reclaimed:,} tokens) into a summary (chat {key})")
        if self.config.get("push_notices", default=True):
            await self._push(f"📜 Context compacted: {len(middle)} messages summarized. "
                             "Full history kept on disk (dotted line) — delete above it to revert.")

    async def _summarize(self, prompt, max_tokens=None):
        """One chat-completions call to the configured local LLM. Returns str or None.

        ``max_tokens`` overrides the output cap (the rolling-summary micro
        merges use a smaller one, min(1500, max_summary_tokens))."""
        url = core.config.get("api", "url", default="")
        key = core.config.get("api", "key", default="")
        model = core.config.get("model", "name", default="")
        timeout = int(self.config.get("summary_timeout", default=300))
        if not url or "API_URL_HERE" in str(url):
            self._log("no API url configured — cannot summarize")
            return None
        if max_tokens is None:
            max_tokens = min(
                int(self._summary_budget(0)) * 2,
                int(core.config.get("api", "max_output_tokens", default=8192)))
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.get("summary_temperature", default=0.4)),
            "reasoning_effort": str(self.config.get("summary_reasoning_effort", default="low") or "low").strip().lower(),
            "max_completion_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(payload).encode()

        def _do():
            return urllib.request.urlopen(
                urllib.request.Request(str(url).rstrip("/") + "/chat/completions",
                                       data=data, headers=headers),
                timeout=timeout)
        try:
            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, _do), timeout=timeout + 10)
            with resp:
                body = json.loads(resp.read().decode())
            choice = (body.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "")
            text = _THINK_BLOCK_RE.sub("", text or "").strip()
            if not text:
                return None
            # Truncation guard : a length-stopped summary is DISCARDED,
            # never stored - truncation reflects a mid-think cut, and storing it
            # poisons the handoff. Fail closed to the existing failure path.
            if str(choice.get("finish_reason", "")).lower() == "length":
                self._log("summarizer output truncated (finish_reason=length) - discarding; "
                          "raise api.max_output_tokens or shrink the middle")
                return None
            return redact_text(text)
        except Exception as e:
            self._log(f"summarizer call failed: {e}")
            return None

    # ---------------- command ----------------
    # ---------------- command ----------------
    @core.module.command("compact", send_to_ai=False, help={
        "": "show context compressor status and stats",
        "now": "force a full compaction right now",
        "reset": "clear failure counters and bail-outs",
    })
    async def cmd_compact(self, args):
        key = self._chat_key()
        st = self._stats_for(key)
        thr = self._threshold_tokens()
        max_context = int(core.config.get("api", "max_context", default=8192))
        usage = self._prompt_usage(self.channel.context.chat)

        if args and args[0] == "now":
            if not thr:
                return "no threshold configured"
            await self.run_full(key, force=True)
            return "compaction attempted (check chat history for a new dotted line)"
        if args and args[0] == "reset":
            self._failures.pop(key, None)
            self._bail_until.pop(key, None)
            self._thrash_until.pop(key, None)
            self._ineffective[key] = 0
            return "failure counters cleared"

        lines = [
            "CONTEXT COMPRESSOR",
            f"  state:        {'ACTIVE' if thr > 0 else 'DISABLED (threshold<=0)'}",
            f"  usage:        {usage:,} / {max_context:,} tokens ({(usage / max_context * 100) if max_context else 0:.1f}%)",
            f"  threshold:    {thr:,} ({float(self.config.get('threshold', default=0.75)) * 100:.0f}% of context)",
            f"  msg limit:    {int(self.config.get('max_messages', default=400))} messages (OR-trigger)",
            f"  tail budget:  {self._tail_budget():,} tokens (floor {MAX_TAIL_MESSAGE_FLOOR}, cap {int(self.config.get('protect_last_n', default=60))} msgs)",
            f"  head protect: {int(self.config.get('protect_first_n', default=5))} messages",
            f"  preset:       {self._preset().get('label', 'default')}",
            f"  compactions:  {st['compactions']} | failures: {st['failures']}",
        ]
        if self._bail_until.get(key, 0) > time.time():
            lines.append(f"  SUSPENDED for {int(self._bail_until[key] - time.time())}s (failure bail-out - /compact reset to clear)")
        if self._thrash_until.get(key, 0) > time.time():
            lines.append(f"  ANTI-THRASH blocked for {int(self._thrash_until[key] - time.time())}s (manual /compact now still works)")
        lines.append("  revert: delete messages above the newest dotted line (full history is on disk)")
        lines.append("  tip: /compact now forces a full compaction immediately")
        return "\n".join(lines)
