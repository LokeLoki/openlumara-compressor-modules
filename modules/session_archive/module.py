# ALWAYS import core at the start. very important.
import core
import hashlib
import json
import os
import re
import time

# Per-chat archive of every conversation row, written as a sidecar so that
# compaction (context_compressor) can fold the live context freely without
# the folded turns being lost. "compaction = deferred retrieval,
# not loss" pillar, implemented without touching the compressor.

# Naive secret patterns; kept conservative to avoid false positives.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(sk-[A-Za-z0-9_\-]{16,})"),          # openai-style keys
    re.compile(r"(?i)(xox[bp]-[A-Za-z0-9\-]{10,})"),      # slack tokens
    re.compile(r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,})"),     # github tokens
    re.compile(r"(?i)(bearer\s+[A-Za-z0-9.\-_]{20,})"),   # bearer tokens
    re.compile(r"(?i)(password\s*[:=]\s*\S+)"),           # password assignments
    re.compile(r"(?i)(api[_\-]?key\s*[:=]\s*\S+)"),      # api key assignments
]


def _extract_text(content):
    """Normalize string or multimodal list content into plain searchable text."""
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


def _row_sig(role, content):
    """Compact deterministic signature to avoid false-positive prefix dedup."""
    raw = f"{role}\0{content}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


class SessionArchive(core.module.Module):
    """Archives every conversation row per chat and lets the AI search buried
    history even after context_compressor folds it away. Deferred retrieval:
    compaction loses nothing."""

    category = "Memory"
    unsafe = False

    settings = {
        "enabled": {
            "description": "Archive every user/assistant/tool message per chat.",
            "default": True
        },
        "backfill_on_start": {
            "description": "On first run, ingest existing history for this chat into the archive.",
            "default": True
        },
        "archive_tool_outputs": {
            "description": "Archive tool outputs before context_compressor demotes or folds them.",
            "default": True
        },
        "redact_secrets": {
            "description": "Redact obvious API keys/tokens/passwords before archiving.",
            "default": True
        },
        "inject_pointer": {
            "description": "Tell the AI in the system prompt how many archived rows exist and how to search them.",
            "default": True
        },
        "search_top_n": {
            "type": "number",
            "description": "Max results returned per archive_search call.",
            "default": 20
        },
        "max_archive_chars_per_chat": {
            "type": "number",
            "description": "Per-chat archive cap in chars (oldest rows dropped when exceeded). 0 = unlimited.",
            "default": 2000000
        },
    }

    # -------------------------
    #   LIFECYCLE
    # -------------------------

    def _ensure_store(self):
        """Lazy initializer so hooks called before on_ready never raise AttributeError."""
        if getattr(self, "_store", None) is None:
            self._store = core.storage.StorageDict("session_archive", type="json")
        if getattr(self, "_backfilled", None) is None:
            bf = self._store.get("_backfilled", [])
            self._backfilled = bf if isinstance(bf, list) else []

    async def on_ready(self):
        self._ensure_store()

    async def on_shutdown(self):
        try:
            if getattr(self, "_store", None) is not None:
                self._store.save()
        except Exception:
            pass

    # -------------------------
    #   KEYING / HELPERS
    # -------------------------

    def _chat_key(self):
        """channel:chat_id - same spirit as context_compressor._chat_key."""
        try:
            ch_id = self.channel.context.chat.get("id") or "current"
        except Exception:
            ch_id = "current"
        try:
            ch_name = self.channel.name
        except Exception:
            ch_name = "unknown"
        return ch_name + ":" + str(ch_id)

    def _redact(self, text):
        if not self.config.get("redact_secrets", default=True):
            return text
        for pat in _SECRET_PATTERNS:
            text = pat.sub("[REDACTED]", text)
        return text

    def _enforce_cap(self, msgs):
        cap = int(self.config.get("max_archive_chars_per_chat", default=2000000) or 0)
        if cap > 0:
            total = sum(len(str(m.get("content", ""))) for m in msgs)
            while total > cap and len(msgs) > 1:
                total -= len(str(msgs[0].get("content", "")))
                msgs.pop(0)

    def _append(self, role, content, dedup_recent=False):
        if not self.config.get("enabled", default=True):
            return
        self._ensure_store()
        content = _extract_text(content)
        if not content.strip():
            return
        redacted = self._redact(content)
        key = self._chat_key()
        chat = self._store.get(key, {"messages": []})
        msgs = chat.get("messages", [])
        sig = _row_sig(role, redacted)
        if dedup_recent:
            for existing in msgs[-25:]:
                if existing.get("sig") == sig:
                    return
        msgs.append({
            "role": role,
            "content": redacted,
            "ts": time.time(),
            "sig": sig,
        })
        self._enforce_cap(msgs)
        chat["messages"] = msgs
        self._store[key] = chat
        self._store.save()  # persist immediately - crash-safe, cheap (single json)

    async def _sync_unarchived_tool_rows(self):
        """Capture raw tool outputs from the live chat before context_compressor
        demotes or summarizes them away."""
        if not self.config.get("enabled", default=True):
            return
        if not self.config.get("archive_tool_outputs", default=True):
            return
        try:
            live_msgs = await self.channel.context.chat.messages.get()
        except Exception:
            return
        if not isinstance(live_msgs, list):
            return
        for m in live_msgs[-20:]:
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            c = _extract_text(m.get("content", ""))
            if not c.strip():
                continue
            # Skip already-demoted 1-line placeholders
            if c.startswith("[") and " chars" in c and len(c) < 400:
                continue
            self._append("tool", c, dedup_recent=True)

    # -------------------------
    #   EVENT HANDLERS
    # -------------------------

    async def on_user_message(self, content: str):
        # one-time backfill of pre-existing history, lazily on first message
        # (channel context is guaranteed bound here, unlike on_ready)
        await self._maybe_backfill()
        self._append("user", content)

    async def on_assistant_message(self, content: str):
        await self._sync_unarchived_tool_rows()
        self._append("assistant", content)

    async def on_system_prompt(self):
        """One-line pointer so the AI knows the archive exists."""
        if not self.config.get("inject_pointer", default=True):
            return None
        self._ensure_store()
        key = self._chat_key()
        n = len(self._store.get(key, {"messages": []}).get("messages", []))
        if n == 0:
            return None
        return (
            "[session_archive] " + format(n, ",") + " earlier messages from this conversation "
            "are archived and searchable. If you need details buried before a "
            "context compaction, call archive_search with specific keywords."
        )

    # -------------------------
    #   AI TOOLS
    # -------------------------

    @core.module.tool
    async def archive_search(self, query: str):
        """
        Search messages archived from THIS chat (including ones compacted away).
        Use for details buried before a context compaction.

        Args:
            query: keywords to search for in archived messages
        """
        self._ensure_store()
        key = self._chat_key()
        msgs = self._store.get(key, {"messages": []}).get("messages", [])
        if not msgs:
            return self.result("No archived messages for this chat yet.", success=True)

        terms = [t for t in re.split(r"\s+", (query or "").strip()) if t]
        scored = []
        for i, m in enumerate(msgs):
            c = m.get("content", "")
            if not isinstance(c, str):
                continue
            cl = c.lower()
            hits = 0
            for t in terms:
                if t.lower() in cl:
                    hits += 1
            if hits:
                scored.append((hits, i, c))
        if not scored:
            return self.result("No archived messages matched the query.", success=True)
        scored.sort(key=lambda x: (-x[0], -x[1]))  # most hits first, newest on ties

        top_n = max(1, int(self.config.get("search_top_n", default=20) or 20))
        out = []
        for hits, i, c in scored[:top_n]:
            role = str(msgs[i].get("role", "?"))
            pos = c.lower().find(terms[0].lower()) if terms else -1
            start = max(0, pos - 80)
            prefix = "..." if start > 0 else ""
            snippet = prefix + c[start:start + 300]
            line = "#" + str(i) + " [" + role + "] " + snippet
            out.append(line)
        header = str(len(out)) + " archived message(s) matched.\n"
        return self.result(header + "\n".join(out), success=True)

    @core.module.tool
    async def archive_stats(self):
        """
        Show how many messages are archived for the current chat and the total archived chats.
        """
        self._ensure_store()
        key = self._chat_key()
        n = len(self._store.get(key, {"messages": []}).get("messages", []))
        total_keys = sum(1 for k in self._store.keys() if not str(k).startswith("_"))
        msg = "This chat: " + format(n, ",") + " archived rows. Total archived chats: " + str(total_keys) + "."
        return self.result(msg, success=True)

    # -------------------------
    #   BACKFILL (existing history)
    # -------------------------

    async def _backfill(self, key):
        """One-time ingest of THIS chat's existing history from the live chat context."""
        self._ensure_store()
        if key in self._backfilled:
            return
        self._backfilled.append(key)
        ingested = 0
        try:
            live_msgs = await self.channel.context.chat.messages.get()
            if isinstance(live_msgs, list) and live_msgs:
                chat = self._store.get(key, {"messages": []})
                existing = chat.get("messages", [])
                have = {
                    m.get("sig") or _row_sig(m.get("role", "?"), str(m.get("content", "")))
                    for m in existing
                }
                rows = []
                for m in live_msgs:
                    if not isinstance(m, dict) or m.get("_metadata"):
                        continue
                    role = m.get("role", "?")
                    c = _extract_text(m.get("content", ""))
                    if not c.strip():
                        continue
                    redacted = self._redact(c)
                    sig = _row_sig(role, redacted)
                    if sig in have:
                        continue
                    have.add(sig)
                    rows.append({
                        "role": role,
                        "content": redacted,
                        "ts": m.get("ts") or m.get("timestamp") or 0,
                        "sig": sig,
                    })
                if rows:
                    merged = rows + existing
                    self._enforce_cap(merged)
                    chat["messages"] = merged
                    self._store[key] = chat
                    ingested = len(rows)
                    self.channel.log(self.name, "backfilled " + str(ingested) + " rows for " + key)
        except Exception as e:
            self.channel.log(self.name, "backfill failed (kept going): " + str(e))
        self._store["_backfilled"] = self._backfilled
        self._store.save()

    async def _maybe_backfill(self):
        if not self.config.get("backfill_on_start", default=True):
            return
        self._ensure_store()
        key = self._chat_key()
        if key not in self._backfilled:
            await self._backfill(key)
