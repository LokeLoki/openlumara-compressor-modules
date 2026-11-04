# OpenLumara Drop-in: Context Compressor + Session Archive

Two modules that replace OpenLumara's default context behavior with a
structure-preserving fold and a searchable per-chat archive. **No core
edits required** — copy the two folders into your `modules/` directory,
enable them in the WebUI, done.

Works on a current-structure OpenLumara install (the `modules/<name>/
module.py` layout). Nothing here touches `core/`.

## context_compressor

Automatic, reversible context compression. Instead of OpenLumara's
"drop the oldest messages" fallback, it folds the middle of the
conversation into a structured, code-verified handoff summary while
keeping the recent conversation **verbatim**.

**How a fold works:**

1. The message list is split into three zones:
   - **head** — first few messages, kept verbatim
   - **tail** — the last `tail_ratio` share of the window (min
     `protect_last_n` messages), kept **word-for-word**
   - **middle** — everything between; the only part summarized
2. When usage crosses `threshold` × `max_context` (or the message
   count crosses `max_messages`), the middle is summarized by a quiet
   side-channel LLM call (direct to your configured API — you never
   see it stream).
3. Code (not the LLM) then bolts on three guarantees:
   - a **machine-stamped counter line** (exact prompt tokens + message
     count at fold time, carried verbatim through every future fold)
   - **verbatim user rows** from the folded region (newest first,
     char-budgeted) — your actual words survive, not a paraphrase
   - an **anchor index** — regex-harvested exact identifiers
4. The result is hard-capped (`handoff_cap_chars`) and spliced in as
   OpenLumara's standard `SUMMARIZATION_CUTOFF` marker + summary pair.

**Fully reversible:** the full history stays on disk. Delete the
messages above the newest dotted line in the WebUI to revert.

**Guards (all module-local):**

- *cooldown* — a pause after every fold, success or failure
- *failure breaker* — `max_consecutive_failures` errors in a row
  suspend auto-compaction until `/compact reset`
- *anti-thrash* — folds that leave the prompt still over threshold
  trip a block, so incompressible sessions (system prompt + tool
  schemas already past the line) stop re-firing the LLM every turn.
  Manual `/compact now` always bypasses.

**Presets** (settings dropdown, renders natively in the WebUI):

- **Default Compression** — engineering/agentic sessions: Goal,
  Completed Actions, Active State, Key Decisions, Errors/Blockers,
  Relevant Files, Critical Context. Anchors: paths, filenames, URLs,
  error strings.
- **Narrative Compression** — storytelling/companionship: People &
  Relationships, Open Threads & Promises, World & Circumstances,
  Timeline, Voice & Tone, Where Things Stand. Anchors: recurring
  capitalized names (≥2 occurrences).

Both presets share the same machinery — a preset only shapes the
summary text, merge rules, anchors, and cap-eviction order.

**Commands:** `/compact` (status), `/compact now` (force a fold),
`/compact reset` (clear failure/anti-thrash counters).

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `summary_preset` | Default Compression | Handoff template (dropdown) |
| `threshold` | 0.75 | Fold when usage ≥ this × `max_context` |
| `tail_ratio` | 0.20 | Share of the window protected verbatim |
| `protect_first_n` | 5 | Head messages kept verbatim |
| `protect_last_n` | 60 | Minimum tail messages kept verbatim |
| `max_messages` | 400 | Count-based fold trigger (OR with tokens) |
| `handoff_cap_chars` | 40000 | Hard cap on the summary; sections evicted in order |
| `summary_temperature` | 0.4 | Summarizer LLM temperature |
| `summary_reasoning_effort` | low | Summarizer reasoning effort |
| `summary_timeout` | 300 | Summarizer call timeout (s) |
| `cooldown_seconds` | 300 | Pause after every fold |
| `max_consecutive_failures` | 3 | Errors before auto-compaction suspends |
| `anti_thrash_recovery_seconds` | 300 | Anti-thrash block duration |
| `feasibility_skip_fraction` | 0.1 | Skip fold if middle is tiny vs threshold after a strike |
| `push_notices` | true | Announce folds in the chat |

Notes:

- Token counting is the standard OpenLumara estimate (chars ÷ 4),
  refined by the API's real prompt-token report whenever available.
- Summaries that get cut off by the output cap are **discarded**, not
  stored (a truncated handoff poisons the next fold).
- Secrets are redacted from summaries before storage.

## session_archive

A per-chat, append-only sidecar archive of every message row, written
as the conversation happens. **Install both modules.** The fold
shrinks what the *model* sees; the archive keeps everything *you*
said and everything it said, forever, searchable. The compressor
demotes old tool results to one-line stubs to fit the window —
without the archive running, those raw outputs are gone for good.
With it, `archive_search` recovers any of them on demand.

- `archive_search` tool: keyword search across the archive
- `archive_stats` tool: sizes and coverage
- Secrets are redacted with the same conservative pattern set
- Backfills the current chat on first run (so day-1 installs get an
  archive immediately)

Storage: `data/session_archive.json` via OpenLumara's standard
StorageDict. No configuration needed; enable and forget.

## Install

1. Copy `modules/context_compressor/` and `modules/session_archive/`
   into your OpenLumara `modules/` directory.
2. Restart OpenLumara (or toggle the modules with `/module`).
3. In the WebUI settings, pick a **summary_preset** and set
   **threshold** to taste (0.75 is a good start on large windows; on
   16k-32k local models consider 0.7 to keep more verbatim tail).

Uninstall: disable the module(s). Existing folds are plain
OpenLumara cutoff markers — stock's own `/compress` and context
builder understand them, so nothing breaks.

## Compatibility

- **OpenLumara:** current module structure, stock `core/` only. The
  compressor hooks `on_assistant_message` (stock-dispatched); no
  core patches, no forks needed.
- **Backends:** any OpenAI-compatible API (llama.cpp, koboldcpp,
  ollama, cloud APIs). The summarizer call uses your existing API
  config.
- Tested with Python 3.11 and 3.12.

## License

Same as OpenLumara (GPL-3.0).
