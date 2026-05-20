"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses auxiliary model (cheap/fast) to summarize middle turns while
protecting head and tail context.

Improvements over v2:
  - Structured summary template with Resolved/Pending question tracking
  - Filter-safe summarizer preamble that treats prior turns as source material
  - "Remaining Work" replaces "Next Steps" to avoid reading as active instructions
  - Clear separator when summary merges into tail message
  - Iterative summary updates (preserves info across multiple compactions)
  - Token-budget tail protection instead of fixed message count
  - Tool output pruning before LLM summarization (cheap pre-pass)
  - Scaled summary budget (proportional to compressed content)
  - Richer tool call/result detail in summarizer input
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, _is_connection_error
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Minimum tokens for the summary output
_MIN_SUMMARY_TOKENS = 2000
# Proportion of compressed content to allocate for summary
_SUMMARY_RATIO = 0.20
# Absolute ceiling for summary tokens (even on very large context windows)
_SUMMARY_TOKENS_CEILING = 12_000

# Codex Responses streams are more timeout-sensitive than normal chat
# completions.  The live regression that motivated this cap had a 134K-char
# noisy summary input (mostly tool JSON/logs) and requested ~15.6K output
# tokens; the same auth/path passed with a ~55K-char synthetic payload.  Keep
# Codex compression near that proven scale and keep one smaller emergency
# payload available for genuine transient stream/network failures.  If Codex
# still hits the adapter's total stream timeout after local budgeting, use a
# local extractive digest rather than another 120s retry followed by an
# unsummarised context-loss marker.  Non-Codex providers stay on the existing
# richer prompt path.
_CODEX_SUMMARY_INPUT_CHAR_BUDGET = 20_000
_CODEX_EMERGENCY_SUMMARY_INPUT_CHAR_BUDGET = 12_000
_CODEX_SUMMARY_TOKENS_CEILING = 3_000
_CODEX_EMERGENCY_SUMMARY_TOKENS_CEILING = 2_000
_CODEX_CONTENT_MAX = 1_800
_CODEX_CONTENT_HEAD = 1_100
_CODEX_CONTENT_TAIL = 450
_CODEX_TOOL_RESULT_MAX = 850
_CODEX_TOOL_RESULT_HEAD = 500
_CODEX_TOOL_RESULT_TAIL = 220
_CODEX_TOOL_ARGS_MAX = 520
_CODEX_TOOL_ARGS_HEAD = 360
_CODEX_SERIALIZED_HEAD_TURNS = 6
_CODEX_SERIALIZED_TAIL_TURNS = 10

# Placeholder used when pruning old tool results
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
# Flat token cost per attached image part.  Real cost varies by provider and
# dimensions (Anthropic ≈ width×height/750, GPT-4o up to ~1700 for
# high-detail 2048×2048, Gemini 258/tile), but 1600 is a realistic ceiling
# that keeps compression budgeting honest for multi-image conversations.
# Matches Claude Code's IMAGE_TOKEN_ESTIMATE constant.
_IMAGE_TOKEN_ESTIMATE = 1600
# Same figure expressed in the char-budget currency the rest of the
# compressor speaks in.  Used when accumulating message "content length"
# for tail-cut decisions.
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600
_SUMMARY_TRANSIENT_RETRY_DELAYS_SECONDS = (2.0,)


def _content_length_for_budget(raw_content: Any) -> int:
    """Return the effective char-length of a message's content for token budgeting.

    Plain strings: ``len(content)``. Multimodal lists: sum of text-part
    ``len(text)`` plus a flat ``_IMAGE_CHAR_EQUIVALENT`` per image part
    (``image_url`` / ``input_image`` / Anthropic-style ``image``). This
    keeps the compressor from treating a turn with 5 attached images as
    near-zero tokens just because the text part is empty.
    """
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))

    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            # text / input_text / tool_result-with-text / anything else with
            # a text field.  Ignore the raw base64 payload inside image_url
            # dicts — dimensions don't matter, only whether it's an image.
            total += len(p.get("text", "") or "")
    return total


def _content_text_for_contains(content: Any) -> str:
    """Return a best-effort text view of message content.

    Used only for substring checks when we need to know whether we've already
    appended a note to a message. Keeps multimodal lists intact elsewhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """Append or prepend plain text to message content safely.

    Compression sometimes needs to add a note or merge a summary into an
    existing message. Message content may be plain text or a multimodal list of
    blocks, so direct string concatenation is not always safe.
    """
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _strip_image_parts_from_parts(parts: Any) -> Any:
    """Strip image parts from an OpenAI-style content-parts list.

    Returns a new list with image_url / image / input_image parts replaced
    by a text placeholder, or None if the list had no images (callers
    skip the replacement in that case). Used by the compressor to prune
    old computer_use screenshots.
    """
    if not isinstance(parts, list):
        return None
    had_image = False
    out = []
    for part in parts:
        if not isinstance(part, dict):
            out.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image", "image_url", "input_image"}:
            had_image = True
            out.append({"type": "text", "text": "[screenshot removed to save context]"})
        else:
            out.append(part)
    return out if had_image else None


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """Shrink long string values inside a tool-call arguments JSON blob while
    preserving JSON validity.

    The ``function.arguments`` field on a tool call is a JSON-encoded string
    passed through to the LLM provider; downstream providers strictly
    validate it and return a non-retryable 400 when it is not well-formed.
    An earlier implementation sliced the raw JSON at a fixed byte offset and
    appended ``...[truncated]`` — which routinely produced strings like::

        {"path": "/foo/bar", "content": "# long markdown
        ...[truncated]

    i.e. an unterminated string and a missing closing brace. MiniMax, for
    example, rejects this with ``invalid function arguments json string``
    and the session gets stuck re-sending the same broken history on every
    turn. See issue #11762 for the observed loop.

    This helper parses the arguments, shrinks long string leaves inside the
    parsed structure, and re-serialises. Non-string values (paths, ints,
    booleans) are preserved intact. If the arguments are not valid JSON
    to begin with — some model backends use non-JSON tool arguments — the
    original string is returned unchanged rather than replaced with
    something neither we nor the backend can parse.
    """
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
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


def _truncate_middle_text(
    text: str,
    max_chars: int,
    *,
    head_chars: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """Keep the start and end of a long string with a clear omission marker."""
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    if head_chars is None:
        head_chars = max(0, int(max_chars * 0.65))
    if tail_chars is None:
        tail_chars = max(0, max_chars - head_chars - 40)
    head_chars = max(0, min(head_chars, max_chars))
    tail_chars = max(0, min(tail_chars, max(0, max_chars - head_chars - 40)))
    omitted = max(0, len(text) - head_chars - tail_chars)
    marker = f"\n...[truncated {omitted:,} chars]...\n"
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    return text[:head_chars].rstrip() + marker + tail


def _compact_tool_args_for_summary(args: str) -> str:
    """Shrink tool-call args for Codex compression without corrupting JSON."""
    if not args:
        return ""
    redacted = redact_sensitive_text(args)
    if len(redacted) <= _CODEX_TOOL_ARGS_MAX:
        return redacted
    shrunk = _truncate_tool_call_args_json(redacted, head_chars=220)
    if len(shrunk) <= _CODEX_TOOL_ARGS_MAX:
        return shrunk
    return _truncate_middle_text(
        shrunk,
        _CODEX_TOOL_ARGS_MAX,
        head_chars=_CODEX_TOOL_ARGS_HEAD,
        tail_chars=120,
    )


_TOOL_SNIPPET_MARKERS = (
    "traceback", "exception", "error", "failed", "failure", "timeout",
    "warning", "exit_code", "exit code", "status", "stderr", "stdout",
    "pytest", "passed", "skipped", "changed", "modified", "diff",
    "path", "file", "line", "matches", "total_count", "summary",
)


def _salient_tool_output_snippet(text: str, max_chars: int) -> str:
    """Extract high-signal lines from noisy tool output for Codex summaries."""
    if not text:
        return ""
    text = redact_sensitive_text(text)
    if len(text) <= max_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    salient: list[str] = []
    seen: set[str] = set()
    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in _TOOL_SNIPPET_MARKERS):
            clipped = _truncate_middle_text(line, 360, head_chars=250, tail_chars=80)
            if clipped not in seen:
                salient.append(clipped)
                seen.add(clipped)
        if len("\n".join(salient)) >= max_chars:
            break

    if not salient:
        head = lines[:4]
        tail = lines[-3:] if len(lines) > 4 else []
        salient = [*head, *tail]

    snippet = "\n".join(salient)
    return _truncate_middle_text(
        snippet,
        max_chars,
        head_chars=int(max_chars * 0.7),
        tail_chars=int(max_chars * 0.2),
    )


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result.

    Used during the pre-compression pruning pass to replace large tool
    outputs with a short but useful description of what the tool did,
    rather than a generic placeholder that carries zero information.

    Returns strings like::

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in {"browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"}:
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in {"skill_view", "skills_list", "skill_manage"}:
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """Default context engine — compresses conversation context via lossy summarization.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages by token budget (most recent ~20K tokens)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
    """

    @property
    def name(self) -> str:
        return "compressor"

    def on_session_reset(self) -> None:
        """Reset all per-session state for /new or /reset."""
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_summary_input_chars = 0
        self._last_summary_input_chars_original = 0
        self._last_summary_payload_reduced = False
        self._last_summary_budget_tokens = 0
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0  # transient errors must not block a fresh session

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Update model info after a model switch or fallback activation."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = max(
            int(context_length * self.threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        # Recalculate token budgets for the new context length so the
        # compressor stays calibrated after a model switch (e.g. 200K → 32K).
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # Floor: never compress below MINIMUM_CONTEXT_LENGTH tokens even if
        # the percentage would suggest a lower value.  This prevents premature
        # compression on large-context models at 50% while keeping the % sane
        # for models right at the minimum.
        self.threshold_tokens = max(
            int(self.context_length * threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        self.compression_count = 0

        # Derive token budgets: ratio is relative to the threshold, not total context
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        self.summary_model = summary_model_override or ""

        # Stores the previous compaction summary for iterative updates
        self._previous_summary: Optional[str] = None
        # Anti-thrashing: track whether last compression was effective
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0
        self._last_summary_error: Optional[str] = None
        # When summary generation fails and a static fallback is inserted,
        # record how many turns were unrecoverably dropped so callers
        # (gateway hygiene, /compress) can surface a visible warning.
        self._last_summary_dropped_count: int = 0
        self._last_summary_fallback_used: bool = False
        # When a user-configured summary model fails and we recover by
        # retrying on the main model, record the failure so gateway /
        # CLI callers can still warn the user even though compression
        # succeeded.  Silent recovery would hide the broken config.
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None
        self._last_summary_input_chars: int = 0
        self._last_summary_input_chars_original: int = 0
        self._last_summary_payload_reduced: bool = False
        self._last_summary_budget_tokens: int = 0

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold.

        Includes anti-thrashing protection: if the last two compressions
        each saved less than 10%, skip compression to avoid infinite loops
        where each pass removes only 1-2 messages.
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each. "
                    "Consider /new to start a fresh session, or /compress <topic> "
                    "for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Tool output pruning (cheap pre-pass, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Replace old tool result contents with informative 1-line summaries.

        Instead of a generic placeholder, generates a summary like::

            [terminal] ran `npm test` -> exit 0, 47 lines output
            [read_file] read config.py from line 1 (3,400 chars)

        Also deduplicates identical tool results (e.g. reading the same file
        5x keeps only the newest full copy) and truncates large tool_call
        arguments in assistant messages outside the protected tail.

        Walks backward from the end, protecting the most recent messages that
        fall within ``protect_tail_tokens`` (when provided) OR the last
        ``protect_tail_count`` messages (backward-compatible default).
        When both are given, the token budget takes priority and the message
        count acts as a hard minimum floor.

        Returns (pruned_messages, pruned_count).
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # Determine the prune boundary
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # Token-budget approach: walk backward accumulating tokens
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result))
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                raw_content = msg.get("content") or ""
                content_len = _content_length_for_budget(raw_content)
                msg_tokens = content_len // _CHARS_PER_TOKEN + 10
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        msg_tokens += len(args) // _CHARS_PER_TOKEN
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            # Translate the budget walk into a "protected count", apply the
            # floor in count-space (where `max` reads naturally: protect at
            # least `min_protect` messages or whatever the budget reserved,
            # whichever is more), then convert back to a prune boundary.
            # Doing this in index-space with `max` would invert the direction
            # (smaller index = MORE protected), so a generous budget would
            # silently get truncated back down to `min_protect`.
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        # Pass 1: Deduplicate identical tool results.
        # When the same file is read multiple times, keep only the most recent
        # full copy and replace older duplicates with a back-reference.
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Multimodal content — dedupe by the text summary if available.
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                # Multimodal dict envelopes ({_multimodal: True, content: [...]}) and
                # other non-string tool-result shapes can't be hashed/deduped by text.
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # This is an older duplicate — replace with back-reference
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Multimodal content (base64 screenshots etc.): strip the image
            # payload — keep a lightweight text placeholder in its place.
            # Without this, an old computer_use screenshot (~1MB base64 +
            # ~1500 real tokens) survives every compression pass forever.
            if isinstance(content, list):
                stripped = _strip_image_parts_from_parts(content)
                if stripped is not None:
                    result[i] = {**msg, "content": stripped}
                    pruned += 1
                continue
            if isinstance(content, dict) and content.get("_multimodal"):
                summary = content.get("text_summary") or "[screenshot removed to save context]"
                result[i] = {**msg, "content": f"[screenshot removed] {summary[:200]}"}
                pruned += 1
                continue
            if not isinstance(content, str):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # Skip already-deduplicated or previously-summarized results
            if content.startswith("[Duplicate tool output"):
                continue
            # Only prune if the content is substantial (>200 chars)
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # Pass 3: Truncate large tool_call arguments in assistant messages
        # outside the protected tail. write_file with 50KB content, for
        # example, survives pruning entirely without this.
        #
        # The shrinking is done inside the parsed JSON structure so the
        # result remains valid JSON — otherwise downstream providers 400
        # on every subsequent turn until the broken call falls out of
        # the window. See ``_truncate_tool_call_args_json`` docstring.
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_transient_summary_error(exc: BaseException) -> bool:
        """Return True for summary-call failures worth retrying immediately.

        Auxiliary compression is often routed through streaming providers.  A
        stream can fail after the provider has accepted the request (for example
        ``peer closed connection ... incomplete chunked read``).  Retrying these
        once is much safer than dropping the middle turns and inserting a
        fallback marker.  Configuration/validation failures should not be
        retried because the same request will fail again and delay the turn.
        """
        err_text = str(exc).lower()
        if "codex auxiliary responses stream exceeded" in err_text:
            # This is the adapter's enforced total wall-clock timeout, not a
            # quick network blip. Retrying another full stream has repeatedly
            # produced a second 120s stall followed by an unsummarised fallback;
            # let the Codex-specific fallback path build a local digest instead.
            return False

        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}:
            return True
        transient_markers = (
            "peer closed connection",
            "incomplete chunked read",
            "chunked read",
            "server disconnected",
            "stream interrupted",
            "connection reset",
            "connection aborted",
            "connection error",
            "read timeout",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "try again",
            "rate limit",
            "overloaded",
            "network",
        )
        return any(marker in err_text for marker in transient_markers)

    def _call_summary_llm_with_retries(
        self,
        call_kwargs: Dict[str, Any],
        *,
        retry_call_kwargs: Dict[str, Any] | None = None,
        retry_description: str = "same payload",
    ):
        """Call the summary LLM, retrying transient streaming/network errors.

        Raises the final exception to the existing fallback/cooldown logic if
        every attempt fails.  RuntimeError keeps the old behaviour because this
        path historically means no auxiliary provider is configured.

        ``retry_call_kwargs`` lets timeout-sensitive routes retry with a smaller
        locally-generated payload instead of spending the retry budget on the
        same request shape that already exceeded the stream timeout.
        """
        max_attempts = 1 + len(_SUMMARY_TRANSIENT_RETRY_DELAYS_SECONDS)
        explicit_model = call_kwargs.get("model")
        main_model = (call_kwargs.get("main_runtime") or {}).get("model")
        # If a configured auxiliary summary model is failing, let the outer
        # fallback path retry once on the main model instead of spending the
        # retry budget on the same auxiliary model.  Main-model summary calls
        # still get the upstream same-model transient retry below.
        using_aux_model = bool(explicit_model and main_model and explicit_model != main_model)
        attempt = 0
        while True:
            attempt += 1
            try:
                return call_llm(**call_kwargs)
            except RuntimeError:
                raise
            except Exception as exc:
                can_retry = (
                    not using_aux_model
                    and attempt < max_attempts
                    and self._is_transient_summary_error(exc)
                )
                if not can_retry:
                    raise
                delay = _SUMMARY_TRANSIENT_RETRY_DELAYS_SECONDS[attempt - 1]
                err_text = str(exc).strip() or exc.__class__.__name__
                if len(err_text) > 220:
                    err_text = err_text[:217].rstrip() + "..."
                if retry_call_kwargs is not None and attempt == 1:
                    next_kwargs = retry_call_kwargs
                    retry_note = retry_description
                else:
                    next_kwargs = call_kwargs
                    retry_note = "same payload"
                logger.info(
                    "Context summary attempt %d/%d failed with transient error; "
                    "retrying in %.1fs with %s: %s",
                    attempt,
                    max_attempts,
                    delay,
                    retry_note,
                    err_text,
                )
                if delay > 0:
                    time.sleep(delay)
                call_kwargs = next_kwargs

    def _compute_summary_budget(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        *,
        codex_responses: bool = False,
    ) -> int:
        """Scale summary token budget with the amount of content being compressed.

        The maximum scales with the model's context window (5% of context,
        capped at ``_SUMMARY_TOKENS_CEILING``) so large-context models get
        richer summaries instead of being hard-capped at 8K tokens.  Codex
        Responses gets a smaller ceiling because long output budgets materially
        increase stream duration and caused real compression timeouts.
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        ceiling = min(self.max_summary_tokens, _CODEX_SUMMARY_TOKENS_CEILING) if codex_responses else self.max_summary_tokens
        return max(_MIN_SUMMARY_TOKENS, min(budget, ceiling))

    # Truncation limits for the summarizer input.  These bound how much of
    # each message the summary model sees — the budget is the *summary*
    # model's context window, not the main model's.
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _summary_uses_codex_responses(self) -> bool:
        """Best-effort detection for the Codex Responses compression path."""
        provider = (self.provider or "").strip().lower()
        api_mode = (self.api_mode or "").strip().lower()
        base_url = (self.base_url or "").strip().lower()
        if provider == "openai-codex" or api_mode == "codex_responses" or "backend-api/codex" in base_url:
            return True
        # If this compressor was constructed without explicit runtime details,
        # fall back to the configured auxiliary.compression provider.  This
        # keeps the cap active when compression is explicitly routed to Codex
        # even if the main model is not Codex.  Unit tests that need non-Codex
        # behaviour pass a concrete non-Codex provider.
        if provider or api_mode or base_url:
            return False
        try:
            from agent.auxiliary_client import _resolve_task_provider_model
            cfg_provider, _, cfg_base_url, _, cfg_api_mode = _resolve_task_provider_model("compression")
        except Exception:
            return False
        cfg_provider = (cfg_provider or "").strip().lower()
        cfg_api_mode = (cfg_api_mode or "").strip().lower()
        cfg_base_url = (cfg_base_url or "").strip().lower()
        return (
            cfg_provider == "openai-codex"
            or cfg_api_mode == "codex_responses"
            or "backend-api/codex" in cfg_base_url
        )

    def _tool_call_lookup(self, turns: List[Dict[str, Any]]) -> Dict[str, tuple[str, str]]:
        """Map tool_call_id -> (tool_name, arguments_json) for serializer hints."""
        lookup: Dict[str, tuple[str, str]] = {}
        for msg in turns:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("id", "") or tc.get("call_id", "")
                    fn = tc.get("function", {}) or {}
                    lookup[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                else:
                    cid = getattr(tc, "id", "") or getattr(tc, "call_id", "") or ""
                    fn = getattr(tc, "function", None)
                    lookup[cid] = (
                        getattr(fn, "name", "unknown") if fn else "unknown",
                        getattr(fn, "arguments", "") if fn else "",
                    )
        return lookup

    def _serialize_for_summary(
        self,
        turns: List[Dict[str, Any]],
        *,
        compact_for_codex: bool = False,
        budget_chars: int | None = None,
    ) -> str:
        """Serialize conversation turns into labeled text for the summarizer.

        Includes tool call arguments and result content (up to
        ``_CONTENT_MAX`` chars per message) so the summarizer can preserve
        specific details like file paths, commands, and outputs.

        All content is redacted before serialization to prevent secrets
        (API keys, tokens, passwords) from leaking into the summary that
        gets sent to the auxiliary model and persisted across compactions.
        """
        tool_lookup = self._tool_call_lookup(turns) if compact_for_codex else {}
        parts = []
        for idx, msg in enumerate(turns):
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")

            # Tool results: keep enough content for the summarizer.  In Codex
            # mode, replace raw JSON/log/skill-doc blobs with a local concise
            # description plus high-signal snippets before the LLM call.
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if compact_for_codex:
                    tool_name, tool_args = tool_lookup.get(tool_id, ("unknown", ""))
                    summary = _summarize_tool_result(tool_name, tool_args, content)
                    snippet = _salient_tool_output_snippet(content, _CODEX_TOOL_RESULT_MAX)
                    if snippet and snippet != content:
                        content = f"{summary}\nSalient output:\n{snippet}"
                    else:
                        content = f"{summary}\n{snippet}" if snippet else summary
                    content = _truncate_middle_text(
                        content,
                        _CODEX_TOOL_RESULT_MAX,
                        head_chars=_CODEX_TOOL_RESULT_HEAD,
                        tail_chars=_CODEX_TOOL_RESULT_TAIL,
                    )
                    parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                    continue
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # Assistant messages: include tool call names AND arguments
            if role == "assistant":
                if compact_for_codex:
                    content = _truncate_middle_text(
                        content,
                        _CODEX_CONTENT_MAX,
                        head_chars=_CODEX_CONTENT_HEAD,
                        tail_chars=_CODEX_CONTENT_TAIL,
                    )
                elif len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            # Truncate long arguments but keep enough for context
                            if compact_for_codex:
                                args = _compact_tool_args_for_summary(args)
                            else:
                                args = redact_sensitive_text(args)
                                if len(args) > self._TOOL_ARGS_MAX:
                                    args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT #{idx + 1}]: {content}" if compact_for_codex else f"[ASSISTANT]: {content}")
                continue

            # User and other roles
            if compact_for_codex:
                content = _truncate_middle_text(
                    content,
                    _CODEX_CONTENT_MAX,
                    head_chars=_CODEX_CONTENT_HEAD,
                    tail_chars=_CODEX_CONTENT_TAIL,
                )
                parts.append(f"[{role.upper()} #{idx + 1}]: {content}")
                continue
            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        if compact_for_codex and budget_chars and len("\n\n".join(parts)) > budget_chars:
            parts = self._fit_serialized_summary_parts(parts, budget_chars)

        return "\n\n".join(parts)

    def _fit_serialized_summary_parts(self, parts: List[str], budget_chars: int) -> List[str]:
        """Fit serialized Codex summary parts inside a total character budget."""
        if len("\n\n".join(parts)) <= budget_chars:
            return parts
        n = len(parts)
        head_n = min(_CODEX_SERIALIZED_HEAD_TURNS, n)
        tail_n = min(_CODEX_SERIALIZED_TAIL_TURNS, max(0, n - head_n))
        head = list(parts[:head_n])
        tail = list(parts[n - tail_n:]) if tail_n else []
        middle = parts[head_n:n - tail_n] if tail_n else parts[head_n:]

        def joined_len(items: List[str]) -> int:
            return len("\n\n".join(items))

        def digest_part(part: str) -> str:
            lines = [line.strip() for line in part.splitlines() if line.strip()]
            if not lines:
                return "- [empty turn]"
            kept: list[str] = [lines[0]]
            for line in lines[1:]:
                lower = line.lower()
                if line.startswith("  ") or any(marker in lower for marker in _TOOL_SNIPPET_MARKERS):
                    kept.append(line)
                if len(kept) >= 4:
                    break
            return "- " + _truncate_middle_text(" | ".join(kept), 320, head_chars=230, tail_chars=60)

        digest_lines = [digest_part(part) for part in middle]
        digest = (
            f"[... {len(middle)} middle turn(s) locally compressed to fit Codex compression budget ...]\n"
            + "\n".join(digest_lines)
        ) if middle else ""
        selected = [*head, digest, *tail] if digest else [*head, *tail]

        if joined_len(selected) <= budget_chars:
            return selected

        fixed_len = joined_len([*head, *tail])
        digest_budget = max(1_200, budget_chars - fixed_len - 500)
        if digest:
            digest = _truncate_middle_text(
                digest,
                digest_budget,
                head_chars=int(digest_budget * 0.72),
                tail_chars=int(digest_budget * 0.18),
            )
            selected = [*head, digest, *tail]
        if joined_len(selected) <= budget_chars:
            return selected

        # If the preserved head/tail alone still exceed the budget, shrink the
        # largest preserved entries one by one, keeping role/tool labels and
        # first/last snippets rather than dropping active context wholesale.
        shrunk = list(selected)
        while joined_len(shrunk) > budget_chars and shrunk:
            largest_idx = max(range(len(shrunk)), key=lambda i: len(shrunk[i]))
            current = shrunk[largest_idx]
            target = max(600, int(len(current) * 0.65))
            if len(current) <= target + 20:
                target = max(300, len(current) - 300)
            shrunk[largest_idx] = _truncate_middle_text(current, target)
            if target <= 300:
                break
        if joined_len(shrunk) > budget_chars:
            return [_truncate_middle_text("\n\n".join(shrunk), budget_chars)]
        return shrunk

    def _fallback_to_main_for_compression(self, e: Exception, reason: str) -> None:
        """Switch from a separate ``summary_model`` back to the main model.

        Centralises the bookkeeping shared by every fallback branch in
        :meth:`_generate_summary` (model-not-found, timeout, JSON decode,
        unknown error): record the aux-model failure for ``/usage``-style
        callers, clear the summary model so the next call uses the main one,
        and clear the cooldown so the immediate retry can run.

        ``reason`` is a short human-readable phrase ("unavailable",
        "timed out", "returned invalid JSON", "failed") that is interpolated
        into the warning log.
        """
        self._summary_model_fallen_back = True
        logging.warning(
            "Summary model '%s' %s (%s). "
            "Falling back to main model '%s' for compression.",
            self.summary_model, reason, e, self.model,
        )
        _err_text = str(e).strip() or e.__class__.__name__
        if len(_err_text) > 220:
            _err_text = _err_text[:217].rstrip() + "..."
        self._last_aux_model_failure_error = _err_text
        self._last_aux_model_failure_model = self.summary_model
        self.summary_model = ""  # empty = use main model
        self._summary_failure_cooldown_until = 0.0  # no cooldown — retry immediately

    def _generate_local_timeout_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        *,
        reason: str,
        focus_topic: Optional[str] = None,
    ) -> str:
        """Build an extractive continuity summary when Codex summary times out.

        This is deliberately not a generic fallback for all providers.  It is a
        last-resort Codex timeout path that preserves a compact, redacted digest
        of the removed turns instead of inserting the unhelpful
        "Summary generation was unavailable" marker.
        """
        digest = self._serialize_for_summary(
            turns_to_summarize,
            compact_for_codex=True,
            budget_chars=_CODEX_EMERGENCY_SUMMARY_INPUT_CHAR_BUDGET,
        )
        latest_compacted_user = "None found in the compacted region."
        for msg in reversed(turns_to_summarize):
            if msg.get("role") == "user":
                latest_compacted_user = _truncate_middle_text(
                    redact_sensitive_text(_content_text_for_contains(msg.get("content"))),
                    1_200,
                    head_chars=800,
                    tail_chars=280,
                ) or latest_compacted_user
                break

        role_counts: Dict[str, int] = {}
        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        role_summary = ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items())) or "none"
        reason_text = _truncate_middle_text(str(reason or "Codex auxiliary summary timed out"), 240)
        focus_line = f"\n- Requested compression focus: {focus_topic}" if focus_topic else ""

        body = f"""## Active Task
The current active task should be taken from the preserved messages after this summary. Most recent compacted user turn: {latest_compacted_user}

## Goal
Preserve continuity after the Codex auxiliary compression summary timed out.

## Constraints & Preferences
- This is a local extractive digest generated by Hermes because the auxiliary Codex summarizer exceeded its stream timeout.
- It is source material only; the preserved messages after this summary remain authoritative for the live next action.{focus_line}

## Completed Actions
Local digest of compacted turns follows under Critical Context. It keeps role labels, tool names, commands/paths/status/error snippets, and first/last high-signal excerpts where available.

## Active State
- Compacted turns preserved in digest: {len(turns_to_summarize)} ({role_summary}).
- Auxiliary summary failure: {reason_text}.

## In Progress
Unknown from local digest alone; inspect the preserved tail messages after this summary for current in-progress work.

## Blocked
- Auxiliary Codex summary timed out before producing a model-written summary.

## Key Decisions
- Hermes used a local extractive digest instead of dropping the compacted turns or inserting an unsummarised fallback marker.

## Resolved Questions
See Critical Context digest.

## Pending User Asks
See the preserved messages after this summary for the active ask; compacted asks are quoted in Critical Context when present.

## Relevant Files
See file paths and tool-output snippets in Critical Context.

## Remaining Work
Continue from the preserved tail messages and use the digest below only for prior context.

## Critical Context
{digest}"""
        summary = redact_sensitive_text(body.strip())
        self._previous_summary = summary
        self._summary_failure_cooldown_until = 0.0
        self._summary_model_fallen_back = False
        self._last_summary_error = None
        return self._with_summary_prefix(summary)

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]], focus_topic: str = None) -> Optional[str]:
        """Generate a structured summary of conversation turns.

        Uses a structured template (Goal, Progress, Decisions, Resolved/Pending
        Questions, Files, Remaining Work) with explicit preamble telling the
        summarizer not to answer questions.  When a previous summary exists,
        generates an iterative update instead of summarizing from scratch.

        Args:
            focus_topic: Optional focus string for guided compression.  When
                provided, the summariser prioritises preserving information
                related to this topic and is more aggressive about compressing
                everything else.  Inspired by Claude Code's ``/compact``.

        Returns None if all attempts fail — the caller should drop
        the middle turns without a summary rather than inject a useless
        placeholder.
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        codex_responses = self._summary_uses_codex_responses()
        summary_budget = self._compute_summary_budget(
            turns_to_summarize,
            codex_responses=codex_responses,
        )
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        original_summary_chars = len(content_to_summarize)
        if codex_responses and original_summary_chars > _CODEX_SUMMARY_INPUT_CHAR_BUDGET:
            content_to_summarize = self._serialize_for_summary(
                turns_to_summarize,
                compact_for_codex=True,
                budget_chars=_CODEX_SUMMARY_INPUT_CHAR_BUDGET,
            )
            if not self.quiet_mode:
                logger.info(
                    "Codex compression payload reduced: %d -> %d chars; summary_budget=%d tokens",
                    original_summary_chars,
                    len(content_to_summarize),
                    summary_budget,
                )
        self._last_summary_input_chars_original = original_summary_chars
        self._last_summary_input_chars = len(content_to_summarize)
        self._last_summary_payload_reduced = len(content_to_summarize) < original_summary_chars
        self._last_summary_budget_tokens = summary_budget

        # Preamble shared by both first-compaction and iterative-update prompts.
        # Keep the wording deliberately plain: Azure/OpenAI-compatible content
        # filters have flagged stronger "injection" / "do not respond" framing.
        _summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Treat the conversation turns below as source material for a "
            "compact record of prior work. "
            "Produce only the structured summary; do not add a greeting, "
            "preamble, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # Shared structured template (used by both paths).  Build it lazily so
        # timeout-prone Codex retries can ask for a smaller emergency summary
        # without duplicating the whole prompt body.
        def _build_template_sections(target_tokens: int) -> str:
            return f"""## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Copy the user's most recent request or
task assignment verbatim — the exact words they used. If multiple tasks
were requested and only some are done, list only the ones NOT yet completed.
The next assistant/continuation must pick up exactly here. If the conversation
says the current thread/session is a parent/controller, strategy lane, or manual
child workflow, preserve that routing explicitly. For Child Tasks/child-topic/
manual child workflows, frame heavy work as the parent deciding/writing the next
manual child prompt or interpreting a returned child summary — not as permission
for parent execution. Do not collapse instructions like "Use Child Tasks" into
an active parent task such as build/deploy/audit/QA. Example:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{target_tokens} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.

Write only the summary body. Do not include any preamble or prefix."""

        def _build_summary_prompt(serialized_turns: str, target_tokens: int) -> str:
            template_sections = _build_template_sections(target_tokens)
            if self._previous_summary:
                # Iterative update: preserve existing info, add new progress
                built_prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{serialized_turns}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled request — this is the most important field for task continuity.

{template_sections}"""
            else:
                # First compaction: summarize from scratch
                built_prompt = f"""{_summarizer_preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{serialized_turns}

Use this exact structure:

{template_sections}"""

            # Inject focus topic guidance when the user provides one via /compress <focus>.
            # This goes at the end of the prompt so it takes precedence.
            if focus_topic:
                built_prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""
            return built_prompt

        prompt = _build_summary_prompt(content_to_summarize, summary_budget)
        emergency_prompt = None
        emergency_summary_budget = max(
            _MIN_SUMMARY_TOKENS,
            min(summary_budget, _CODEX_EMERGENCY_SUMMARY_TOKENS_CEILING),
        )
        if codex_responses:
            emergency_content = self._serialize_for_summary(
                turns_to_summarize,
                compact_for_codex=True,
                budget_chars=_CODEX_EMERGENCY_SUMMARY_INPUT_CHAR_BUDGET,
            )
            if len(emergency_content) < len(content_to_summarize):
                emergency_prompt = _build_summary_prompt(
                    emergency_content,
                    emergency_summary_budget,
                )

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(summary_budget * 1.3),
                # timeout resolved from auxiliary.compression.timeout config by call_llm
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            retry_call_kwargs = None
            if emergency_prompt:
                retry_call_kwargs = {
                    **call_kwargs,
                    "messages": [{"role": "user", "content": emergency_prompt}],
                    "max_tokens": int(emergency_summary_budget * 1.3),
                }
            response = self._call_summary_llm_with_retries(
                call_kwargs,
                retry_call_kwargs=retry_call_kwargs,
                retry_description="emergency Codex summary payload",
            )
            content = response.choices[0].message.content
            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""
            # Redact the summary output as well — the summarizer LLM may
            # ignore prompt instructions and echo back secrets verbatim.
            summary = redact_sensitive_text(content.strip())
            # Store for iterative updates on next compaction
            self._previous_summary = summary
            self._summary_failure_cooldown_until = 0.0
            self._summary_model_fallen_back = False
            self._last_summary_error = None
            return self._with_summary_prefix(summary)
        except RuntimeError:
            # No provider configured — long cooldown, unlikely to self-resolve
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            self._last_summary_error = "no auxiliary LLM provider configured"
            logging.warning("Context compression: no provider available for "
                            "summary. Middle turns will be dropped without summary "
                            "for %d seconds.",
                            _SUMMARY_FAILURE_COOLDOWN_SECONDS)
            return None
        except Exception as e:
            # If the summary model is different from the main model and the
            # error looks permanent (model not found, 503, 404), fall back to
            # using the main model instead of entering cooldown that leaves
            # context growing unbounded.  (#8620 sub-issue 4)
            _status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            _err_str = str(e).lower()
            _is_model_not_found = (
                _status in {404, 503}
                or "model_not_found" in _err_str
                or "does not exist" in _err_str
                or "no available channel" in _err_str
            )
            _is_timeout = (
                _status in {408, 429, 502, 504}
                or "timeout" in _err_str
            )
            # Non-JSON / malformed-body responses from misconfigured providers
            # or proxies (e.g. an HTML 502 page returned with
            # ``Content-Type: application/json``) bubble up as
            # ``json.JSONDecodeError`` from the OpenAI SDK's ``response.json()``,
            # or as a wrapping ``APIResponseValidationError`` whose message
            # carries the substring "expecting value".  Treat these like a
            # transient provider failure: one retry on the main model, then a
            # short cooldown.  Issue #22244.
            _is_json_decode = (
                isinstance(e, json.JSONDecodeError)
                or "expecting value" in _err_str
            )
            # httpcore / httpx streaming premature-close errors surface as
            # ConnectionError subclasses or plain Exception with characteristic
            # substrings ("incomplete chunked read", "peer closed connection",
            # "response ended prematurely", "unexpected eof").  These are
            # transient network events; treat them like a timeout so we fall
            # back to the main model instead of entering a 60-second cooldown.
            # See issue #18458.
            _is_streaming_closed = _is_connection_error(e)
            _is_codex_total_timeout = "codex auxiliary responses stream exceeded" in _err_str
            if codex_responses and _is_codex_total_timeout:
                err_text = str(e).strip() or e.__class__.__name__
                if len(err_text) > 220:
                    err_text = err_text[:217].rstrip() + "..."
                self._last_aux_model_failure_error = err_text
                self._last_aux_model_failure_model = self.summary_model or self.model
                logging.warning(
                    "Codex compression summary timed out after local payload budgeting; "
                    "using local extractive continuity digest instead of an unsummarised fallback marker: %s",
                    err_text,
                )
                return self._generate_local_timeout_summary(
                    turns_to_summarize,
                    reason=err_text,
                    focus_topic=focus_topic,
                )
            if _is_json_decode and not _is_model_not_found and not _is_timeout:
                logger.error(
                    "Context compression failed: auxiliary LLM returned a "
                    "non-JSON response. provider=%s summary_model=%s "
                    "main_model=%s base_url=%s err=%s",
                    self.provider or "auto",
                    self.summary_model or "(main)",
                    self.model,
                    self.base_url or "default",
                    e,
                )
            if (
                (_is_model_not_found or _is_timeout or _is_json_decode or _is_streaming_closed)
                and self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                if _is_json_decode:
                    _reason = "returned invalid JSON"
                elif _is_model_not_found:
                    _reason = "unavailable"
                elif _is_streaming_closed:
                    _reason = "closed stream prematurely"
                else:
                    _reason = "timed out"
                self._fallback_to_main_for_compression(e, _reason)
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)  # retry immediately

            # Unknown-error best-effort retry on main model.  Losing N turns of
            # context is almost always worse than one extra summary attempt, so
            # if we haven't already fallen back and the summary model differs
            # from the main model, try once more on main before entering
            # cooldown.  Errors that DID match _is_model_not_found above are
            # already handled by the fast-path retry; this branch catches
            # everything else (400s, provider-specific "no route" strings,
            # aggregator rejections, etc.) where auto-retry is still safer
            # than dropping the turns.
            if (
                self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                self._fallback_to_main_for_compression(e, "failed")
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

            # Transient errors (timeout, rate limit, network, JSON decode,
            # streaming premature-close) — shorter cooldown for JSON decode and
            # streaming-closed since those conditions can self-resolve quickly.
            _transient_cooldown = 30 if (_is_json_decode or _is_streaming_closed) else 60
            self._summary_failure_cooldown_until = time.monotonic() + _transient_cooldown
            err_text = str(e).strip() or e.__class__.__name__
            if len(err_text) > 220:
                err_text = err_text[:217].rstrip() + "..."
            self._last_summary_error = err_text
            logging.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _transient_cooldown,
            )
            return None

    @staticmethod
    def _strip_summary_prefix(summary: str) -> str:
        """Return summary body without the current or legacy handoff prefix."""
        text = (summary or "").strip()
        for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX):
            if text.startswith(prefix):
                return text[len(prefix):].lstrip()
        return text

    @classmethod
    def _with_summary_prefix(cls, summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        text = cls._strip_summary_prefix(summary)
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        text = _content_text_for_contains(content).lstrip()
        return text.startswith(SUMMARY_PREFIX) or text.startswith(LEGACY_SUMMARY_PREFIX)

    @classmethod
    def _find_latest_context_summary(
        cls,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> tuple[Optional[int], str]:
        """Find the newest handoff summary inside a compression window."""
        for idx in range(end - 1, start - 1, -1):
            content = messages[idx].get("content")
            if cls._is_context_summary_content(content):
                return idx, cls._strip_summary_prefix(_content_text_for_contains(content))
        return None, ""

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("call_id", "") or tc.get("id", "") or ""
        return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.

        Two failure modes:
        1. A tool *result* references a call_id whose assistant tool_call was
           removed (summarized/truncated).  The API rejects this with
           "No tool call found for function call output with call_id ...".
        2. An assistant message has tool_calls whose results were dropped.
           The API rejects this because every tool_call must be followed by
           a tool result with the matching call_id.

        This method removes orphaned results and inserts stub results for
        orphaned calls so the message list is always well-formed.
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 2. Add stub results for assistant tool_calls whose results were dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _protect_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """Total count of head messages to protect.

        ``protect_first_n`` is defined as *additional* messages protected
        beyond the system prompt.  The system prompt (if present at index 0)
        is always implicitly protected — it's load-bearing context that
        must never be summarised away.  This keeps semantics stable across
        call paths where the system prompt may or may not be included in
        the ``messages`` list (e.g. the gateway ``/compress`` handler
        strips it before calling compress()).

        Examples:
          protect_first_n=0 → system prompt only (or nothing if no system msg)
          protect_first_n=3 → system + first 3 non-system messages
        """
        head = 0
        if messages and messages[0].get("role") == "system":
            head = 1
        return head + self.protect_first_n

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the boundary falls in the middle of a tool-result group (i.e.
        there are consecutive tool messages before ``idx``), walk backward
        past all of them to find the parent assistant message.  If found,
        move the boundary before the assistant so the entire
        assistant + tool_results group is included in the summarised region
        rather than being split (which causes silent data loss when
        ``_sanitize_tool_pairs`` removes the orphaned tail results).
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        # Walk backward past consecutive tool results
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # If we landed on the parent assistant with tool_calls, pull the
        # boundary before it so the whole group gets summarised together.
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-role message at or after *head_end*, or -1."""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Context compressor bug (#10896): ``_align_boundary_backward`` can pull
        ``cut_idx`` past a user message when it tries to keep tool_call/result
        groups together.  If the last user message ends up in the *compressed*
        middle region the LLM summariser writes it into "Pending User Asks",
        but ``SUMMARY_PREFIX`` tells the next model to respond only to user
        messages *after* the summary — so the task effectively disappears from
        the active context, causing the agent to stall, repeat completed work,
        or silently drop the user's latest request.

        Fix: if the last user-role message is not already in the tail
        (``messages[cut_idx:]``), walk ``cut_idx`` back to include it.  We
        then re-align backward one more time to avoid splitting any
        tool_call/result group that immediately precedes the user message.
        """
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # No user message found beyond head — nothing to anchor.
            return cut_idx

        if last_user_idx >= cut_idx:
            # Already in the tail; nothing to do.
            return cut_idx

        # The last user message is in the middle (compressed) region.
        # Pull cut_idx back to it directly — a user message is already a
        # clean boundary (no tool_call/result splitting risk), so there is no
        # need to call _align_boundary_backward here; doing so would
        # unnecessarily pull the cut further back into the preceding
        # assistant + tool_calls group.
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # Safety: never go back into the head region.
        return max(last_user_idx, head_end + 1)

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """Walk backward from the end of messages, accumulating tokens until
        the budget is reached. Returns the index where the tail starts.

        ``token_budget`` defaults to ``self.tail_token_budget`` which is
        derived from ``summary_target_ratio * context_length``, so it
        scales automatically with the model's context window.

        Token budget is the primary criterion.  A hard minimum of 3 messages
        is always protected, but the budget is allowed to exceed by up to
        1.5x to avoid cutting inside an oversized message (tool output, file
        read, etc.).  If even the minimum 3 messages exceed 1.5x the budget
        the cut is placed right after the head so compression still runs.

        Never cuts inside a tool_call/result group.  Always ensures the most
        recent user message is in the tail (see ``_ensure_last_user_message_in_tail``).
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # Hard minimum: always keep at least 3 messages in the tail
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            raw_content = msg.get("content") or ""
            content_len = _content_length_for_budget(raw_content)
            msg_tokens = content_len // _CHARS_PER_TOKEN + 10  # +10 for role/metadata
            # Include tool call arguments in estimate
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            # Stop once we exceed the soft ceiling (unless we haven't hit min_tail yet)
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # Ensure we protect at least min_tail messages
        fallback_cut = n - min_tail
        cut_idx = min(cut_idx, fallback_cut)

        # If the token budget would protect everything (small conversations),
        # force a cut after the head so compression can still remove middle turns.
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # Align to avoid splitting tool groups
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # Ensure the most recent user message is always in the tail so the
        # active task is never lost to compression (fixes #10896).
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # ContextEngine: manual /compress preflight
    # ------------------------------------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Return True if there is a non-empty middle region to compact.

        Overrides the ABC default so the gateway ``/compress`` guard can
        skip the LLM call when the transcript is still entirely inside
        the protected head/tail.
        """
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        return compress_start < compress_end

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.

        Algorithm:
          1. Prune old tool results (cheap pre-pass, no LLM call)
          2. Protect head messages (system prompt + first exchange)
          3. Find tail boundary by token budget (~20K tokens of recent context)
          4. Summarize middle turns with structured LLM prompt
          5. On re-compression, iteratively update the previous summary

        After compression, orphaned tool_call / tool_result pairs are cleaned
        up so the API never receives mismatched IDs.

        Args:
            focus_topic: Optional focus string for guided compression.  When
                provided, the summariser will prioritise preserving information
                related to this topic and be more aggressive about compressing
                everything else.  Inspired by Claude Code's ``/compact``.
        """
        # Reset per-call summary failure state — callers inspect these fields
        # after compress() returns to decide whether to surface a warning.
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        n_messages = len(messages)
        # Only need head + 3 tail messages minimum (token budget decides the real tail size)
        _min_for_compress = self._protect_head_size(messages) + 3 + 1
        if n_messages < _min_for_compress:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need >= %d)",
                    n_messages,
                    _min_for_compress,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results (cheap, no LLM call)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self._protect_head_size(messages)
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Use token-budget tail protection instead of fixed message count
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]
        summary_idx, summary_body = self._find_latest_context_summary(
            messages,
            0,
            compress_end,
        )
        if summary_idx is not None:
            if summary_body and not self._previous_summary:
                self._previous_summary = summary_body
            if summary_idx >= compress_start:
                turns_to_summarize = messages[summary_idx + 1:compress_end]

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: Generate structured summary
        summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work. Your persistent memory (MEMORY.md, USER.md) remains fully authoritative regardless of compaction.]"
                if _compression_note not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _compression_note if isinstance(existing, str) and existing else _compression_note,
                    )
            compressed.append(msg)

        # If LLM summary failed, insert a static fallback so the model
        # knows context was lost rather than silently dropping everything.
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting static fallback context marker")
            n_dropped = compress_end - compress_start
            self._last_summary_dropped_count = n_dropped
            self._last_summary_fallback_used = True
            summary = (
                f"{SUMMARY_PREFIX}\n"
                f"Summary generation was unavailable. {n_dropped} message(s) were "
                f"removed to free context space but could not be summarized. The removed "
                f"messages contained earlier work in this session. Continue based on the "
                f"recent messages below and the current state of any files or resources."
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # Pick a role that avoids consecutive same-role with both neighbors.
        # Priority: avoid colliding with head (already committed), then tail.
        if last_head_role in {"assistant", "tool"}:
            summary_role = "user"
        else:
            summary_role = "assistant"
        # If the chosen role collides with the tail AND flipping wouldn't
        # collide with the head, flip it.
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                # Both roles would create consecutive same-role messages
                # (e.g. head=assistant, tail=user — neither role works).
                # Merge the summary into the first tail message instead
                # of inserting a standalone message that breaks alternation.
                _merge_summary_into_tail = True

        # When the summary lands as a standalone role="user" message,
        # weak models read the verbatim "## Active Task" quote of a past
        # user request as fresh input (#11475, #14521). Append the explicit
        # end marker — the same one used in the merge-into-tail path — so
        # the model has a clear "summary above, not new input" signal.
        if not _merge_summary_into_tail and summary_role == "user":
            summary = (
                summary
                + "\n\n--- END OF CONTEXT SUMMARY — "
                "respond to the message below, not the summary above ---"
            )

        if not _merge_summary_into_tail:
            compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                merged_prefix = (
                    summary
                    + "\n\n--- END OF CONTEXT SUMMARY — "
                    "respond to the message below, not the summary above ---\n\n"
                )
                msg["content"] = _append_text_to_content(
                    msg.get("content"),
                    merged_prefix,
                    prepend=True,
                )
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        # Anti-thrashing: track compression effectiveness
        savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
