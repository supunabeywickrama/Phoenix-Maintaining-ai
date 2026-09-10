"""
Tolerant JSON parsing for LLM responses.

Even in JSON mode the model can hit max_tokens mid-object, and the response then
arrives as a syntactically invalid prefix — the errors look like
"Unterminated string starting at: line 37 column 20" or
"Expecting ',' delimiter: line 125 column 13". A strict json.loads throws all of
that output away, which is why pattern generation and figure splitting
intermittently fell back to defaults or failed all retries.

loads_tolerant() first tries a strict parse, and only if that fails rewinds to
the last structurally complete point and closes the open brackets, keeping the
items that did arrive intact.
"""
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers some models add despite JSON mode."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()
    return t


def _safe_points(text: str) -> list[tuple[int, list[str]]]:
    """
    Scan once and record every index where the document is structurally
    complete-so-far, along with the closers still owed at that index.
    """
    points: list[tuple[int, list[str]]] = []
    stack: list[str] = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
            points.append((i + 1, list(stack)))
        elif ch == ",":
            # cut *before* the comma: everything up to here is a whole value
            points.append((i, list(stack)))

    return points


def loads_tolerant(text: Optional[str], default: Any = None) -> Any:
    """
    Parse JSON from an LLM response, salvaging truncated output where possible.

    Returns `default` only if nothing usable can be recovered.
    """
    if not text:
        return default

    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        pass

    # Walk candidate cut points from the latest backwards, closing open brackets.
    for idx, owed in reversed(_safe_points(cleaned)):
        candidate = cleaned[:idx] + "".join(reversed(owed))
        try:
            parsed = json.loads(candidate)
            logger.warning(
                "Recovered truncated LLM JSON: kept %d of %d chars", idx, len(cleaned)
            )
            return parsed
        except json.JSONDecodeError:
            continue

    logger.error("Could not salvage LLM JSON (%d chars)", len(cleaned))
    return default
