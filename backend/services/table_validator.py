"""
Structural validation for camelot-extracted tables.

camelot's `stream` flavor guesses column boundaries from whitespace and
routinely misfires on ordinary paragraphs or bulleted spec lists, producing a
1-2 row grid with mostly empty cells and no real header. The old gate here was
just "at least 2 rows and 2 columns", which that garbage sails through — the
empty header cells then get papered over with "Col 2", "Col 3" placeholders,
and an LLM asked to summarise a near-empty grid tends to invent a plausible
title instead of admitting there is nothing there. Reject it here, before any
of that happens.
"""
import re


def is_valid_table(df, flavor: str = "lattice") -> tuple:
    """(is_valid, reason). `flavor` matters: stream-detected tables are pure
    whitespace guesswork, so they need a materially stronger signal than a
    lattice table backed by actual ruled borders."""
    if df is None or df.empty:
        return False, "empty dataframe"

    rows, cols = df.shape
    if rows < 2 or cols < 2:
        return False, f"too small ({rows}x{cols})"

    # DataFrame.applymap was removed in pandas 3.0 in favor of .map(); .map()
    # on a DataFrame did not exist before pandas 2.1. Cheaper and version-proof
    # to just map each column's Series instead of picking one API to depend on.
    cells = df.fillna("").astype(str).apply(lambda col: col.str.strip())
    total = rows * cols
    non_empty = int((cells != "").values.sum())
    fill_ratio = non_empty / total if total else 0.0

    min_fill = 0.55 if flavor == "stream" else 0.35
    if fill_ratio < min_fill:
        return False, f"too sparse ({fill_ratio:.0%} filled, need {min_fill:.0%})"

    header = cells.iloc[0].tolist()
    header_filled = sum(1 for h in header if h)
    if header_filled / max(cols, 1) < 0.7:
        return False, f"header mostly empty ({header_filled}/{cols} cells)"

    # A real header names distinct columns; repeated or near-identical header
    # cells are a sign camelot sliced one column of running text into several.
    distinct = len({h.lower() for h in header if h})
    if distinct < max(2, int(cols * 0.6)):
        return False, "header cells are not distinct enough to be real column names"

    # Two rows total (header + one data row) is a very common shape for a
    # misdetected paragraph; only accept it when that one row is densely filled.
    if rows == 2 and fill_ratio < 0.8:
        return False, "single sparse data row - likely a misread paragraph, not a table"

    # A "table" that is just the same value repeated across the row (common
    # when stream mode splits a justified line of text into fake columns).
    body_rows = cells.iloc[1:]
    if len(body_rows) > 0:
        dup_rows = sum(1 for _, r in body_rows.iterrows() if len(set(v for v in r if v)) <= 1)
        if dup_rows / len(body_rows) > 0.6:
            return False, "data rows are mostly repeated/blank values"

    return True, f"{rows}x{cols}, {fill_ratio:.0%} filled"


_PLACEHOLDER = re.compile(r"^(col(umn)?\s*\d+|-|n/?a)$", re.IGNORECASE)


def markdown_is_usable(markdown: str) -> bool:
    """Runtime guard for tables already sitting in the database from before
    this validator existed. A garbage table stored under the old, weaker gate
    should still never reach chat."""
    if not markdown:
        return False
    lines = [l for l in markdown.strip().splitlines() if l.strip()]
    if len(lines) < 3:  # header + separator + at least one data row
        return False

    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    if sum(1 for c in header_cells if c and not _PLACEHOLDER.match(c)) < 2:
        return False

    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        real = [c for c in cells if c and not _PLACEHOLDER.match(c)]
        if len(real) >= 2:
            return True
    return False
