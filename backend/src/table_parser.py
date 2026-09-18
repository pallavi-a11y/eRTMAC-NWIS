"""
table_parser.py
Heuristic table detection over already-extracted page text - genuinely parses
pipe-delimited row runs into row/cell structure, not just cosmetic
reformatting. Applied at read/display time against already-stored page_text
(page_extractions.page_text / corrected_text) - no VLM call, no schema
change, no re-extraction, works retroactively on documents processed before
this existed (verified against document_id=10's already-stored real text).

WHY THIS OPTION OVER STRUCTURED VLM TABLE EXTRACTION (decided after explicit
cost/risk comparison, not by default): asking the model to additionally emit
structured table cells in the same Pass 2 call increases required output
length, which is exactly the axis on which the model was just found to fail
(see unified_parser.py's looks_incomplete/text_possibly_truncated - the model
hit its own natural stop token mid-transcription on a dense page well before
any token/context ceiling). Adding more required output per page risks making
that worse, not just adding a feature. This heuristic instead works entirely
off text the model already produced, at zero extra generation cost and zero
extra truncation risk.

LIMITATION, accepted rather than silently glossed over: this is a heuristic
over the model's own inconsistent delimiter usage, not a guaranteed parse. A
table row the model transcribed with no "|" characters at all (observed
directly in this project's real samples - document_id=10 page 4's casing row:
"13-3/8\" 27,569 lbs/mile of pipe through the perforated section...", zero
pipes) is not recoverable here and is left as plain prose, unchanged. A
partial win across the corpus is worth having even though it won't reach
every page - this is the same "heuristic + human review as the safety net,
don't chase a perfect automated fix" pattern already used for the
confidence-vs-correctness and hazard-fabrication limitations elsewhere in
this project.
"""
import html as html_escape
import re
from typing import Dict, List, Optional

_MIN_PIPES_PER_LINE = 1  # a line needs >=1 "|" to look like a delimited row. Was 2 - wrong for a genuine
                          # 2-column "label | value" table, which only ever needs one separator per row
                          # (verified directly: SVD-WKO-014 page 1's real transcription put exactly one "|"
                          # in two of its rows - "Document item | Record:" - which the >=2 threshold silently
                          # rejected outright, never even reaching the run-length check below).
_MIN_TABLE_LINES = 2     # a single pipe-containing line isn't enough to call it a table (could be a lone label)

# A markdown-style separator cell ("---", ":---:", "-", or empty) - the model
# frequently emits a full markdown separator row itself (verified directly:
# document_id=10 page 2's real stored text includes a literal "| --- | --- |
# --- | ---|" line right after its header row). That is formatting, not data,
# and must not be treated as a table row nor duplicated against the separator
# this module already inserts in to_markdown_table().
_SEPARATOR_CELL_RE = re.compile(r"^[:\-]*$")

# A data row with fewer than this fraction of the header's column count is
# rejected from the table rather than padded - found as a real, serious bug
# (document_id GEL-187 page 2's bit record table): the model pipe-delimited
# an 11-column header but only pipe-delimited the LAST 2-3 values of every
# data row, leaving the first 7 values as one space-separated blob with no
# pipes at all. The old padding logic crammed that whole blob into column 1,
# put the next two real values into columns 2-3, and left columns 4-11 blank
# - a table that LOOKS correctly structured but silently shows wrong data in
# the wrong cells, which is worse than plain text (a reviewer has no visual
# cue anything is wrong). 0.5 is deliberately not stricter than that: a row
# just one or two cells short of the header (the original, already-real
# "ragged row" case this module was built to tolerate) must still be padded
# as before, not rejected - only a row that's missing MOST of its columns
# gets excluded.
_MIN_ROW_FILL_RATIO = 0.5

# A row with far MORE cells than the header is just as untrustworthy as one
# with far fewer, and was NOT being caught before this existed - found as a
# real bug against document_id=1000017 page 2: a long flat list of schema
# field names ("well_id | latitude | longitude | ...") got line-wrapped by
# the source PDF across several print lines, and one of those wrapped lines
# alone had 12 cells against a 4-cell header. Nothing rejected it, so
# _flush's width = max(len(r) for every row) ballooned the WHOLE table to
# 12 columns, padding the real header out with 8 empty trailing cells and
# scrambling every other row into columns that didn't correspond to
# anything - not a wide real table, just wrapped prose that happens to
# contain "|" characters as a list separator. 1.5 mirrors _MIN_ROW_FILL_RATIO's
# tolerance for a row a little off from the header, while still catching a
# row that is 2-3x wider, which no genuine ragged-row case in real data has
# produced.
_MAX_ROW_FILL_RATIO = 1.5

# Deliberately stricter than _MIN_ROW_FILL_RATIO above - see its use at the
# blank-line-inside-a-run check for why crossing a gap needs a closer column
# match than merely tolerating one ragged row does.
_MIN_BLANK_GAP_FILL_RATIO = 0.8


def _is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(c.strip()) for c in cells) and any(c.strip() for c in cells)


def _column_always_empty(rows: List[List[str]]) -> bool:
    """
    True when every data row (header excluded by the caller) has an empty
    value in the same column index - found as a real, separate bug from the
    ragged-row one above (document_id GEL-187 page 2's "FIG. 2 PRESSURE / MUD
    WEIGHT PROFILE" chart): the model formatted a chart's axis tick labels
    and legend entries (400, 1,600, LOT, MUD LOSS, ...) as a fake two-column
    table where the second column - meant to hold the chart's actual
    plotted values, which cannot be read off a rendered curve - is empty on
    every single row. That is not a table; it is chart-caption content the
    model mistakenly pipe-formatted. A column that is 100% empty across
    every row is a strong, specific signal of exactly this pattern, not a
    normal sparse real column (a genuinely sparse column still has SOME
    populated rows in real data)."""
    if not rows:
        return False
    width = len(rows[0])
    for col in range(width):
        if all((row[col].strip() == "" if col < len(row) else True) for row in rows):
            return True
    return False


_KV_CELL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ /]*\s*=")
_MIN_KV_HEADER_RATIO = 0.5


def _header_looks_like_key_value(headers: List[str]) -> bool:
    """
    True when most of the "header" row's own cells are themselves key=value
    pairs (e.g. "well_id=GEL-187") rather than plain column labels (e.g.
    "DEPTH-m") - found as a real bug (document GEL-187 page 4's "13.0
    STRUCTURED SUMMARY RECORD (KEY=VALUE)" block): that block is a flat list
    of individual facts that happens to use "|" between them on each line,
    not a grid table with the same columns repeated per row. Treating its
    first line as a header produced a nonsensical table where a whole
    key=value fact ("well_id=GEL-187") was shown as a column heading. A real
    column header is just a label and never itself contains a literal "="
    with an assigned value - this is a specific, safe signal to reject a
    "table" this shape rather than render it."""
    if not headers:
        return False
    kv_like = sum(1 for h in headers if _KV_CELL_RE.match(h.strip()))
    return kv_like / len(headers) >= _MIN_KV_HEADER_RATIO


_FIELD_NAME_CELL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MIN_FIELD_NAME_RATIO = 0.7


def _looks_like_flat_field_list(header: List[str], rows: List[List[str]]) -> bool:
    """
    True when most of the table's cells - header AND data rows alike - are
    themselves bare identifier-style field names (e.g. "mud_weight_sg",
    "well_location", "NPT") rather than a real header/data split - found as
    a real bug (document_id=1000017 page 2's "AI-READY STRUCTURED STRING"
    block): a long flat list of schema field names, wrapped by the source
    PDF across several print lines that each happened to contain "|"
    characters, was detected as a table whose "data rows" were just MORE
    field names, not actual values (numbers, dates, short prose) - a real
    table's data rows look different in kind from its header; this one's
    rows look identical to it. This is the sibling of
    _header_looks_like_key_value above (same flat-list-mistaken-for-a-grid
    root cause, different concrete shape), checked across every populated
    cell in the table rather than just the header, since the header alone
    here ("well_id", "latitude", "longitude") looks like a perfectly
    plausible column-label row on its own - only checking the data rows too
    reveals they are not values, they are more labels."""
    all_cells = [c for c in header if c.strip()] + [c for row in rows for c in row if c.strip()]
    if not all_cells:
        return False
    field_like = sum(1 for c in all_cells if _FIELD_NAME_CELL_RE.match(c.strip()))
    return field_like / len(all_cells) >= _MIN_FIELD_NAME_RATIO


def _trim_trailing_fragment_rows(indexed_rows: List[tuple]) -> List[tuple]:
    """
    Drops trailing (line_index, row) entries where only one cell is
    populated (every other cell blank), working from the end of the table's
    data rows inward and stopping at the first row that isn't a fragment -
    found as a real bug (document GEL-187 page 4's "FIGURE 6 NON-PRODUCTIVE
    TIME PARETO" chart): the first several rows of a detected 2-column table
    were a genuine, if garbled, attempt at describing the chart's bars (both
    cells populated), but the model then continued generating the chart's
    raw axis-tick numbers (80, 70, 100, 50, ...) as further "rows" of the
    SAME table, each with only its first cell populated. _column_always_empty
    cannot catch this - the EARLY rows genuinely have both columns filled,
    only the trailing tail degrades. Trimming from the end inward targets
    exactly that degradation without touching a real, if occasionally
    sparse, row elsewhere in the table - only a fragment run reaching all
    the way to the table's last row is removed. Returns (line_index, row)
    pairs so the caller can still show the trimmed lines as plain text
    (their original line numbers stay known) rather than silently losing
    them."""
    if len(indexed_rows) < 1 or len(indexed_rows[0][1]) < 2:
        return indexed_rows
    trimmed = list(indexed_rows)
    while trimmed:
        _, row = trimmed[-1]
        if sum(1 for cell in row if cell.strip()) <= 1:
            trimmed.pop()
        else:
            break
    return trimmed


def _split_row(line: str) -> List[str]:
    # Strip a leading/trailing "|" first so "| a | b |" doesn't produce empty
    # first/last cells - the model's own output is inconsistent about whether
    # rows are pipe-terminated on both ends (observed in real samples).
    stripped = line.strip().strip("|").strip()
    return [cell.strip() for cell in stripped.split("|")]


def extract_tables(page_text: Optional[str]) -> List[Dict]:
    """
    Scans page_text line by line for runs of pipe-delimited rows and returns
    them as real structured tables:
    [{"start_line": int, "end_line": int, "headers": [...], "rows": [[...], ...]}]

    A blank line BETWEEN two pipe-delimited lines does not break the run - this
    was found necessary against real data, not assumed: this project's real
    samples put each table row on its own paragraph-like block separated by a
    blank line (the model does not emit tight consecutive table lines the way
    a hand-authored markdown table would), and an earlier version of this
    function that broke a run on any non-pipe line only ever recovered 1 of 4
    real rows on document_id=10 page 2 as a result. A genuine prose line
    (non-blank, and not pipe-delimited) still ends the run.

    The first row of a detected run is treated as the header row - matches
    this project's real sample data, where the model consistently transcribes
    a header row first. A markdown separator row (see _is_separator_row) is
    recognized and dropped rather than kept as a fake data row. Ragged rows (a
    row transcribed with fewer cells than the header - a real failure mode
    observed directly, e.g. page 2's phase rows vary in cell count) are padded
    with empty strings rather than dropping the whole table over one short row
    - UNLESS a row is missing MOST of its columns (see _MIN_ROW_FILL_RATIO),
    in which case it's excluded from the table entirely and falls through as
    plain text instead of scrambling real values into the wrong cells.

    A detected table where one entire column is empty on every data row is
    also discarded outright (see _column_always_empty) rather than rendered -
    this is the signature of a chart's axis/legend content having been
    pipe-formatted as if it were a table (see that function's docstring for
    the real example this was found against), not an actual table worth
    showing as one.
    """
    if not page_text:
        return []

    lines = page_text.split("\n")
    tables: List[Dict] = []
    current_run: List[tuple] = []  # (line_index, cells), pipe-lines only - interior blanks are not stored here

    def _flush():
        if len(current_run) >= _MIN_TABLE_LINES:
            start_line = current_run[0][0]
            indexed_data_rows = [(i, cells) for i, cells in current_run if not _is_separator_row(cells)]
            if indexed_data_rows:
                width = max(len(r) for _, r in indexed_data_rows)
                padded = [(i, r + [""] * (width - len(r))) for i, r in indexed_data_rows]
                header_line, header = padded[0]
                body = _trim_trailing_fragment_rows(padded[1:])
                rows = [r for _, r in body]
                # end_line reflects whatever survived trimming, not the
                # original last line - a trimmed-off tail must fall back to
                # plain text (its raw line still exists and must render
                # SOMEWHERE), not be silently dropped just because it used
                # to be inside this table's line range.
                end_line = body[-1][0] if body else header_line
                if (
                    rows
                    and not _column_always_empty(rows)
                    and not _header_looks_like_key_value(header)
                    and not _looks_like_flat_field_list(header, rows)
                ):
                    tables.append(
                        {
                            "start_line": start_line,
                            "end_line": end_line,
                            "headers": header,
                            "rows": rows,
                        }
                    )
        current_run.clear()

    for i, line in enumerate(lines):
        if line.count("|") >= _MIN_PIPES_PER_LINE:
            cells = _split_row(line)
            if current_run and not _is_separator_row(cells):
                header_width = len(current_run[0][1])
                if header_width and (
                    len(cells) < header_width * _MIN_ROW_FILL_RATIO
                    or len(cells) > header_width * _MAX_ROW_FILL_RATIO
                ):
                    # Missing most of its columns, OR far more of them than
                    # the header has - not trustworthy as a row of THIS
                    # table either way. Skip it WITHOUT flushing (the run,
                    # and its header, stay open) so a later well-formed row
                    # can still join the same table. Flushing here instead
                    # was tried first and rejected: it let the very next
                    # ragged row become a fresh "header" for a new table,
                    # which just relocated the scrambled-column problem into
                    # a table header cell instead of actually fixing it.
                    continue
            current_run.append((i, cells))
        elif line.strip() == "" and current_run:
            # A blank line does not automatically mean "still the same
            # table" - found as a real bug from the fix above: a genuinely
            # DIFFERENT table's header (e.g. "TOC-m | CMT VOL-M3 | BOND
            # IDX", 3 cells) sitting right after a blank line was passing
            # the general ragged-row tolerance (3 cells is >=50% of a
            # 5-cell header) and getting silently absorbed as a bogus extra
            # "row" of the FIRST table instead of starting its own table -
            # which is exactly the case _merge_adjacent_split_tables below
            # exists to recombine correctly. Crossing a blank line requires
            # a much closer column-count match (_MIN_BLANK_GAP_FILL_RATIO)
            # than tolerating a ragged row does - a real gap inside one
            # table still has a similar cell count on the other side; a
            # different table's header usually does not.
            next_cells = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    continue
                if lines[j].count("|") >= _MIN_PIPES_PER_LINE:
                    next_cells = _split_row(lines[j])
                break
            header_width = len(current_run[0][1])
            if next_cells is not None and header_width and len(next_cells) >= header_width * _MIN_BLANK_GAP_FILL_RATIO:
                continue
            _flush()
        else:
            _flush()
    _flush()
    return _merge_adjacent_split_tables(tables)


_MAX_MERGE_GAP_LINES = 3  # allow a couple of blank lines between two tables and still consider them adjacent


def _merge_adjacent_split_tables(tables: List[Dict]) -> List[Dict]:
    """
    Merges two adjacent detected tables into one wider table when they look
    like the same table split into column-groups by the model, rather than
    two genuinely separate tables - found as a real, distinct failure mode
    (document GEL-187 page 4's casing table): the model transcribed a real
    8-column table (STRING/OD-in/WT-lb/ft/GRADE/SHOE-m MD/TOC-m/CMT VOL-m3/
    BOND IDX) as two separate 5-and-3-column table blocks with a blank line
    between them, instead of keeping all 8 columns on one row. Every
    individual value was still correct for its own row - this is purely a
    presentation split, not the data-scrambling error the ragged-row check
    above guards against - but a table sliced into two disconnected halves
    still doesn't match what is actually on the page.

    Two tables are merged only when they sit right next to each other (a gap
    small enough that nothing meaningful could be between them) AND have the
    exact same row count - a specific, deliberately narrow signal that they
    are column-groups of one table, not a coincidence of two unrelated
    tables landing near each other. Getting a merge wrong in some future case
    only costs a wider single table instead of two side by side - a smaller
    downside than the split this fixes.
    """
    if len(tables) < 2:
        return tables
    merged: List[Dict] = [tables[0]]
    for table in tables[1:]:
        prev = merged[-1]
        gap = table["start_line"] - prev["end_line"]
        if 0 < gap <= _MAX_MERGE_GAP_LINES and len(table["rows"]) == len(prev["rows"]):
            merged[-1] = {
                "start_line": prev["start_line"],
                "end_line": table["end_line"],
                "headers": prev["headers"] + table["headers"],
                "rows": [pr + tr for pr, tr in zip(prev["rows"], table["rows"])],
            }
        else:
            merged.append(table)
    return merged


def to_markdown_table(table: Dict) -> str:
    headers = table["headers"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in table["rows"]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def to_html_table(table: Dict) -> str:
    """
    Real <table> markup - actual bordered rows/columns, not markdown pipe
    text. A <textarea> can only ever display flat characters, so no amount
    of text formatting can make it visually look like a structured grid;
    this is what actually produces one.
    """
    header_html = "".join(f"<th>{html_escape.escape(h)}</th>" for h in table["headers"])
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{html_escape.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in table["rows"]
    )
    return f'<table class="detected-table"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>'


def render_page_as_html(page_text: Optional[str]) -> dict:
    """
    Same detection as render_page_with_tables, but produces real HTML:
    detected table regions become genuine <table> elements, everything else
    stays as escaped plain text inside <pre> blocks (preserving line breaks).
    Returns {"tables": [...], "html": str}. Read-only, no DB write.
    """
    if not page_text:
        return {"tables": [], "html": ""}

    tables = extract_tables(page_text)
    if not tables:
        return {"tables": [], "html": f"<pre>{html_escape.escape(page_text)}</pre>"}

    lines = page_text.split("\n")
    table_by_start = {t["start_line"]: t for t in tables}
    covered = set()
    for t in tables:
        covered.update(range(t["start_line"], t["end_line"] + 1))

    out_parts = []
    text_buffer: List[str] = []

    def flush_text():
        if text_buffer:
            out_parts.append(f"<pre>{html_escape.escape(chr(10).join(text_buffer))}</pre>")
            text_buffer.clear()

    i = 0
    while i < len(lines):
        if i in table_by_start:
            flush_text()
            out_parts.append(to_html_table(table_by_start[i]))
            i = table_by_start[i]["end_line"] + 1
            continue
        if i in covered:
            i += 1
            continue
        text_buffer.append(lines[i])
        i += 1
    flush_text()

    return {"tables": tables, "html": "\n".join(out_parts)}


def render_page_with_tables(page_text: Optional[str]) -> dict:
    """
    Display-time transform over already-stored text: returns
    {"tables": [...structured, see extract_tables...], "text_with_tables_formatted": str}
    where each detected pipe-delimited row run is replaced by a clean markdown
    table block; every other line (prose, headers, anything without enough
    pipes to look tabular) passes through completely unchanged. Read-only -
    does not touch the database and is safe to call on every page-view
    request, including for documents processed before this module existed.
    """
    if not page_text:
        return {"tables": [], "text_with_tables_formatted": page_text or ""}

    tables = extract_tables(page_text)
    if not tables:
        return {"tables": [], "text_with_tables_formatted": page_text}

    lines = page_text.split("\n")
    table_by_start = {t["start_line"]: t for t in tables}
    covered = set()
    for t in tables:
        covered.update(range(t["start_line"], t["end_line"] + 1))

    out_lines = []
    i = 0
    while i < len(lines):
        if i in table_by_start:
            out_lines.append(to_markdown_table(table_by_start[i]))
            i = table_by_start[i]["end_line"] + 1
            continue
        if i in covered:
            i += 1
            continue
        out_lines.append(lines[i])
        i += 1

    return {"tables": tables, "text_with_tables_formatted": "\n".join(out_lines)}
