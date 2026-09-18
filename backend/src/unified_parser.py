"""
unified_parser.py
Zero-OCR extraction pipeline: turns rasterized WCR/DDR page PNGs (from
pdf_processor.py) into structured records via Qwen2.5-VL over Ollama, with a
confidence signal on every extracted field.

KNOWN, ACCEPTED LIMITATION - confidence tracks legibility, not correctness:
verified directly (test_vlm_extraction.py vs test_vlm_extraction_clean.py) that
the model's stated confidence reflects how legible the source pixels looked, not
whether its reading is factually correct. On a degraded sample page, "WESTERN
INDIA PETROLEUM CORP." was misread as "WESTERN INDIAN OIL CORP." and still
tagged MEDIUM confidence, not LOW - the model does not detect its own
misreadings. This is an accepted, documented limitation, not a bug to chase:
asking a 3B VLM to self-grade factual accuracy is a harder problem than this
project needs to solve. Mitigations instead: (1) the "Supplementary Well-File
Notes" prose cross-check when a table value is ambiguous, (2) NEEDS_REVIEW
document status + human review for low-confidence/missing safety-critical
fields (coordinates, hazard depth). Do not try to prompt-engineer around this -
it was evaluated and rejected in favor of the review-based mitigation.

CORRECTED APPROACH (was a known, accepted limitation for one session; fixed
properly once tried): early sessions treated "the model does not reliably emit
strictly valid JSON" as a long tail of malformations to patch one regex at a
time in _parse_json_response() - across three full 20-page runs, three
DIFFERENT classes turned up in turn (inconsistent ```json fencing, bare
unquoted confidence enums, trailing commas + a stray escape fragment). That
was the wrong first move: Ollama's `format` parameter accepts a JSON Schema for
grammar-constrained decoding (see PASS1_SCHEMA/PASS2_SCHEMA below, threaded
through utils.call_ollama_chat), and testing it directly (not assumed) showed
it eliminates JSON syntax/shape failures outright - no fences, no bare enums,
no trailing commas, no garbage top-level keys, because the decoder is
constrained to only ever emit tokens matching the schema. This should have
been tried before writing any repair regex. The old regex repairs in
_parse_json_response() are kept as cheap, idempotent defense-in-depth (harmless
no-ops against already-valid JSON) rather than removed, but the schema is now
the primary fix, not an afterthought.

IMPORTANT - schema-constrained output does NOT fix semantic fabrication: tested
directly on the same hazard-free parameter-table page that first exposed the
fabrication problem - with the schema enforced, the model still produced
"Stuck Pipe" and "Lost Circulation" entries (with depth_m: null this time,
rather than a fake depth, but still fabricated categories with no basis on the
page). The HAZARD_CATALOG whitelist + _hazard_mentioned_in_text corroboration
check below remain necessary regardless of the schema - constraining syntax
does not constrain truthfulness, and this is not expected to change with any
future prompt tweak.

Deliberately NOT VLM-judged, for the same "deterministic beats free-form LLM
judgment on safety-adjacent data" reasoning:
  - hazard severity -> looked up from hazard_type via HAZARD_CATALOG (shared
    with the fabrication whitelist below - these two were originally separate
    dicts and drifted apart in practice, so they were merged into one)
  - well lifecycle status -> derived from keyword rules over transcribed page
    text (derive_well_status), separate from well_type which IS extracted
    as-is from the document's own "Well Status" field (see the type/status
    distinction below)

Out of scope for this module (by design, not oversight): embedding page text
and indexing into Qdrant is vector_store.py's job, not built yet. Callers that
need semantic search/RAG should take this module's per-page text output and
hand it to vector_store.py separately.
"""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from src import database, figure_extractor, table_parser, vector_store
from src.config import VLM_MODEL_NAME, VLM_NUM_CTX_PASS1, VLM_NUM_CTX_PASS2
from src.utils import (
    OllamaConnectionError,
    OllamaUnavailableError,
    call_ollama_chat,
    normalize_numeric,
    wait_for_ollama_recovery,
)

# ---------------------------------------------------------------------------
# Single canonical hazard taxonomy - severity lookup AND the fabrication
# whitelist (_hazard_mentioned_in_text, below) both read from this ONE
# structure so they cannot independently drift apart. This already happened
# once during development: "blowout" was added to the old, separate keyword
# whitelist without being added to the old, separate severity table - which
# would have silently defaulted a blowout (arguably the single most
# catastrophic drilling event) to MEDIUM severity. Exactly four categories
# here, matching the originally authorized severity design - "kick" is kept
# as an alias for "gas kick" (same event, shorter phrasing the model
# sometimes uses as hazard_type), not a new category.
# ---------------------------------------------------------------------------
HAZARD_CATALOG = {
    "gas kick": {"severity": "CRITICAL", "keywords": ("gas kick", "kick")},
    "kick": {"severity": "CRITICAL", "keywords": ("kick",)},
    "stuck pipe": {"severity": "HIGH", "keywords": ("stuck pipe", "stuck")},
    "lost circulation": {"severity": "MEDIUM", "keywords": ("lost circulation", "circulation loss", "losses")},
    "washout": {"severity": "MEDIUM", "keywords": ("washout", "wash out")},
}
DEFAULT_SEVERITY = "MEDIUM"  # safe default for any hazard_type not in the catalog above


def lookup_severity(hazard_type: Optional[str]) -> str:
    if not hazard_type:
        return DEFAULT_SEVERITY
    entry = HAZARD_CATALOG.get(hazard_type.strip().lower())
    return entry["severity"] if entry else DEFAULT_SEVERITY


# ---------------------------------------------------------------------------
# Well lifecycle status - derived via keyword rules over transcribed page text,
# not VLM-judged. This is `status` (ACTIVE/PLUGGED/SUSPENDED/UNKNOWN); it is
# NOT the same as `well_type` (e.g. "Oil Producer"), which Pass 1 extracts
# directly from the document's own "Well Status" field. A well can say "Oil
# Producer" in well_type while actually being suspended - see the sample WCR's
# own Section 19 "Rig Abandonment/Suspension Status".
# ---------------------------------------------------------------------------
_PLUGGED_KEYWORDS = ("plugged", "plugging", "p&a", "abandon")
_SUSPENSION_KEYWORDS = ("suspend", "suspended", "suspension")


def derive_well_status(all_page_texts: list) -> str:
    combined = " ".join(t.lower() for t in all_page_texts if t)
    # Plugged checked first: an abandonment section is a stronger, more permanent
    # claim than a suspension mention, and some phrasing ("suspended pending
    # future plugging") could otherwise match both.
    if any(k in combined for k in _PLUGGED_KEYWORDS):
        return "PLUGGED"
    if any(k in combined for k in _SUSPENSION_KEYWORDS):
        return "SUSPENDED"
    return "ACTIVE"


# ---------------------------------------------------------------------------
# Well-comparison enrichment fields - derived the same way as well lifecycle
# status above: regex over already-transcribed page text, NOT a new VLM call.
# Deliberately kept out of the image-based Pass 1/Pass 2 extraction (see
# database.py's migrate_schema comment) - adding more fields there would make
# pages that already fail under VRAM/output-length pressure fail more, not
# less. This runs as a cheap post-processing step over text that already
# extracted successfully, so it can't make any existing document fail
# differently.
#
# Scope note on "drilling parameters" from the PS wording: the only
# drilling-parameter-style field that shows up as a single, reliably-labeled
# value in a WCR/DDR (rather than as a full time-series table, which would
# need real table parsing, not a keyword scan) is mud weight. So "drilling
# parameters" is represented here by mud weight specifically, not a broader
# set of fields.
# ---------------------------------------------------------------------------
_FORMATION_LABELS = (
    "target formation", "primary reservoir", "reservoir formation",
    "formation", "reservoir",
)
_MUD_TYPE_LABELS = (
    "mud type", "drilling fluid type", "drilling fluid", "fluid type", "mud system",
)
_MUD_WEIGHT_LABELS = (
    "mud weight", "mud density", "drilling fluid density",
)
_CASING_LABELS = (
    "casing program", "casing design", "casing scheme", "casing string", "casing setting depth",
)
_CEMENTING_LABELS = (
    "cementing program", "cement program", "cementing practice", "cement slurry",
    "top of cement", "cementing details",
)
_RESERVOIR_CHAR_LABELS = (
    "reservoir pressure", "porosity", "permeability", "reservoir characteristics",
    "net pay", "pay thickness",
)
# Allows a numeric-starting capture (mud weight "9.2 ppg", porosity "18%") and
# a percent sign, unlike the original formation/mud-type template which only
# needed alphabetic values.
_LABELED_FIELD_TEMPLATE = r"(?:{labels})\s*[:|]\s*([A-Za-z0-9][A-Za-z0-9 ,./%\-]{{1,60}})"
_FORMATION_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_FORMATION_LABELS)), re.IGNORECASE)
_MUD_TYPE_LABEL_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_MUD_TYPE_LABELS)), re.IGNORECASE)
_MUD_WEIGHT_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_MUD_WEIGHT_LABELS)), re.IGNORECASE)
_CASING_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_CASING_LABELS)), re.IGNORECASE)
_CEMENTING_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_CEMENTING_LABELS)), re.IGNORECASE)
_RESERVOIR_CHAR_RE = re.compile(_LABELED_FIELD_TEMPLATE.format(labels="|".join(_RESERVOIR_CHAR_LABELS)), re.IGNORECASE)

# Real, standard industry mud-system names - a closed vocabulary safe to
# match as a bare keyword anywhere in the text. Tried BEFORE the labeled-
# field regex above for mud type specifically: a transcribed markdown table's
# own HEADER row ("Depth Interval | Mud Type | Mud Weight | Viscosity") uses
# the exact same pipe-delimited shape as a data row, so the labeled regex can
# accidentally capture the next column header instead of a real value.
# Matching a known mud name directly sidesteps that table-structure ambiguity
# entirely. Formation names have no equivalent closed vocabulary (they're
# open-ended proper nouns), so formation relies on the labeled regex only.
_MUD_TYPE_KEYWORDS = (
    "synthetic oil-based", "synthetic oil based", "oil-based mud", "oil based mud",
    "water-based mud", "water based mud", "kcl-polymer", "kcl polymer",
    "invert emulsion", "polymer mud", "bentonite mud",
)


def _clean_captured_field(raw: str) -> Optional[str]:
    """Trims a regex-captured field value down to a plausible single phrase -
    a transcribed markdown table often runs several pipe-delimited cells
    together on one line, so this cuts off at the next pipe/newline rather
    than returning a whole table row."""
    value = raw.split("|")[0].split("\n")[0].strip(" .:-")
    if len(value) < 2:
        return None
    return value[:60]


def derive_well_enrichment(all_page_texts: list) -> dict:
    """Returns a dict of {formation, mud_type, mud_weight, casing_notes,
    cementing_notes, reservoir_notes} - any value may be None if nothing in
    the document's transcribed text matched. First match found (in page
    order) wins for each field; not an attempt at exhaustively finding every
    mention, matching this project's existing preference for simple,
    explainable heuristics over exact correctness (see derive_well_status
    above)."""
    combined = "\n".join(t for t in all_page_texts if t)

    formation = None
    match = _FORMATION_RE.search(combined)
    if match:
        formation = _clean_captured_field(match.group(1))

    mud_type = None
    lowered = combined.lower()
    for keyword in _MUD_TYPE_KEYWORDS:
        if keyword in lowered:
            mud_type = keyword.title()
            break
    if mud_type is None:
        match = _MUD_TYPE_LABEL_RE.search(combined)
        if match:
            mud_type = _clean_captured_field(match.group(1))

    mud_weight = None
    match = _MUD_WEIGHT_RE.search(combined)
    if match:
        mud_weight = _clean_captured_field(match.group(1))

    casing_notes = None
    match = _CASING_RE.search(combined)
    if match:
        casing_notes = _clean_captured_field(match.group(1))

    cementing_notes = None
    match = _CEMENTING_RE.search(combined)
    if match:
        cementing_notes = _clean_captured_field(match.group(1))

    reservoir_notes = None
    match = _RESERVOIR_CHAR_RE.search(combined)
    if match:
        reservoir_notes = _clean_captured_field(match.group(1))

    return {
        "formation": formation,
        "mud_type": mud_type,
        "mud_weight": mud_weight,
        "casing_notes": casing_notes,
        "cementing_notes": cementing_notes,
        "reservoir_notes": reservoir_notes,
    }


# ---------------------------------------------------------------------------
# JSON parsing - tolerant of the fence inconsistency noted in the module docstring,
# and of a systematic malformation found during the full 20-page run: confidence
# values sometimes come back as bare unquoted JSON tokens
# (`"confidence": LOW` instead of `"confidence": "LOW"`), which is invalid JSON.
# This was silently discarding otherwise-correct, high-quality extractions - it's
# specifically what caused Pass 1 to lose a genuine HIGH-confidence coordinate
# reading on page 2 during that run (verified: the raw response had the right
# lat/lon values, just wrapped in unparseable JSON over this one issue).
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_BARE_ENUM_RE = re.compile(r":\s*(HIGH|MEDIUM|LOW)(\s*[,}\]])")
# Found on a later run: a trailing comma before a closing bracket/brace
# (`"LOW"\n     },\n   ]\n}` - comma right before `]`), a separate, unrelated
# malformation from the bare-enum one above. This is one of the most common
# LLM JSON mistakes generally, not specific to this project - safe to always
# strip since a legitimate trailing comma is never valid JSON anyway.
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")
_VALID_CONFIDENCE = ("HIGH", "MEDIUM", "LOW")


def _parse_json_response(raw_text: str) -> Optional[dict]:
    """Best-effort repair before parsing. NOTE (added after a second full run):
    this is not a closed set - a later run hit yet another malformation
    (a stray escape fragment) that neither repair here catches. Treat this
    function as "fixes the malformations found so far", not "guaranteed to
    parse anything the model produces" - _call_and_parse's retry exists
    precisely because this list of repairs will likely never be complete."""
    cleaned = _FENCE_RE.sub("", raw_text.strip())
    cleaned = _BARE_ENUM_RE.sub(r': "\1"\2', cleaned)
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _log(document_id: int, message: str) -> None:
    """
    Plain print, not the logging module - matches this project's existing
    style (no logging framework set up anywhere else). flush=True so this
    is visible immediately in both a standalone script's console AND a live
    server's uvicorn output while a background task is still running, not
    buffered until the whole (multi-minute) run_extraction() call returns.
    """
    print(f"[doc {document_id}] {message}", flush=True)


_RETRY_BACKOFF_SECONDS = 3.0


def _grayscale_retry_copy(image_path: str) -> str:
    """
    Saves a grayscale-converted copy of image_path alongside it (deterministic
    name, overwritten each call - does not proliferate), for use as a
    last-resort retry variant when every normal RGB attempt returned
    Ollama's done=False failure (see call_ollama_chat's docstring).

    Verified directly, not assumed, and NOT a universal fix: on
    document_id=15 (NHK042), page 9 failed 4/4 times in RGB and then
    succeeded 2/2 times in grayscale; page 1 of the SAME document failed
    consistently in BOTH RGB (4/4) and grayscale (3/3). Tried as a final
    fallback specifically because it sometimes recovers a page an RGB-only
    retry never would, at negligible extra cost - not because it is
    guaranteed to help, and some pages will still exhaust every attempt.
    """
    from PIL import Image

    gray_path = str(Path(image_path).with_name(Path(image_path).stem + "_gray_retry.png"))
    Image.open(image_path).convert("L").convert("RGB").save(gray_path, format="PNG")
    return gray_path


def _call_and_parse(
    prompt: str,
    image_path: str,
    num_ctx: int,
    format: Optional[dict] = None,
    max_attempts: int = 3,
    is_acceptable: Optional[Callable[[dict], bool]] = None,
) -> tuple:
    """Calls the VLM and parses its JSON response, retrying on failure with a
    short backoff between attempts.

    ACCEPTED HARDWARE LIMITATION (decided over chasing a deeper fix - see
    config.py's VLM_NUM_CTX_PASS2 note and README.md): on this 4GB card, holding
    enough context for even one page image (~4300+ tokens, unavoidable) leaves
    the model unable to fit fully in VRAM - `ollama ps` showed a 59%/41% CPU/GPU
    split with ~175MB VRAM free. This intermittently produces an incomplete
    response (see utils.call_ollama_chat's done=False check) rather than a clean
    error. It reproduced twice, then succeeded on the very next identical call -
    this looks like transient VRAM/driver pressure, not a deterministic wall, so
    retrying with a short pause (for the driver/allocator to settle) is the
    accepted mitigation rather than lowering render DPI (would hurt legibility,
    the system's core purpose) or swapping models (a bigger, riskier change for
    an intermittent issue that retrying already resolves in practice). This
    means a page can now take multiple call-lengths of wall time in the worst
    case - expected and acceptable for this project's demo/MVP scope, not a
    production throughput target.

    `is_acceptable`: an optional second gate beyond "is this valid JSON at all" -
    added after finding (verified directly, not assumed - see extract_page_content's
    docstring) that the model can emit perfectly valid, schema-conforming JSON
    around a page_text it stopped transcribing partway through: it hits its own
    natural stop token mid-page (Ollama reports done_reason="stop", nowhere near
    num_ctx/num_predict), and schema-constrained decoding just forces the JSON
    to close cleanly around whatever was generated so far. That is invisible to
    the plain "did json.loads succeed" check this function already did - so a
    second, content-level check is needed for the caller to ask for a retry on
    a result that parsed fine but looks incomplete. Returns (parsed, accepted,
    failure_reason): accepted=False when every attempt parsed but never
    satisfied is_acceptable - the LAST attempt's parsed result is still
    returned (a likely-truncated transcription is more useful to a human
    reviewer than nothing), but the caller must not treat accepted=False as a
    clean, trustworthy result. failure_reason is None except when parsed is
    None (a total failure - not even one attempt produced parseable JSON), in
    which case it's a short human-readable explanation distinguishing the two
    ways that can happen: Ollama never returning a usable response at all
    (matches the documented VRAM-pressure pattern above) vs. a response
    coming back that couldn't be parsed as JSON even after repair.

    Also returns (as the 4th/5th elements) the raw response text and
    done_reason from the LAST attempt that actually got a response, whenever
    parsed is None - callers that know how to salvage a specific field out of
    truncated JSON (see extract_page_content's use of
    _salvage_truncated_page_text) need the original text, since by
    definition it never became valid JSON. Both are None on any successful
    return, since there is nothing to salvage."""
    last_parsed = None
    last_raw_text = None
    last_done_reason = None
    generation_failures = 0
    json_failures = 0
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(_RETRY_BACKOFF_SECONDS)
        # Last-resort variant: only on the final attempt, and only if every
        # prior attempt failed to produce ANY parseable response (last_parsed
        # still None - not merely "parsed but is_acceptable rejected it",
        # which is a different, already-handled failure mode) - try a
        # grayscale-converted copy of the same page instead of RGB. See
        # _grayscale_retry_copy's docstring for why, and why this is not
        # attempted on earlier attempts (RGB succeeds most of the time).
        attempt_image_path = image_path
        if attempt == max_attempts - 1 and last_parsed is None and max_attempts > 1:
            try:
                attempt_image_path = _grayscale_retry_copy(image_path)
            except Exception:
                attempt_image_path = image_path  # conversion itself failed - fall back to plain RGB
        # Retried here, NOT counted against `attempt`, whenever the server
        # itself is unreachable (see OllamaConnectionError's docstring - this
        # is the "outer interrupt" case: something external, most often
        # Ollama's own silent auto-updater, tore the server down mid-call).
        # It reliably comes back on its own, so this page waits for it and
        # resumes instead of burning one of its limited quality-retry
        # attempts, or worse, being marked failed for a reason that had
        # nothing to do with this page's own content.
        response = None
        while True:
            try:
                response = call_ollama_chat(
                    model=VLM_MODEL_NAME, prompt=prompt, image_path=attempt_image_path, num_ctx=num_ctx, format=format
                )
                break
            except OllamaConnectionError:
                print(
                    "[ocr] Ollama unreachable mid-call - waiting for it to come back "
                    "(likely a background update) before resuming this page...",
                    flush=True,
                )
                if wait_for_ollama_recovery():
                    print("[ocr] Ollama is back - resuming this page", flush=True)
                    continue
                print(
                    "[ocr] Ollama did not come back within the wait window - "
                    "treating this as a failed attempt for now",
                    flush=True,
                )
                break
            except OllamaUnavailableError:
                break
        if response is None:
            generation_failures += 1
            continue
        last_raw_text = response["message"]["content"]
        last_done_reason = response.get("done_reason")
        parsed = _parse_json_response(last_raw_text)
        if parsed is None:
            json_failures += 1
            continue
        last_parsed = parsed
        if is_acceptable is None or is_acceptable(parsed):
            return parsed, True, None, None, None
    if last_parsed is None:
        attempt_phrase = "attempt" if max_attempts == 1 else f"{max_attempts} attempts"
        if generation_failures and not json_failures:
            failure_reason = (
                f"Ollama could not generate a usable response ({attempt_phrase}) - this matches "
                f"the known GPU memory pressure pattern on this card's 4GB VRAM limit, often "
                f"triggered by an unusually dense/content-heavy page. Not retried further, to "
                f"avoid piling more load onto an already-stressed GPU - this page was skipped "
                f"so the rest of the document could keep processing."
            )
        elif json_failures and not generation_failures:
            failure_reason = (
                f"The model responded ({attempt_phrase}) but its output could not be parsed as "
                f"valid JSON even after repair."
            )
        else:
            failure_reason = (
                f"Extraction failed ({attempt_phrase}) - a mix of unusable Ollama responses and "
                f"unparseable output."
            )
        return last_parsed, False, failure_reason, last_raw_text, last_done_reason
    return last_parsed, False, None, None, None


def _field(d: dict, key: str) -> tuple:
    """Pulls a {"value": ..., "confidence": ...} entry out of a parsed response,
    tolerating a missing key or a malformed entry (treated as LOW/unknown rather
    than raising - a parser that crashes on one odd field loses the whole page)."""
    entry = d.get(key) if isinstance(d, dict) else None
    if not isinstance(entry, dict):
        return None, "LOW"
    value = entry.get("value")
    confidence = entry.get("confidence") if entry.get("confidence") in _VALID_CONFIDENCE else "LOW"
    return value, confidence


def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Pass 1: header metadata (well_name, operator, location, coordinates, depth,
# well_type), run against pages 1-2 and merged preferring higher confidence.
# Prompt verified against both a degraded and a clean real sample (see
# test_vlm_extraction.py / test_vlm_extraction_clean.py).
# ---------------------------------------------------------------------------
PASS1_PROMPT = """You are looking at one page of a scanned Well Completion Report (WCR).
The scan quality may be poor - some characters may be missing, faded, or corrupted.

Extract the following fields as JSON. For each field, also state your confidence
(HIGH, MEDIUM, or LOW) based on how legible that specific piece of text was.
If a field is not present on this page or fully illegible, use null for the value
and LOW for confidence. Do not guess or invent a value you cannot actually read.

Fields to extract:
- well_name (the Well ID)
- operator (company name)
- field_location (the FIELD: line, e.g. a place name - read it as written, do not
  look up or infer a place name that is not printed on the page)
- latitude (decimal degrees; include the N/S letter if shown, e.g. "23.2443 N")
- longitude (decimal degrees; include the E/W letter if shown, e.g. "72.5036 E")
- total_depth_m (the total/final measured depth in meters, as printed, with units)
- well_status_type (the document's own "Well Status" field, e.g. "Oil Producer",
  "Gas Producer", "Dry", "Injector" - extract it exactly as written, do not infer
  whether the well is currently active)

Respond with ONLY a JSON object in this exact shape, no other text:
{
  "well_name": {"value": ..., "confidence": ...},
  "operator": {"value": ..., "confidence": ...},
  "field_location": {"value": ..., "confidence": ...},
  "latitude": {"value": ..., "confidence": ...},
  "longitude": {"value": ..., "confidence": ...},
  "total_depth_m": {"value": ..., "confidence": ...},
  "well_status_type": {"value": ..., "confidence": ...}
}"""

_HEADER_FIELD_KEYS = (
    "well_name", "operator", "field_location",
    "latitude", "longitude", "total_depth_m", "well_status_type",
)
_HEADER_NUMERIC_KEYS = ("latitude", "longitude", "total_depth_m")


def _value_confidence_schema(value_types: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "value": {"type": value_types},
            "confidence": {"type": "string", "enum": list(_VALID_CONFIDENCE)},
        },
        "required": ["value", "confidence"],
    }


# Schema-constrained output (Ollama's `format` param as a JSON Schema, not just
# format="json") - verified directly this eliminates the JSON syntax/shape
# failures the regex repairs above were chasing one at a time (fences, bare
# enums, trailing commas, garbage top-level keys). Numeric-ish fields allow
# string/number/null since the model reports e.g. "23.2443 N" as a string with
# a compass letter, not a bare number - normalize_numeric() still does the
# actual parsing downstream regardless of which type comes back.
PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "well_name": _value_confidence_schema(["string", "null"]),
        "operator": _value_confidence_schema(["string", "null"]),
        "field_location": _value_confidence_schema(["string", "null"]),
        "latitude": _value_confidence_schema(["string", "number", "null"]),
        "longitude": _value_confidence_schema(["string", "number", "null"]),
        "total_depth_m": _value_confidence_schema(["string", "number", "null"]),
        "well_status_type": _value_confidence_schema(["string", "null"]),
    },
    "required": list(_HEADER_FIELD_KEYS),
}


def _empty_header() -> dict:
    return {key: (None, "LOW") for key in _HEADER_FIELD_KEYS}


def extract_header_fields(image_path: str) -> tuple:
    """Runs the Pass 1 prompt against a single page image. Returns
    (field_dict, parse_ok). field_dict maps field_name -> (value, confidence).
    Numeric fields are passed through normalize_numeric(); on parse failure
    confidence is forced to LOW regardless of what the model claimed (see
    utils.normalize_numeric's docstring).

    parse_ok=False means the model's response wasn't valid JSON at all even
    after a retry (e.g. a truncated/runaway response - observed directly during
    development) - this is NOT the same as "the model legitimately found
    nothing here" and callers must not treat it as an ordinary empty result
    (see parse_document)."""
    parsed, _accepted, _reason, _raw, _done_reason = _call_and_parse(
        PASS1_PROMPT, image_path, VLM_NUM_CTX_PASS1, format=PASS1_SCHEMA
    )
    if parsed is None:
        return _empty_header(), False

    result = {key: _field(parsed, key) for key in _HEADER_FIELD_KEYS}
    for key in _HEADER_NUMERIC_KEYS:
        value, confidence = result[key]
        numeric_value, ok = normalize_numeric(value)
        if not ok:
            confidence = "LOW"
        result[key] = (numeric_value, confidence)
    return result, True


_CONFIDENCE_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0}


def _merge_field(current: tuple, candidate: tuple) -> tuple:
    """Prefers the higher-confidence non-null reading between two pages' Pass 1
    results for the same field. Ties (or a null current value) fall through to
    the candidate so a later page can fill in what an earlier page missed."""
    cur_value, cur_conf = current
    cand_value, cand_conf = candidate
    if cur_value is not None and _CONFIDENCE_RANK.get(cur_conf, 0) >= _CONFIDENCE_RANK.get(cand_conf, 0):
        return current
    if cand_value is not None:
        return candidate
    return current


# ---------------------------------------------------------------------------
# Pass 2: full page text transcription + NPT hazard extraction, run on every page.
# Prompt verified against a real degraded sample (test_vlm_pass2_ctx.py) -
# correctly transcribed a table + a ~30-line supplementary-notes block and
# tagged an ambiguous hazard depth reading LOW confidence.
# ---------------------------------------------------------------------------
PASS2_PROMPT = """You are looking at one page of a scanned Well Completion Report (WCR).
Scan quality may be poor - some characters may be missing, faded, or corrupted.

1. Transcribe the full readable text content of this page (tables and prose).
   Transcribe ALL the way to the bottom of the page - continue through every
   remaining paragraph and every remaining table row, even if the content is
   dense, repetitive, or hard to read. Do not stop early or summarize instead
   of transcribing; an incomplete transcription is worse than a slow one.
   BEFORE you finish, look at the ENTIRE page one more time and check for any
   table BELOW the prose paragraphs you have already transcribed - a page
   finishing with a complete, well-formed paragraph is NOT a sign you are
   done; many pages have one or more tables after the prose, and you must
   transcribe every one of them too, not stop once the prose reads as complete.
   A chart, graph, plot, or diagram (anything with plotted lines, bars, axes,
   or a legend, rather than rows of labeled text) is captured separately as
   an image and is NOT part of your job - do not transcribe its plotted data,
   its axis labels/gridlines, or attempt to describe it, and do NOT treat its
   axes or gridlines as a table. Skip straight past it and continue
   transcribing whatever real text (prose or tables) comes after it on the
   page. Only its caption/title text, if present as actual printed text near
   it, belongs in your transcription - as a plain line, not a table row.
   If you see ANY table or form made of actual printed text - including a
   simple two-column "label / value" layout (e.g. a document header block
   with rows like "Well status: Shut-in", "Total depth: 2,140 m") - you MUST
   put each row on its own line and separate every column with a "|"
   character, for example: "Well status | Shut-in" on one line, "Total depth
   | 2,140 m" on the next line. Do NOT merge multiple table/form rows into a
   single run-on sentence - every distinct row in the image must become its
   own line in your output, even if that makes the transcription longer. If
   there are multiple separate tables on the page, transcribe ALL of them,
   not just the first one.
2. Identify any Non-Productive Time (NPT) hazards mentioned on this page
   (e.g. Stuck Pipe, Lost Circulation, Gas Kick, Washout), each with its depth
   in meters if stated, and a confidence (HIGH, MEDIUM, LOW) for the depth reading.

Respond with ONLY a JSON object in this exact shape, no other text:
{
  "page_text": "...",
  "hazards": [
    {"hazard_type": "...", "depth_m": {"value": ..., "confidence": "..."}, "description": "..."}
  ]
}
If no hazards are mentioned, use an empty array for "hazards"."""

# Schema-constrained output (see PASS1_SCHEMA's comment above for why - same
# reasoning applies here). depth_m.value allows string/number/null for the
# same reason as Pass 1's numeric-ish fields.
PASS2_SCHEMA = {
    "type": "object",
    "properties": {
        "page_text": {"type": "string"},
        "hazards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard_type": {"type": "string"},
                    "depth_m": _value_confidence_schema(["string", "number", "null"]),
                    "description": {"type": "string"},
                },
                "required": ["hazard_type", "depth_m", "description"],
            },
        },
    },
    "required": ["page_text", "hazards"],
}

# Self-consistency guard against a severe, reproducible failure mode observed
# directly during development: on a page with NO hazard content at all (a
# parameter-table/executive-summary page), the model fabricated four separate
# hazard records - Stuck Pipe, Lost Circulation, Gas Kick, Washout, all at the
# same depth, all tagged HIGH confidence, each with a near-identical templated
# description ("...a significant depth for potential X issues"). This is a
# different failure than the documented legibility-vs-correctness gap - that
# was misreading real text; this is inventing events with no textual basis at
# all, and confidently. NEEDS_REVIEW / LOW-confidence exclusion do NOT catch
# this, since it comes through confidently wrong.
#
# Fix, part 1: a hazard is only trusted if its own hazard_type keyword actually
# appears in THIS SAME response's page_text. Cheap (no extra VLM call),
# deterministic, and matches the project's existing pattern of not trusting
# free-form VLM judgment for safety-relevant data (see severity/status).
#
# Fix, part 2 (added after a full 20-page run): the keyword check alone wasn't
# enough - it let through hazard_type="Lost Time Incidents" from page 18's HSE
# summary ("...zero Lost Time Incidents (LTI)..."), because that phrase really
# does appear on the page - it's just describing a safety METRIC, not an NPT
# drilling hazard event. A loose "does this word appear anywhere" check can't
# tell those apart; only a closed set of real hazard categories can. So
# hazard_type is now matched against the HAZARD_CATALOG whitelist (defined
# above, shared with lookup_severity) rather than accepted as free-form model
# output - a hazard_type outside this set is rejected outright, regardless of
# what page_text contains.
#
# LIMITATION, understood and accepted rather than silently glossed over: this
# is exact (case-insensitive) substring matching against a small hand-picked
# synonym list per category - it does NOT tolerate the model's own
# transcription corrupting the hazard keyword itself (e.g. page_text reading
# "circu_ation" instead of "circulation" would fail to match even if
# hazard_type came back clean as "Lost Circulation"). This trades fabrication
# risk for a real false-negative risk on genuinely garbled keyword spans - it
# is not fuzzy/typo-tolerant. In testing on this project's synthetic sample,
# real hazard mentions survived because word-level structure stayed mostly
# intact even where individual letters were dropped, but this is a property of
# THIS corruption pattern, not a guarantee for other scans. Accepted for now
# because NEEDS_REVIEW + human review is the intended safety net for exactly
# this kind of uncertainty (same reasoning as the confidence-vs-correctness
# limitation) - not something to solve by loosening the match and reopening
# the fabrication hole part 1/2 above just closed.
def _hazard_mentioned_in_text(hazard_type: Optional[str], page_text: str) -> bool:
    if not hazard_type or not page_text:
        return False
    entry = HAZARD_CATALOG.get(hazard_type.strip().lower())
    if entry is None:
        return False
    haystack = page_text.lower()
    return any(needle in haystack for needle in entry["keywords"])


# ---------------------------------------------------------------------------
# Completeness heuristic - catches the failure mode verified directly during
# the OCR-review investigation: on a dense page (real example: the Section 5.0
# Cementing page of the Gujarat sample), the model hit its own natural stop
# token partway through transcription (Ollama reported done_reason="stop",
# eval_count=549 - nowhere near num_ctx=8192 or num_predict=2048, so this is
# NOT a context/token-budget bug, it's the model choosing to quit early) and
# schema-constrained decoding then closed out otherwise-valid JSON around the
# incomplete string. json.loads succeeds, parse_ok=True, and nothing else in
# the pipeline would ever notice.
#
# This is a heuristic, not a guarantee: it flags text that doesn't end on a
# sentence/table boundary as "possibly truncated". Trailing table-formatting
# characters (pipes, dashes, tabs) are stripped first since a table's last row
# often ends in "|" rather than punctuation (observed directly in this
# project's real samples) - that is not truncation, it's just how a table row
# looks. A false positive here costs one extra retry attempt, which is a cheap
# price against silently keeping a truncated transcription.
# ---------------------------------------------------------------------------
_TERMINAL_CHARS = (".", "!", "?", '"', "'", ")")
_TABLE_TRAILING_CHARS = "| \t-"


def looks_incomplete(page_text: Optional[str]) -> bool:
    if not page_text:
        return False  # an empty page is a separate, already-handled case - not a truncation
    stripped = page_text.rstrip().rstrip(_TABLE_TRAILING_CHARS).rstrip()
    if not stripped:
        return False
    return not stripped.endswith(_TERMINAL_CHARS)


def _page_text_acceptable(parsed: dict) -> bool:
    return not looks_incomplete(parsed.get("page_text") if isinstance(parsed, dict) else None)


# ---------------------------------------------------------------------------
# Salvage for a response that hit Ollama's num_predict cap mid-generation
# (done_reason="length") and was cut off before ever becoming valid JSON -
# previously this meant the ENTIRE page was discarded (parse_ok=False, zero
# text), even when almost all of it had already been generated. Verified
# directly (document_id=1000010, page 4, a dense ONGC well completion report
# page with a full hazard/NPT table plus a long structured-summary section):
# the model completed page_text in full and was only cut off partway through
# the LAST hazard's description field, deep inside the hazards array - the
# transcription itself was complete and valid, just trailing JSON was not.
#
# Two patterns are tried, matching the two places a cutoff can land:
#   1. page_text closed properly (a real closing quote followed by the
#      "hazards" key) and the cutoff happened later, inside hazards - this is
#      the common case, and salvage here is high-confidence since page_text
#      is genuine, complete, valid JSON string content.
#   2. page_text itself was still being generated when the cutoff hit, so it
#      never closed - the fallback takes everything after the opening quote
#      to the end of the raw response as an unterminated string, trimming a
#      trailing incomplete escape sequence if the cutoff landed mid-escape.
# Either way, hazards are never salvaged from a truncated response - an
# in-progress hazard object cannot be trusted as a complete, corroborated
# reading, and dropping hazards on a partially-recovered page is a much
# smaller loss than discarding its entire transcribed text.
# ---------------------------------------------------------------------------
_PAGE_TEXT_CLOSED_RE = re.compile(r'"page_text"\s*:\s*"(.*?)"\s*,\s*"hazards"', re.DOTALL)
_PAGE_TEXT_OPEN_RE = re.compile(r'"page_text"\s*:\s*"', re.DOTALL)
_MIN_SALVAGE_LENGTH = 200


def _salvage_truncated_page_text(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None

    closed_match = _PAGE_TEXT_CLOSED_RE.search(raw_text)
    if closed_match:
        try:
            salvaged = json.loads('"' + closed_match.group(1) + '"')
        except json.JSONDecodeError:
            salvaged = None
        if salvaged and len(salvaged.strip()) >= _MIN_SALVAGE_LENGTH:
            return salvaged.strip()

    open_match = _PAGE_TEXT_OPEN_RE.search(raw_text)
    if not open_match:
        return None
    fragment = raw_text[open_match.end():]
    for trim in range(6):
        candidate = fragment[: len(fragment) - trim] if trim else fragment
        try:
            salvaged = json.loads('"' + candidate + '"')
        except json.JSONDecodeError:
            continue
        salvaged = salvaged.strip()
        return salvaged if len(salvaged) >= _MIN_SALVAGE_LENGTH else None
    return None


# ---------------------------------------------------------------------------
# A second, DIFFERENT completeness signal from looks_incomplete() above -
# found directly against real data (document_id=16 page 6), not assumed: the
# model can decide it is "done" transcribing right after a clean, complete
# sentence, while silently never attempting real content further down the
# page - in that real case, two entire tables, two chart captions, and a
# footer line, none of it garbled or cut off mid-word, just never
# transcribed at all. parse_ok is True and looks_incomplete() finds nothing
# wrong, because the text genuinely ends on a proper sentence boundary - the
# model's own output gives no signal that anything is missing.
#
# looks_incomplete() cannot catch this by construction (it only checks
# whether the LAST sentence looks cut off), so this needs an independent
# signal that does not come from the model's own text at all: the CV-based
# figure/table detector (figure_extractor.py) runs on the page's pixels
# directly, blind to what the model claims it transcribed. If that detector
# finds a real, non-whole-page visual region (a table or chart shape) but
# the transcribed text contains no pipe-delimited table structure anywhere,
# the two independent signals disagree - and the pixel-based one is trusted
# here, since it did not come from the same model call that may have
# stopped early.
# ---------------------------------------------------------------------------
def _has_non_page_figures(figures: list, image_path: str) -> bool:
    """True if at least one figure_extractor.py detection on this page is a
    real, smaller region - not just the whole page mistaken for one (the
    known, separately-documented, still-unfixed bounding-box-vs-pixel-count
    bug in figure_extractor.py). A whole-page false positive would otherwise
    make this check fire on every page that has one, which defeats its
    purpose - so it is filtered out here rather than trusted as a signal."""
    if not figures:
        return False
    from PIL import Image

    with Image.open(image_path) as img:
        page_area = img.width * img.height
    for figure in figures:
        x0, y0, x1, y1 = figure["bbox"]
        figure_area = (x1 - x0) * (y1 - y0)
        if page_area > 0 and figure_area < 0.6 * page_area:
            return True
    return False


def page_missing_visible_content(page_text: str, figures: list, image_path: str) -> bool:
    """
    True when the page's rendered image visibly has a real (non-whole-page)
    table/chart-shaped region, but the transcribed page_text contains no
    pipe-delimited table structure at all - the signature of the failure
    mode described above. Callers should treat this the same as a parse
    failure for review-flagging purposes: a clean-looking, fully-parsed
    result can still be missing real content.
    """
    if not _has_non_page_figures(figures, image_path):
        return False
    return len(table_parser.extract_tables(page_text)) == 0


def extract_page_content(image_path: str) -> dict:
    """Runs the Pass 2 prompt against a single page image. Returns
    {"page_text": str, "hazards": [...], "parse_ok": bool, "text_possibly_truncated": bool,
    "failure_reason": Optional[str]}.
    A hazard whose depth fails numeric normalization keeps its LOW-downgraded
    confidence rather than being dropped here - the caller decides whether to
    store it (see parse_document: only a totally unparseable depth is dropped).
    A hazard not corroborated anywhere in this page's own page_text is dropped
    outright here, not just downgraded - see _hazard_mentioned_in_text above.

    parse_ok=False (even after the retry in _call_and_parse) means this page's
    text/hazards could not be extracted at all - NOT the same as "genuinely no
    hazards on this page", and callers must not silently treat it as a clean
    empty page (see parse_document). failure_reason carries a short
    human-readable explanation for this case (see _call_and_parse) so a UI can
    surface WHY a specific page failed instead of leaving it indistinguishable
    from a page that simply hasn't been reached yet - found as a real gap:
    previously a hard-failed page's row still existed in page_extractions but
    had no failure detail attached anywhere, and the frontend's status badge
    didn't check parse_ok at all, so a fully failed page could show a
    misleading "Extracted" badge.

    text_possibly_truncated=True means every retry attempt produced valid JSON
    (parse_ok is still True) but the page_text still looked incomplete by the
    looks_incomplete() heuristic above - see that function's docstring. Callers
    must treat this the same as a parse failure for review-flagging purposes
    (see run_extraction's needs_review computation): the JSON being valid does
    NOT mean the transcription is complete.

    max_attempts=1: deliberately no retry/grayscale-fallback here (unlike
    _call_and_parse's own default of 3) - each attempt is a full VLM
    generation call and costs real GPU cycles on this 4GB card, so retrying a
    page that just failed under memory pressure piles more load onto a card
    that's already stressed instead of relieving it. One bounded attempt
    (still capped by call_ollama_chat's num_predict, so it can't run forever)
    gives the page a real chance without compounding the very pressure that
    caused the failure. The connection-drop wait-and-resume path inside
    _call_and_parse is unaffected by this - waiting for Ollama's server to
    come back after an auto-update kill costs no GPU cycles, so that still
    happens regardless of max_attempts."""
    parsed, accepted, failure_reason, raw_text, done_reason = _call_and_parse(
        PASS2_PROMPT,
        image_path,
        VLM_NUM_CTX_PASS2,
        format=PASS2_SCHEMA,
        max_attempts=1,
        is_acceptable=_page_text_acceptable,
    )
    if parsed is None:
        if done_reason == "length":
            salvaged_text = _salvage_truncated_page_text(raw_text)
            if salvaged_text is not None:
                return {
                    "page_text": salvaged_text,
                    "hazards": [],
                    "parse_ok": True,
                    "text_possibly_truncated": True,
                    "failure_reason": None,
                }
            failure_reason = (
                "This page's content exceeded the model's per-page output limit and was cut "
                "off before enough of it could be recovered - the page is unusually dense "
                "(a lot of text/table content to transcribe in one pass)."
            )
        return {
            "page_text": "",
            "hazards": [],
            "parse_ok": False,
            "text_possibly_truncated": False,
            "failure_reason": failure_reason,
        }

    page_text = parsed.get("page_text") or ""
    hazards_raw = parsed.get("hazards")
    hazards = []
    if isinstance(hazards_raw, list):
        for raw_hazard in hazards_raw:
            if not isinstance(raw_hazard, dict):
                continue
            hazard_type = raw_hazard.get("hazard_type")
            if not _hazard_mentioned_in_text(hazard_type, page_text):
                continue
            depth_value, depth_confidence = _field(raw_hazard, "depth_m")
            depth_numeric, ok = normalize_numeric(depth_value)
            if not ok:
                depth_confidence = "LOW"
            if depth_numeric is not None and depth_numeric <= 0:
                # A depth of zero or negative is never a legitimate downhole
                # hazard reading in this domain (every well here is thousands
                # of meters deep) - observed directly as garbage numeric parses
                # (0.0, then later -1.4286 on a different run) rather than real
                # readings. Originally only guarded == 0; broadened to <= 0
                # after the negative case turned up - same failure class, not
                # a new one. Treat it the same as "failed to parse" (None) so
                # it flows through parse_document's existing "no parseable
                # depth -> drop" path, rather than inventing a semantics for a
                # value that shouldn't occur.
                depth_numeric = None
            hazards.append(
                {
                    "hazard_type": hazard_type,
                    "depth_m": depth_numeric,
                    "confidence": depth_confidence,
                    "description": raw_hazard.get("description"),
                }
            )
    return {
        "page_text": page_text,
        "hazards": hazards,
        "parse_ok": True,
        "text_possibly_truncated": not accepted,
        "failure_reason": None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def register_document(pdf_path: str, pages: list, doc_type: str = "WCR") -> dict:
    """
    Fast, synchronous phase: hash the file, dedup-check it, and create the
    `documents` row. Split out from the old all-in-one parse_document() so a
    router can call this directly and hand the client a document_id to poll
    immediately, before handing run_extraction() (the slow VLM part, minutes
    not milliseconds) to a background task - per the finalized upload design
    (background task + document ID for polling, not a blocking request).

    Returns {"status": "REJECTED_DUPLICATE", "document_id": ...} if an
    identical file was already ingested (run_extraction should NOT be called
    in that case - there is nothing new to extract), else
    {"status": "PROCESSING", "document_id": ...}.
    """
    filename = Path(pdf_path).name
    file_hash = hash_file(pdf_path)

    existing_doc = database.find_document_by_hash(file_hash)
    if existing_doc is not None:
        return {
            "status": "REJECTED_DUPLICATE",
            "message": f"Identical file already ingested as document_id={existing_doc['document_id']}",
            "document_id": existing_doc["document_id"],
        }

    document_id = database.create_document(
        filename=filename,
        doc_type=doc_type,
        file_path=pdf_path,
        page_count=len(pages),
        file_hash=file_hash,
    )
    return {"status": "PROCESSING", "document_id": document_id}


def run_extraction(document_id: int, pages: list) -> dict:
    """
    The slow phase: Pass 1 + Pass 2 across all pages, well/hazard creation,
    final status update. Assumes register_document() already created the
    `documents` row for document_id - does not create or dedup-check it again.

    pages: pdf_processor.rasterize_pdf()'s output - a list of
           {"page_num": int, "image_path": str, ...} for pages already rendered
           to disk. This module does not rasterize; that stays pdf_processor's
           job (already built and tested).

    Returns a summary dict describing what was inserted and the document's
    final status - COMPLETE or NEEDS_REVIEW.
    """
    document = database.get_document(document_id)
    pdf_path = document["file_path"]

    # Persist the artifact->page mapping now, for every page, regardless of
    # extraction outcome - image_artifacts existed in the schema but nothing
    # ever wrote to it before, so there was no way to look up a document's
    # page images from the database at all.
    #
    # Also runs figure_extractor's CV heuristic here, per page, right after
    # rasterization - deliberately independent of and NOT gated on Pass 1/
    # Pass 2's VLM extraction below (no _VLM_LOCK contention, purely visual,
    # see figure_extractor.py's module docstring for why a heuristic over a
    # rendered PNG was chosen over a pretrained layout model or a VLM call).
    doc_slug = f"doc{document_id}"
    figures_by_page = {}  # page_num -> figures list, reused by Pass 2's page_missing_visible_content() check below
    _log(document_id, f"Starting extraction: {len(pages)} page(s)")
    for page in pages:
        database.create_image_artifact(
            well_id=None,
            document_id=document_id,
            page_num=page["page_num"],
            artifact_type="Full Page",
            artifact_kind="PAGE_RENDER",
            file_path=page["image_path"],
            # .get(), not [] - pages passed in by one-off scripts (run_ingestion.py)
            # only carry page_num/image_path, not width/height; nullable columns, fine either way.
            width=page.get("width"),
            height=page.get("height"),
        )
        figures = figure_extractor.save_figures_for_page(
            page["image_path"], doc_slug=doc_slug, page_num=page["page_num"]
        )
        figures_by_page[page["page_num"]] = figures
        for figure in figures:
            database.create_image_artifact(
                well_id=None,
                document_id=document_id,
                page_num=page["page_num"],
                artifact_type="Detected Figure",
                artifact_kind="EMBEDDED_FIGURE",
                file_path=figure["file_path"],
                bbox=figure["bbox"],
                width=figure["width"],
                height=figure["height"],
            )
        _log(
            document_id,
            f"Page {page['page_num']}/{len(pages)}: rendered, {len(figures)} figure(s) detected",
        )

    # --- Pass 1: header metadata from pages 1-2, merged field-by-field ---
    header = _empty_header()
    any_pass1_parse_failed = False
    pass1_pages_failed_to_parse = []
    for page in pages[:2]:
        page_header, parse_ok = extract_header_fields(page["image_path"])
        if not parse_ok:
            any_pass1_parse_failed = True
            pass1_pages_failed_to_parse.append(page["page_num"])
        database.upsert_page_extraction(
            document_id=document_id,
            page_num=page["page_num"],
            parse_ok=parse_ok,
            header_fields_json=json.dumps(page_header),
        )
        for key in _HEADER_FIELD_KEYS:
            header[key] = _merge_field(header[key], page_header[key])
        _log(
            document_id,
            f"Pass 1 - page {page['page_num']}/2: header extraction {'OK' if parse_ok else 'FAILED'}",
        )

    well_name_value, _well_name_confidence = header["well_name"]
    operator_value, _ = header["operator"]
    depth_value, _ = header["total_depth_m"]
    well_type_value, _ = header["well_status_type"]
    field_location_value, _ = header["field_location"]
    lat_value, lat_confidence = header["latitude"]
    lon_value, lon_confidence = header["longitude"]

    # Design decision: LOW-confidence or missing coordinates never participate
    # in hazard_monitor.py's 10km radius check. That is enforced by
    # location_verified rather than by discarding the value - we never throw
    # away a reading, just refuse to treat it as trustworthy until reviewed.
    location_verified = lat_value is not None and lon_value is not None and (
        lat_confidence == "HIGH" and lon_confidence == "HIGH"
    )

    # --- Pass 2: every page, transcribed text + hazards ---
    all_page_texts = []
    hazards_by_page = []  # list of (page_num, hazard_dict)
    pages_failed_to_parse = []
    pages_with_incomplete_text = []
    pages_missing_visible_content = []
    for page in pages:
        content = extract_page_content(page["image_path"])
        if not content["parse_ok"]:
            pages_failed_to_parse.append(page["page_num"])
        if content.get("text_possibly_truncated"):
            pages_with_incomplete_text.append(page["page_num"])
        missing_content = content["parse_ok"] and page_missing_visible_content(
            content["page_text"], figures_by_page.get(page["page_num"], []), page["image_path"]
        )
        if missing_content:
            pages_missing_visible_content.append(page["page_num"])
        database.upsert_page_extraction(
            document_id=document_id,
            page_num=page["page_num"],
            page_text=content["page_text"],
            parse_ok=content["parse_ok"],
            failure_reason=content.get("failure_reason"),
        )
        all_page_texts.append(content["page_text"])
        for hazard in content["hazards"]:
            hazards_by_page.append((page["page_num"], hazard))
        status_flags = []
        if not content["parse_ok"]:
            status_flags.append("FAILED")
        elif content.get("text_possibly_truncated"):
            status_flags.append("POSSIBLY TRUNCATED")
        elif missing_content:
            status_flags.append("POSSIBLY MISSING TABLE/CHART CONTENT")
        else:
            status_flags.append("OK")
        if content["hazards"]:
            status_flags.append(f"{len(content['hazards'])} hazard(s) found")
        _log(
            document_id,
            f"Pass 2 - page {page['page_num']}/{len(pages)}: OCR {', '.join(status_flags)}",
        )

    well_status = derive_well_status(all_page_texts)
    enrichment = derive_well_enrichment(all_page_texts)

    # --- Dedup: same well_name (case-insensitive/trimmed) -> reuse the well,
    # link this document to it, don't create a duplicate well row ---
    existing_well = database.find_well_by_name(well_name_value) if well_name_value else None
    if existing_well is not None:
        well_id = existing_well["well_id"]
        database.update_well_enrichment(well_id, **enrichment)
    else:
        well_id = database.create_well(
            well_name=well_name_value or f"UNKNOWN ({document['filename']})",
            operator=operator_value,
            latitude=lat_value,
            longitude=lon_value,
            total_depth_m=depth_value,
            document_id=document_id,
            file_path=pdf_path,
            status=well_status,
            location_verified=location_verified,
            well_type=well_type_value,
            field_location=field_location_value,
            formation=enrichment["formation"],
            mud_type=enrichment["mud_type"],
            mud_weight=enrichment["mud_weight"],
            casing_notes=enrichment["casing_notes"],
            cementing_notes=enrichment["cementing_notes"],
            reservoir_notes=enrichment["reservoir_notes"],
        )
    database.link_document_to_well(document_id, well_id)

    # --- Index every page's transcribed text into Qdrant for DrillMind's RAG
    # retrieval - done here (after well_id is known) rather than inside the
    # Pass 2 loop above, since indexing wants the real well_id for citation
    # metadata, not None. Found and fixed as a real gap, not built
    # defensively in advance: this call was documented in this module's own
    # header docstring as "vector_store.py's job, not built yet" but nothing
    # ever actually invoked it from the ingestion pipeline - every document
    # processed before this fix has real transcribed text in SQLite that was
    # never searchable via DrillMind at all (verified directly: Qdrant held
    # only 2 indexed chunks, both leftover from early manual testing, before
    # this fix and the accompanying backfill for already-processed documents).
    # Qdrant's embedded storage only allows one process to hold it open at a
    # time - a real, reproducible conflict when a standalone script (e.g. a
    # resume/recovery run) executes while the FastAPI server is also up.
    # Indexing is a supplementary RAG feature, not core to the document's
    # OCR/hazard record, so a lock conflict here must not take down the rest
    # of a multi-minute extraction run that already did the expensive work -
    # caught and logged instead of raised, matching this file's existing
    # resilience pattern (grayscale retry, bounded retries) rather than
    # letting one non-critical side effect crash everything after it.
    pages_indexed = 0
    try:
        for page, page_text in zip(pages, all_page_texts):
            if vector_store.index_page(
                well_id=well_id, document_id=document_id, page_num=page["page_num"], page_text=page_text
            ):
                pages_indexed += 1
        _log(document_id, f"Indexed {pages_indexed}/{len(pages)} page(s) into the vector store")
    except Exception as exc:
        _log(document_id, f"Vector indexing skipped - Qdrant unavailable ({exc.__class__.__name__}: {exc})")

    # --- Insert hazards. A hazard with no parseable depth at all is dropped -
    # not stored as safety data, since depth is the entire point of a hazard
    # record (see hazard_monitor.py's design: it correlates on depth). A hazard
    # with a LOW-confidence depth IS stored (never discard a reading), just
    # excluded from live alerting by hazard_monitor.py's confidence filter. ---
    hazard_ids = []
    for page_num, hazard in hazards_by_page:
        if hazard["depth_m"] is None:
            continue
        hazard_id = database.create_hazard(
            well_id=well_id,
            hazard_type=hazard.get("hazard_type") or "Unknown",
            depth_m=hazard["depth_m"],
            confidence=hazard.get("confidence", "LOW"),
            severity=lookup_severity(hazard.get("hazard_type")),
            page_num=page_num,
            description=hazard.get("description"),
        )
        hazard_ids.append(hazard_id)

    # --- Final document status: NEEDS_REVIEW is a normal, expected outcome for
    # a degraded scan, not a rare failure case (see project design principle).
    # A page that flat-out failed to parse (even after retry) forces review too -
    # it is NOT the same as a page with legitimately no hazards, and pretending
    # otherwise would let a page's real content go silently unextracted. ---
    well_name_ok = well_name_value is not None and header["well_name"][1] in ("HIGH", "MEDIUM")
    any_low_confidence_hazard = any(h.get("confidence") == "LOW" for _, h in hazards_by_page)
    # A page that parsed as valid JSON but still looks truncated (see
    # extract_page_content's text_possibly_truncated) is NOT a clean result -
    # forces review the same as an outright parse failure, since the model
    # quitting mid-transcription is a real content gap, not just a legibility
    # confidence question.
    any_parse_failure = (
        any_pass1_parse_failed
        or bool(pages_failed_to_parse)
        or bool(pages_with_incomplete_text)
        or bool(pages_missing_visible_content)
    )
    needs_review = not well_name_ok or not location_verified or any_low_confidence_hazard or any_parse_failure
    final_status = "NEEDS_REVIEW" if needs_review else "COMPLETE"
    database.update_document_status(document_id, final_status)
    _log(document_id, f"Extraction complete: status={final_status}, well={well_name_value!r}, hazards_inserted={len(hazard_ids)}")

    return {
        "status": final_status,
        "document_id": document_id,
        "well_id": well_id,
        "well_name": well_name_value,
        "location_verified": location_verified,
        "hazards_inserted": len(hazard_ids),
        "pages_processed": len(pages),
        "pass1_pages_failed_to_parse": pass1_pages_failed_to_parse,
        "pass2_pages_failed_to_parse": pages_failed_to_parse,
        "pages_with_incomplete_text": pages_with_incomplete_text,
        "pages_missing_visible_content": pages_missing_visible_content,
    }


def resume_extraction(document_id: int, pages: list) -> dict:
    """
    Re-runs extraction ONLY for pages that don't already have valid, parsed
    content - skips every page that already succeeded, rather than redoing
    an entire multi-page document from scratch. Built for a real failure
    mode this project hit in practice, not a hypothetical: Ollama's
    background auto-updater killing the server mid-run (see config.py's
    OLLAMA_HOST comment) showed up as document_id=15's real run - pages 1-8
    of 33 succeeded normally, then pages 9-33 ALL failed with no recovery,
    consistent with the server going down partway through and staying down
    for the rest of that run. Re-running the whole document via
    register_document()+run_extraction() would hit the duplicate-hash check
    and refuse to process it again anyway, and would wastefully redo the
    pages that already worked.

    Assumes register_document() + at least one prior run_extraction() or
    resume_extraction() call already created the documents/wells rows for
    this document_id - this function only fills gaps, it never creates them.

    pages: the full page list (page_num + image_path) for every page in the
    document, typically reconstructed from the image_artifacts rows already
    recorded for this document_id (see resume_ingestion.py) rather than
    re-rasterizing - pages that already succeeded are looked up and skipped
    by page_num, their image_path is only used if a retry is actually needed.

    Does NOT retroactively fix well-record fields (operator, coordinates,
    etc.) if Pass 1 originally failed and only now succeeds on resume - that
    is a real, accepted scope limit, not an oversight: merging newly-
    recovered header fields into an already-created well record safely
    (without risking overwriting a field with a worse reading) is a
    meaningfully different problem than resuming Pass 2 text/hazard
    extraction, and wasn't the failure this was built to fix. A resume that
    still has Pass 1 failures logs a note; NEEDS_REVIEW is forced either way.
    """
    document = database.get_document(document_id)
    well_id = document.get("well_id")
    existing_by_page = {e["page_num"]: e for e in database.get_page_extractions_for_document(document_id)}

    # Figure detection already ran for every page during the original
    # run_extraction() call (that loop is unconditional, independent of
    # whether Pass 2 later failed) - reuse those existing rows for
    # page_missing_visible_content() below rather than re-running the CV
    # detector here.
    figures_by_page: dict = {}
    for artifact in database.get_image_artifacts_for_document(document_id):
        if artifact.get("artifact_kind") == "EMBEDDED_FIGURE":
            figures_by_page.setdefault(artifact["page_num"], []).append(
                {"bbox": (artifact["bbox_x0"], artifact["bbox_y0"], artifact["bbox_x1"], artifact["bbox_y1"])}
            )

    _log(document_id, f"Resuming extraction: checking {len(pages)} page(s) for gaps")

    # --- Resume Pass 1 only for pages 1-2 that previously failed ---
    pass1_pages_still_failed = []
    for page in pages[:2]:
        existing = existing_by_page.get(page["page_num"])
        if existing and existing.get("parse_ok"):
            continue  # already succeeded - trust it, don't re-call the VLM
        page_header, parse_ok = extract_header_fields(page["image_path"])
        if not parse_ok:
            pass1_pages_still_failed.append(page["page_num"])
        database.upsert_page_extraction(
            document_id=document_id,
            page_num=page["page_num"],
            parse_ok=parse_ok,
            header_fields_json=json.dumps(page_header),
        )
        _log(document_id, f"Resume Pass 1 - page {page['page_num']}/2: header extraction {'OK' if parse_ok else 'FAILED'}")
    if pass1_pages_still_failed:
        _log(
            document_id,
            f"NOTE: Pass 1 still failed on page(s) {pass1_pages_still_failed} after resume - "
            f"well record fields are not retroactively updated by resume_extraction(), review manually",
        )

    # --- Resume Pass 2 only for pages that don't already have real text ---
    pages_failed_to_parse = []
    pages_with_incomplete_text = []
    pages_missing_visible_content = []
    hazard_ids = []
    resumed_page_count = 0
    for page in pages:
        existing = existing_by_page.get(page["page_num"])
        if existing and existing.get("parse_ok") and existing.get("page_text"):
            continue  # already has real transcribed text - skip, don't re-call the VLM
        resumed_page_count += 1
        content = extract_page_content(page["image_path"])
        if not content["parse_ok"]:
            pages_failed_to_parse.append(page["page_num"])
        if content.get("text_possibly_truncated"):
            pages_with_incomplete_text.append(page["page_num"])
        missing_content = content["parse_ok"] and page_missing_visible_content(
            content["page_text"], figures_by_page.get(page["page_num"], []), page["image_path"]
        )
        if missing_content:
            pages_missing_visible_content.append(page["page_num"])
        database.upsert_page_extraction(
            document_id=document_id,
            page_num=page["page_num"],
            page_text=content["page_text"],
            parse_ok=content["parse_ok"],
            failure_reason=content.get("failure_reason"),
        )
        if well_id is not None:
            for hazard in content["hazards"]:
                if hazard["depth_m"] is None:
                    continue
                hazard_id = database.create_hazard(
                    well_id=well_id,
                    hazard_type=hazard.get("hazard_type") or "Unknown",
                    depth_m=hazard["depth_m"],
                    confidence=hazard.get("confidence", "LOW"),
                    severity=lookup_severity(hazard.get("hazard_type")),
                    page_num=page["page_num"],
                    description=hazard.get("description"),
                )
                hazard_ids.append(hazard_id)
        status_flags = ["FAILED" if not content["parse_ok"] else ("POSSIBLY TRUNCATED" if content.get("text_possibly_truncated") else ("POSSIBLY MISSING TABLE/CHART CONTENT" if missing_content else "OK"))]
        if content["hazards"]:
            status_flags.append(f"{len(content['hazards'])} hazard(s) found")
        _log(document_id, f"Resume Pass 2 - page {page['page_num']}/{len(pages)}: OCR {', '.join(status_flags)}")

    if resumed_page_count == 0:
        _log(document_id, "Nothing to resume - every page already had valid text")

    all_extractions = database.get_page_extractions_for_document(document_id)

    # --- Recompute well lifecycle status from ALL page texts (old + newly resumed), not just this resume's delta ---
    if well_id is not None:
        all_page_texts = [e.get("page_text") for e in all_extractions]
        database.update_well_status(well_id, derive_well_status(all_page_texts))
        database.update_well_enrichment(well_id, **derive_well_enrichment(all_page_texts))

    # --- Index every page with real text into Qdrant, not just the ones
    # touched by this resume - a document that predates the indexing fix
    # (see run_extraction's equivalent step) may have plenty of already-good
    # pages that were never indexed either. index_page() upserts by a
    # deterministic (document_id, page_num) ID, so re-indexing an
    # already-indexed page is a harmless no-op, not a duplicate. ---
    # See run_extraction()'s equivalent block for why this is caught rather
    # than raised: a Qdrant lock conflict with a concurrently-running server
    # is a real, reproducible case (not hypothetical), and indexing is
    # supplementary to this function's actual job of filling text/hazard gaps.
    pages_indexed = 0
    try:
        for extraction in all_extractions:
            if vector_store.index_page(
                well_id=well_id,
                document_id=document_id,
                page_num=extraction["page_num"],
                page_text=extraction.get("page_text"),
            ):
                pages_indexed += 1
        _log(document_id, f"Indexed {pages_indexed}/{len(all_extractions)} page(s) into the vector store")
    except Exception as exc:
        _log(document_id, f"Vector indexing skipped - Qdrant unavailable ({exc.__class__.__name__}: {exc})")

    # --- Recompute final document status from the FULL combined picture ---
    any_parse_failure = any(not e.get("parse_ok") for e in all_extractions)
    well = database.get_well(well_id) if well_id is not None else None
    well_name_ok = bool(well and well.get("well_name") and not str(well["well_name"]).startswith("UNKNOWN"))
    location_verified = bool(well and well.get("location_verified"))
    all_hazards = database.get_hazards_for_well(well_id) if well_id is not None else []
    any_low_confidence_hazard = any(h.get("confidence") == "LOW" for h in all_hazards)
    needs_review = (
        not well_name_ok
        or not location_verified
        or any_low_confidence_hazard
        or any_parse_failure
        or bool(pass1_pages_still_failed)
        or bool(pages_missing_visible_content)
    )
    final_status = "NEEDS_REVIEW" if needs_review else "COMPLETE"
    database.update_document_status(document_id, final_status)
    _log(
        document_id,
        f"Resume complete: status={final_status}, pages_resumed={resumed_page_count}, "
        f"still_failed={pages_failed_to_parse}",
    )

    return {
        "status": final_status,
        "document_id": document_id,
        "well_id": well_id,
        "pages_resumed": resumed_page_count,
        "hazards_inserted_this_resume": len(hazard_ids),
        "pass1_pages_still_failed": pass1_pages_still_failed,
        "pass2_pages_still_failed": pages_failed_to_parse,
        "pages_with_incomplete_text": pages_with_incomplete_text,
        "pages_missing_visible_content": pages_missing_visible_content,
    }


def parse_document(pdf_path: str, pages: list, doc_type: str = "WCR") -> dict:
    """Convenience wrapper combining register_document() + run_extraction()
    synchronously in one call - kept for test scripts/one-off runs that want
    the old all-in-one behavior. New code (the upload router) should call the
    two phases separately to get a document_id back before the slow part
    runs."""
    registration = register_document(pdf_path, pages, doc_type)
    if registration["status"] == "REJECTED_DUPLICATE":
        return registration
    return run_extraction(registration["document_id"], pages)
