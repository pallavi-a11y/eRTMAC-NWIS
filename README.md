# eRTMAC-NWIS

**Enhanced Real-Time Monitoring and Analytical Engine — Nearby Well Information System**

An offline, AI-powered document intelligence system that converts scanned Well Completion Reports
(WCRs) and drilling records into searchable, structured, hazard-aware data — built for
**Smart India Hackathon 2026, Problem Statement #26121 (Oil India Limited)**.

No cloud APIs, no internet dependency, no per-query billing: every model in this stack runs locally
via [Ollama](https://ollama.com), on hardware as modest as a 4GB-VRAM laptop GPU.

---

## Table of Contents

- [The problem](#the-problem)
- [The solution](#the-solution)
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Research references](#research-references)

---

## The problem

Decades of drilling reports sit as scattered, unsearchable paper and scans. These documents are
degraded, inconsistent, and mixed-format (typewriter, handwriting, photocopy-of-a-photocopy) — the
kind of scan quality that defeats naive OCR outright. Classical OCR tools transcribe *characters*,
but have no understanding of *meaning*: they cannot tell you which number is a depth, which phrase
is a safety hazard, or which block of text is a table. As a result, there is no way to check whether
a new well's planned location and depth previously caused hazards at nearby wells — documented risks
get rediscovered the hard way, driving costly Non-Productive Time (NPT) and threatening rig safety.
The archive is large and growing — too vast for manual digitization.

## The solution

eRTMAC-NWIS uses a locally-hosted **vision-language model (VLM)** — not classical OCR — to read each
scanned page the way a person would: recognizing *what a field means*, not just *what characters are
on it*. The result is fed into:

- A **structured, searchable corpus** (SQLite + a local vector index) that grows more valuable with
  every document added, at no additional cost.
- **DrillMind**, a retrieval-augmented (RAG) chat assistant that answers questions grounded in your
  actual ingested documents, with every answer citing its exact source page.
- An **offline well map** and a **hazard proximity checker** that instantly correlates a proposed new
  well against every historical hazard recorded near it, at a similar depth.
- A **human-in-the-loop review interface** — every page's extracted text is editable, and a saved
  correction is preferred over the model's own output for all future search and chat, so no record
  stays AI-trusted only.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Summary view: total wells, active wells, hazards found, documents ingested. |
| **Upload** | Upload a PDF; triggers the full OCR/extraction pipeline in the background. |
| **Corpus** | Browse every ingested document; view each page's source image beside its editable extracted text; download corrected text. |
| **Map Visualisation** | Every well plotted on a fully offline India basemap (no internet map tiles); search, radius-filter, and add new wells. |
| **Risk Monitor** | Enter a proposed well's position and bit depth; instantly see historical hazards recorded at nearby wells within a configurable radius and depth window. |
| **DrillMind** | Ask natural-language questions about the ingested corpus; answers are retrieval-grounded and cite the source document/page. |

## Architecture

```
PDF upload
   │
   ▼
Rasterize every page → 200 DPI PNG            (src/pdf_processor.py)
   │
   ▼
Duplicate-hash check → skip if already ingested
   │
   ├──▶ PASS 1 (pages 1–2 only): header fields       ──┐
   │     well name, operator, depth, coordinates,      │  Qwen2.5-VL 3B
   │     status — JSON-schema constrained               │  via Ollama,
   │                                                     │  serialized behind
   ├──▶ PASS 2 (every page): full text + hazards      ──┘  one global lock
   │     transcribes prose/tables, lists any NPT
   │     hazards mentioned, with depth + confidence
   │
   ├──▶ Figure/table detection (independent of the VLM)   (src/figure_extractor.py)
   │     grayscale → Otsu threshold → connected components
   │     crops charts/diagrams out as their own image files
   │
   ▼
Table reconstruction — heuristic parser turns the model's
pipe-delimited text back into real tables at display time     (src/table_parser.py)
   │
   ▼
   ├──▶ SQLite (data/app.db)         — structured fields, hazards, raw text
   └──▶ Qdrant (data/qdrant_db)      — 384-dim embeddings of every page's text
                                        (bge-small-en-v1.5, CPU)
   │
   ▼
Dashboard / Corpus / Map / Risk Monitor / DrillMind
  read from the corpus above — DrillMind additionally retrieves
  the most relevant embedded passages before answering
```

**Why a two-pass pipeline instead of one call per page:** a single call trying to extract header
fields *and* transcribe the full page *and* find hazards increases the required output length —
exactly the axis on which this constrained hardware was found to fail (a longer expected output
raises the risk of hitting the model's output cap mid-generation). Splitting header extraction
(cheap, run twice per document) from full transcription (run once per page, for the one job it has)
keeps each call's cost proportional to what it actually needs to do.

**Why heuristic table reconstruction instead of asking the model for structured tables directly:**
same reasoning — asking the VLM to *also* emit structured table JSON in the same call increases
output length and truncation risk. The heuristic in `table_parser.py` instead reconstructs tables
from text the model already had to produce anyway, at zero extra generation cost.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vision-language model | **Qwen2.5-VL 3B** (Q4_K_M quantized), served via **Ollama** | Fits a 4GB VRAM budget; fully offline; open-weight. |
| Embeddings | **bge-small-en-v1.5** (33M params, 384-dim), forced onto CPU | Keeps the entire GPU budget free for the VLM instead of competing for the same constrained VRAM. |
| Vector store | **Qdrant**, embedded/local mode | Zero-setup, no server process, matches the offline requirement — unlike a cloud vector DB. |
| Structured storage | **SQLite** | Zero-configuration, single-file, appropriate for a single-user local tool. |
| Figure/table detection | **OpenCV** (grayscale + Otsu threshold + connected components) | Independent of the language model — cannot be fooled by a model hallucination, and vice versa. |
| Backend | **FastAPI** (Python) | Async support for background ingestion; same language as every ML library in use. |
| Frontend | **React** (Vite) + **Leaflet**/`react-leaflet` | Leaflet supports a static local image basemap (`ImageOverlay`), which is what makes the map genuinely usable fully offline. |

## Project structure

```
SIH/
├── backend/                  FastAPI backend
│   ├── main.py                Entrypoint — mounts all routers, initializes the DB
│   ├── routers/                One file per API resource (corpus, upload, review, risk, wells_map, drillmind, dashboard)
│   ├── src/
│   │   ├── config.py            All paths, thresholds, and model identifiers
│   │   ├── pdf_processor.py      PDF → page image rasterization
│   │   ├── unified_parser.py     Pass 1 / Pass 2 VLM extraction, hazard fabrication guard
│   │   ├── table_parser.py       Heuristic table reconstruction from extracted text
│   │   ├── figure_extractor.py   OpenCV-based chart/diagram detection & cropping
│   │   ├── vector_store.py       Embedding + Qdrant indexing/retrieval
│   │   ├── rag_engine.py         DrillMind's grounded chat logic
│   │   ├── hazard_monitor.py     Offset-well proximity/hazard correlation
│   │   ├── database.py           SQLite schema + all queries
│   │   └── utils.py              Ollama call wrapper, numeric/coordinate parsing
│   └── requirements.txt
├── ertmac-nwis/               React (Vite) frontend
│   └── src/pages/               Dashboard, Upload, Corpus, MapVisualise, RiskMonitor, DrillMind
├── start.sh                   One command to launch backend + frontend together
├── stop.sh                    Stops both
└── README.md                  This file
```

## Prerequisites

- **Python 3.11+** with `venv`
- **Node.js 18+** and npm
- **[Ollama](https://ollama.com)** installed and running locally
- The model pulled once: `ollama pull qwen2.5vl:3b`

## Setup

```bash
# 1. Clone
git clone https://github.com/pallavi-a11y/eRTMAC-NWIS.git
cd eRTMAC-NWIS

# 2. Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows — use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# 3. Frontend
cd ../ertmac-nwis
npm install
```

## Running the app

**One command** (Windows, Git Bash), from the repo root:

```bash
bash start.sh
```

This launches the backend (`:8000`) and frontend (`:5173`) each in their own window, waits until
both respond, and opens the app in your browser. Stop both with:

```bash
bash stop.sh
```

**Manually**, in two terminals:

```bash
# Terminal 1 — backend
cd backend
venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd ertmac-nwis
npm run dev
```

Then open `http://localhost:5173`.

## API reference

All routes are served by the FastAPI backend at `http://localhost:8000`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/dashboard/summary` | Total wells, active wells, hazards, documents. |
| `GET` | `/api/corpus` | List every ingested document. |
| `GET` | `/api/corpus/{corpus_id}` | One document's metadata. |
| `GET` | `/api/corpus/{corpus_id}/file` | Download the original uploaded PDF. |
| `DELETE` | `/api/corpus/{corpus_id}` | Delete a document and its extracted data. |
| `POST` | `/api/upload` | Upload a PDF and start ingestion. |
| `GET` | `/api/review/{document_id}/pages` | Per-page extraction status for a document. |
| `GET` | `/api/review/{document_id}/{page_num}` | A page's extracted text, header fields, hazards, rendered HTML. |
| `GET` | `/api/review/{document_id}/{page_num}/image` | The rasterized page image. |
| `POST` | `/api/review/{document_id}/{page_num}/correction` | Save a human correction to a page's text. |
| `GET` | `/api/review/{document_id}/{page_num}/download` | Download a page's current text as `.txt`. |
| `GET` | `/review/{document_id}` | Standalone server-rendered review tool (outside the React app). |
| `GET` | `/api/wells` | List every well. |
| `POST` | `/api/wells` | Add a new well. |
| `POST` | `/api/wells/compare` | Compare wells. |
| `POST` | `/api/risk/telemetry` | Offset-well hazard proximity check (position + depth in, nearby hazards out). |
| `POST` | `/api/chat` | DrillMind's RAG-grounded chat endpoint. |

## Configuration

Every tunable value lives in `backend/src/config.py`, not scattered across files. The one
environment variable read at runtime:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where the backend looks for the Ollama server. |

Key constants (measured against real data, not assumed defaults — see the comments in
`config.py` for the specific test cases each was tuned against):

- `RENDER_DPI = 200` — page rasterization resolution.
- `VLM_NUM_CTX_PASS1 / PASS2 = 8192` — context window size; deliberately *not* larger, since a
  bigger context window pre-allocates VRAM regardless of tokens actually used.
- `EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"`, `EMBEDDING_DIM = 384`.
- `SEARCH_RADIUS_KM = 10.0`, `DEPTH_WINDOW_M = 50.0` — the Risk Monitor's "nearby" definition.

## Known limitations

Stated plainly, not glossed over:

- **Hardware ceiling, not eliminated.** The quantized 3.8B model already occupies most of a 4GB
  GPU's VRAM. An unusually dense page can still fail extraction outright; the pipeline detects this,
  isolates it to the one affected page, and surfaces a clear failure reason for human review — it
  does not silently produce wrong data, but it also does not claim a zero failure rate.
- **Confidence reflects legibility, not independent factual verification.** A HIGH-confidence field
  means the model is confident it read the scan correctly — not that the underlying number has been
  checked against any external source.
- **Heuristic table/figure detection has known edge cases.** `table_parser.py`'s pipe-delimited-text
  parser is a heuristic over the model's own inconsistent formatting, not a formally guaranteed
  parse — several real failure patterns (chart axes mistaken for tables, wrapped field-name lists
  mistaken for tables) have been found and fixed, but it is not proven exhaustive.
- **Ingestion is a batch job, not instant.** Roughly 20–90+ seconds per page depending on content
  density. Querying an already-ingested corpus (DrillMind, Risk Monitor, Map) is fast — well under a
  second — since those are database/vector lookups, not fresh model calls.

## Research references

| Reference | Relevance |
|---|---|
| Jeong et al. (Schlumberger), US Patent 11,143,775 B2, 2021 — *Automated Offset Well Analysis* | Establishes offset-well risk analysis as a formal, industry-recognized methodology; this project implements the retrieval/correlation half of that concept. |
| Damarla & Zhu, arXiv:2511.06607, 2025 — *Explainable Probabilistic ML for Drilling Fluid Loss of Circulation* | Direct precedent for lost-circulation prediction with an explainability layer, mirroring this project's confidence-labelling approach. |
| Azadivash, *Heliyon*, 2024 (e41059) — *Lost circulation intensity characterization using ML and well-log data* | Peer-reviewed precedent for severity-graded (not just binary) hazard classification. |
| Achyut Mani Tripathi, IIT Guwahati, 2021 (PhD Thesis) — *Anomaly Detection in Oil Well Drilling Operations Using AI* | India-specific, field-validated (Assam) precedent for AI-based drilling anomaly detection. |
| Qwen Team — Qwen2.5-VL | The vision-language model this project runs locally via Ollama. |
| IADC Lexicon — *Well Completion Report* | Industry-standard definition of a WCR's expected fields, grounding this project's extraction schema. |
