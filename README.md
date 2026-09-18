## What this is

eRTMAC-NWIS is a local, offline well-intelligence platform for ingesting Well Completion Reports (WCRs) and Daily Drilling Reports (DDRs), extracting structured data with a local vision-language model, visualising wells, monitoring nearby hazards, and querying indexed documents through Drill Mind. Contributions can target the FastAPI/Python processing pipeline, React interface, offline map, OCR review workflow, or retrieval and risk logic.

### Stack
- **Language(s):** Python, JavaScript/JSX, CSS
- **Framework / runtime:** FastAPI + Uvicorn backend; React 19 + Vite frontend
- **Notable libraries:** Ollama/Qwen2.5-VL, pypdfium2, Qdrant, Sentence Transformers, Leaflet/react-leaflet, Recharts

## How it's organized

```text
backend/
  main.py                    FastAPI entry point and router registration
  routers/                   API endpoints for uploads, corpus, dashboard,
                             map data, risk monitoring, Drill Mind, and OCR review
  src/
    unified_parser.py        Multi-pass PDF/VLM extraction pipeline
    pdf_processor.py         PDF-to-PNG rasterisation
    table_parser.py          Table detection and HTML rendering
    figure_extractor.py      Embedded figure/chart extraction
    database.py              SQLite persistence and domain queries
    vector_store.py          CPU-based embeddings and local Qdrant indexing
    rag_engine.py            Grounded Drill Mind retrieval and responses
    hazard_monitor.py        Spatial/depth hazard correlation
    utils.py                 Ollama calls and shared utilities
  run_ingestion.py           Ingestion helpers
  review_document_pages.py   OCR/page-review tooling
  test_*.py                  VLM and setup tests

ertmac-nwis/
  src/
    pages/                   Dashboard, upload, map, risk, corpus, and Drill Mind views
    components/              Shared UI such as Sidebar
    api/client.js            Central Axios backend client
    data/indiaPlaces.js      Offline map labels and place data
    assets/                  Local basemap and other frontend assets
  package.json               Vite scripts and frontend dependencies

start.sh                     Starts backend and frontend on Windows
stop.sh                      Stops services on ports 8000 and 5173
```

**How it fits together:** The React application routes users to Dashboard, Upload, Map Visualise, Risk Monitor, Corpus, and Drill Mind pages through `src/App.jsx`. Uploads go to `POST /api/upload`; the backend rasterises PDFs with `pdf_processor.py`, registers them through `unified_parser.py`, and runs VLM extraction in the background. Extracted page text is embedded by `vector_store.py` into local Qdrant storage, then `rag_engine.py` retrieves grounded context for `POST /api/chat`. Risk monitoring combines verified well coordinates with hazard depths in `hazard_monitor.py`.

### Good contribution areas

- **Frontend improvements:** work in `ertmac-nwis/src/pages/` and `src/components/`. For example, improve upload progress/status handling in `Upload.jsx`, dashboard visualisations, accessibility, responsive layouts, or Drill Mind citation presentation.
- **Backend/API features:** add or improve routers under `backend/routers/`, then register new routers in `backend/main.py`.
- **Extraction quality:** improve `backend/src/unified_parser.py`, `table_parser.py`, `figure_extractor.py`, or `pdf_processor.py`. Be careful to preserve the existing distinction between raw extraction and human-corrected text.
- **Search quality:** improve `vector_store.py` and `rag_engine.py`. Drill Mind is intentionally grounded only in retrieved corpus content; do not add a fallback to general model knowledge.
- **Risk and map functionality:** extend `hazard_monitor.py`, `routers/risk.py`, `routers/wells_map.py`, or `MapVisualise.jsx`. Coordinate verification and confidence filtering are important safety behaviors.
- **OCR review:** improve the standalone review interface in `backend/routers/review.py` and the table round-trip behavior between HTML and pipe-delimited text.
- **Testing and documentation:** expand the existing VLM tests, add tests for parser edge cases, document Ollama/model setup, and improve the sparse setup instructions.

## How to run it

The repository provides separate backend and frontend manifests. From a fresh clone:

```bash
git clone https://github.com/pallavi-a11y/eRTMAC-NWIS.git
cd eRTMAC-NWIS
```

Set up the backend:

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

The backend requires a running local Ollama installation with the configured model:

```bash
ollama list
ollama pull qwen2.5vl:3b
```

Start the FastAPI backend:

```bash
cd backend
venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal, install and run the frontend:

```bash
cd ertmac-nwis
npm install
npm run dev
```

Open `http://localhost:5173`. 






