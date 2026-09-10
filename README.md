# PS26227 Earth Observation Change Intelligence

PS26227 is a cross-modal Earth Observation system for searching Sentinel-2 L2A imagery and detecting persistent land-use change. It combines natural-language and image retrieval with multi-temporal registration, change detection, water analysis, false-alarm suppression, and analyst review.

## Architecture

```text
Sentinel-2 L2A GeoTIFF/COG
        |
        v
Raster validation and uint16 reflectance normalization
        |
        v
512x512 tiling at 10 m GSD
        |
        +--> CLIP-RSICD image/text encoder -> 512-d semantic_tiles Qdrant collection
        |
        +--> Clay MAE multispectral encoder -> 768-d spectral_tiles Qdrant collection
        |
        +--> PostGIS metadata catalog, or SQLite fallback for local development
        |
        v
Cloud masking -> AROSICS registration -> Open-CD -> NDWI -> temporal analysis
        |
        v
CCS reranking -> FastAPI API -> React + MapLibre GL analyst dashboard
```

### Core services

- **Ingestion:** `backend/ingestion/tiler.py` loads Sentinel-2 L2A `uint16` bands, normalizes reflectance, creates stable `location_key` grid cells, writes RGB evidence tiles, and indexes both embedding spaces.
- **Vector search:** Qdrant stores independent 512-dimensional CLIP-RSICD semantic vectors and 768-dimensional Clay MAE spectral vectors using cosine similarity.
- **Metadata:** PostgreSQL/PostGIS is the production catalog. SQLite is the automatic local fallback and stores tile metadata, change results, and review audit records.
- **Change analysis:** s2cloudless quality masking, AROSICS sub-pixel co-registration, Open-CD SNUNet/TinyCD inference, NDWI water extent analysis, and earliest persistent-change estimation.
- **API:** FastAPI exposes search, ingestion, results, temporal evidence, health, and analyst review endpoints.
- **UI:** React, Vite, and MapLibre GL provide map discovery, confidence markers, split-screen evidence comparison, CCS visualization, and review logging.

## False-Alarm Suppression

The Change Confidence Score is a multi-factor engineering score, not a calibrated probability:

$$
CCS = 0.40E + 0.20(1-C) + 0.20R + 0.20T
$$

Where:

- $E$ is Open-CD change evidence.
- $C$ is cloud contamination, so clear-sky quality is $1-C$.
- $R$ is co-registration quality.
- $T$ is temporal consistency across observations.

Classification thresholds:

- `CCS >= 0.70`: high-confidence change, shown in green.
- `0.40 <= CCS < 0.70`: analyst review required, shown in amber.
- `CCS < 0.40`: suppressed by default, shown in red or available through suppressed-result controls.

The final retrieval score combines semantic similarity and change confidence:

$$
final\_score = 0.50 \cdot semantic\_similarity + 0.50 \cdot CCS
$$

## Watch-Folder Ingestion

`backend/ingestion/watch_folder.py` provides incremental scene monitoring for `data/incoming/`.

The service:

1. Handles native file creation events and atomic temporary-file-to-final-file move events.
2. Moves stabilization work to a background thread so the watchdog event loop is never blocked.
3. Waits for a non-empty file to become readable with stable size and modification time across repeated checks.
4. Processes only after the OS copy/write operation has settled.
5. Captures worker exceptions and failed ingestion results in logs and handler state.
6. Derives the scene center from GeoTIFF CRS and bounds when available.
7. Tiles, masks, embeds, and upserts only the affected `location_key` cells.
8. Re-runs temporal change analysis only for those affected cells; it does not rebuild the archive.

The integration test uses an isolated Qdrant directory through `QDRANT_STORAGE_DIR` and removes its temporary scene and vector-store artifacts after execution.

## Analyst UI

The frontend is in `frontend/` and includes:

- MapLibre GL confidence-colored markers and location selection.
- Natural-language, image, and multimodal search.
- Before/after split comparison slider with change-mask evidence.
- CCS component visualizer for change evidence, cloud quality, registration, and temporal consistency.
- Similar-site discovery from stored semantic vectors.
- Confirm/reject analyst workflow with notes and persisted audit logging.
- Evidence export and temporal history views.

## Quickstart

### Requirements

- Python 3.11+
- Node.js and npm
- A project virtual environment at `.venv`
- Optional: PostgreSQL/PostGIS and a remote Qdrant service

Install Python dependencies in the project environment:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

### Start the backend

From the repository root:

```powershell
.venv\Scripts\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

- API documentation: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Start the frontend

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Vite uses port 3000 by default and selects the next available port when 3000 is occupied.

### Start frontend and backend with the command loop

From the repository root, run:

```powershell
.\start.ps1
```

At the `start>` prompt, use:

- `backend` to start FastAPI.
- `frontend` to start Vite.
- `both` to start both services.
- `status` to show running service processes.
- `stop` to stop both services and keep the prompt open.
- `exit` to stop any services started by the script and close the loop.

PowerShell may require this one-time session command if script execution is restricted:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Start the watch-folder service

The API can start the background watcher through the application lifecycle. For a standalone watcher:

```powershell
.venv\Scripts\python -m backend.ingestion.watch_folder
```

Copy `.tif`, `.tiff`, `.cog`, or `.geotiff` scenes into `data/incoming/` after the watcher is running.

## Model Weights

Run the staging command when network access and approved model sources are available:

```powershell
.venv\Scripts\python scripts/download_weights.py
```

Expected locations for full-accuracy artifacts:

- `data/weights/clip_rsicd/` for CLIP-RSICD Hugging Face files.
- `data/weights/open_cd/open_cd_snunet.pt` for the Open-CD checkpoint.
- `data/weights/clay_v1_5.pt` for Clay MAE weights.

Configure downloads with `CLIP_HF_REPO`, `OPEN_CD_WEIGHTS_URL`, `OPEN_CD_HF_REPO`, and `OPEN_CD_HF_FILE`. Set `OPEN_CD_SHA256` to verify checkpoint integrity.

Verify a disconnected environment without network access:

```powershell
.venv\Scripts\python scripts/download_weights.py --offline
```

The current repository contains zero-byte placeholders rather than the CLIP-RSICD and Open-CD pretrained artifacts. Offline verification therefore succeeds with explicit messages that the runtime will use its deterministic CLIP encoder and Otsu Open-CD fallback. These fallbacks are suitable for local pipeline validation, not full pretrained-model accuracy.

## Testing

Run the complete Python suite:

```powershell
.venv\Scripts\python -m pytest -q
```

The suite covers API health, search, review, CCS, temporal logic, registration, edge cases, and incremental ingestion. The incremental test creates a 512x512 GeoTIFF, sends it through the watch-folder handler, verifies Qdrant and SQLite/PostGIS-compatible persistence, confirms that only the expected grid cell changes, and cleans up.

Run the integration test alone with isolated Qdrant storage:

```powershell
$env:QDRANT_STORAGE_DIR = "$PWD\data\qdrant_ingest_test"
$env:QDRANT_LOCAL_PATH = $env:QDRANT_STORAGE_DIR
.venv\Scripts\python -m pytest tests/test_incremental_ingestion.py -q
```

Run the offline model check:

```powershell
.venv\Scripts\python scripts/download_weights.py --offline
```

Run the Playwright frontend test:

```powershell
cd frontend
npx playwright install
npx playwright test tests/ps26227.spec.js --project=chromium
```

The Playwright flow searches for newly built structures near a river, checks confidence markers, drags the comparison slider, and confirms an analyst review.

## API Surface

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service, model, and Qdrant health |
| `POST` | `/search/text` | Natural-language semantic search |
| `POST` | `/search/image` | Image similarity search |
| `POST` | `/search/multimodal` | Text and image blended search |
| `POST` | `/ingest` | Synchronous scene ingestion |
| `GET` | `/results/{change_id}` | Change result and review history |
| `GET` | `/results/{location_key}/temporal` | Location timeline and earliest change |
| `GET` | `/results/{change_id}/evidence` | Evidence export |
| `GET` | `/results/{location_key}/similar` | Similar-site discovery |
| `POST` | `/review/{change_id}` | Confirm or reject a result with notes |
| `GET` | `/review/{change_id}` | Review audit history |

## Configuration

Important environment variables include:

- `USE_LOCAL_SQLITE_FALLBACK=true`
- `USE_LOCAL_QDRANT_STORAGE=true`
- `QDRANT_STORAGE_DIR` and `QDRANT_LOCAL_PATH`
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `CLIP_RSICD_MODEL`, `CLAY_WEIGHTS_PATH`, `OPEN_CD_CHECKPOINT`
- `TILE_SIZE_PX`, `TILE_GSD_METERS`, and `CLOUD_PROB_THRESHOLD`

Keep local databases, Qdrant storage, raw imagery, generated tiles, and model artifacts out of production source control unless they are intentionally published release assets.

## Docker

To start the containerized stack:

## Free demo deployment

The default `render.yaml` is a free Render Blueprint for the FastAPI backend. It uses temporary local SQLite and Qdrant storage, so data can be lost when the free service restarts or redeploys.

Deploy the backend from Render with **New + > Blueprint**, select this repository, choose branch `main`, and use `render.yaml`. After deployment, copy the backend URL, for example `https://ps26227-backend-demo.onrender.com`.

Deploy the frontend separately on Netlify:

1. Choose **Add new site > Import an existing project** and select `pothulaannapurna8/ORBI`.
2. Netlify reads `netlify.toml`; confirm the publish directory is `frontend/dist`.
3. Add environment variable `VITE_API_URL` with the Render backend URL.
4. Deploy the site and open its Netlify URL.

This free split setup is for demos and testing. The Render backend may sleep when idle, and local database/vector data is not persistent.

```powershell
cd docker
docker compose up -d --build
docker compose ps
```

## Repository Layout

```text
backend/                 FastAPI services and geospatial processing
backend/ingestion/       Tiling and incremental watch-folder ingestion
backend/embeddings/      CLIP-RSICD and Clay encoders
backend/change_detection/Temporal change, NDWI, CCS, and Open-CD logic
backend/database/        SQLite/PostGIS-compatible metadata and Qdrant access
frontend/                React + Vite + MapLibre analyst dashboard
scripts/                 Dataset staging, indexing, and weight management
tests/                   API and incremental-ingestion integration tests
evaluation/              Retrieval and change-detection evaluation scripts
configs/                 Model and weight configuration
```

## Operational Notes

- Missing pretrained model files are handled explicitly by deterministic fallbacks; install approved weights before making accuracy claims.
- The local Qdrant client locks its storage directory. Use `QDRANT_STORAGE_DIR` for parallel tests or separate application processes.
- The SQLite fallback is for development and tests. Use PostgreSQL/PostGIS for concurrent production workloads.
- Review decisions are audit records and should be retained according to the deployment’s data-governance policy.
