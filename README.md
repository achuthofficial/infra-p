# Earthwork Takeoff

Upload a road plan set (PDF) and get the cut and fill of every cross-section,
the volumes per road, and the feet-to-pixel calibration of every sheet.

## What to upload

* **One PDF** with the cross-section sheets: CAD exports or scans both work.
  The whole plan set is fine; pages without cross-sections are skipped.
* Staged sets: keep the stage in the file name or title block (e.g.
  `ROAD - STAGE 2.pdf`); only that stage's work is measured.
* Large files (100 MB+) are fine: the app uploads in chunks (limit
  `MAX_UPLOAD_MB`, default 2048).

## Run it

Backend (Python 3.11+):

```bash
sudo apt-get install tesseract-ocr     # OCR for scanned sheets
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Frontend (Node 18+), in a second terminal:

```bash
cd frontend
npm install
npm run dev                            # open http://localhost:5173
```

Or with Docker: `docker build -t earthwork . && docker run -p 8000:8000 earthwork`,
then open http://localhost:8000.

## Deploy on Railway

The repo deploys as one service (app + API in one container) from `Dockerfile`;
`railway.json` sets the health check (`/api/health`) and restart policy.

1. Railway → **New Project** → **Deploy from GitHub repo** → pick this repo.
2. **Settings → Networking → Generate Domain**. That's it: open the domain.

Optional variables (**Variables** tab):

| Variable | Default | |
|---|---|---|
| `API_KEYS` | unset (open) | comma-separated keys required as `X-API-Key` on `/api/v1` |
| `WORKERS` | min(4, CPUs) | processes per job; use 1–2 on instances with under 4 GB RAM |
| `MAX_UPLOAD_MB` | 2048 | largest PDF accepted |
| `JOB_TTL_SECONDS` | 21600 | how long results stay available |

Jobs are kept in memory and temp files, so a redeploy clears them; plan sets
are simply uploaded again.

## Deploy on Cloud Run (API) + Vercel (app)

The API runs on Cloud Run at `https://infra-p-172025772606.us-central1.run.app`;
the app on Vercel calls it directly (`frontend/.env.production` sets
`VITE_API_BASE`, and the API allows every `*.vercel.app` origin by default,
previews included; change it with `ALLOWED_ORIGIN_REGEX`).

**Vercel**: import the repo, set **Root Directory** to `frontend` (framework
Vite, settings come from `frontend/vercel.json`). To point at another backend,
set `VITE_API_BASE` in the project's environment variables and redeploy.

**Cloud Run**: jobs live in the memory of the instance that received the
upload and keep processing after the upload request returns, so for several
users at once the service needs one instance that is always on, with CPU
always allocated:

```bash
gcloud run deploy infra-p --source . --region us-central1 --allow-unauthenticated \
  --min-instances 1 --max-instances 1 --no-cpu-throttling \
  --cpu 4 --memory 8Gi --timeout 3600 --concurrency 80 \
  --set-env-vars MAX_CONCURRENT_JOBS=2,WORKERS=2
```

* `--max-instances 1`: a second instance would not know the first one's jobs
  (uploads and polls would randomly 404).
* `--min-instances 1` and `--no-cpu-throttling`: jobs aren't wiped by scaling
  to zero, and keep running between the browser's polls.
* `MAX_CONCURRENT_JOBS` jobs run at once, each with `WORKERS` processes; the
  rest wait as *queued*. Keep `MAX_CONCURRENT_JOBS × WORKERS` near the CPU count
  and about 1 GB of memory per worker.

## Use it

* **New takeoff**: upload the PDF. **Results** then shows every section (traced
  lines over the drawing), cut / fill per side and the volumes. Sections below
  85 % confidence are flagged for review and can be edited.
* **Calibration**: upload the PDF to get each sheet's scale as JSON.
* **API**: try the two API endpoints in the browser and copy the code to call them.

## API

| Endpoint | Returns |
|---|---|
| `POST /api/v1/calibration` (form field `file` = the PDF) | job id (`202`), or the result when ready within `?wait=` seconds |
| `GET /api/v1/calibration/{id}` | feet-to-pixel calibration of every sheet |
| `POST /api/v1/cross-sections` (form field `file`) | job id (`202`), or the result when ready within `?wait=` seconds |
| `GET /api/v1/cross-sections/{id}` | every cross-section: station, left / right cut and fill, confidence |

`202` means still processing: poll the `GET` until it returns `200`. Full docs
at `/api/v1/docs`. Set `API_KEYS=key1,key2` on the server to require an
`X-API-Key` header.

```bash
curl -F "file=@plans.pdf" "http://localhost:8000/api/v1/cross-sections?wait=300"
```

## Tests

```bash
cd backend && python -m pytest -q
ruff check .                           # lint (pip install ruff)
```
