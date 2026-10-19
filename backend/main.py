"""
HTTP API for the earthwork pipeline.

The app's endpoints (api/routes_*.py):

    GET  /api/health                         dependency check
    GET  /api/accuracy                       measured accuracy per plan-set format
    POST /api/jobs                           upload a plan-set PDF -> {job_id}
    POST /api/uploads                        start a chunked upload -> {upload_id, chunk_size}
    PUT  /api/uploads/{id}/{n}               chunk n (raw bytes, in order)
    POST /api/uploads/{id}/complete          assemble the PDF and start the job -> {job_id}
    GET  /api/jobs/{id}                      status, progress, and (when done) the result
    PUT  /api/jobs/{id}/edits                save hand-edited cut/fill areas; volumes are recomputed
    GET  /api/jobs/{id}/sections/{n}.png     audit overlay for one cross-section
    GET  /api/jobs/{id}/sections/{n}/profile.json   its traced lines, in image pixels
    GET  /api/jobs/{id}/pages/{n}.png        a whole sheet of the original PDF, as an image
    GET  /api/jobs/{id}/pages/{n}.json       the sections on that sheet, in its coordinates
    GET  /api/jobs/{id}/calibration          feet-to-pixel calibration per sheet + cut / fill shapes
    GET  /api/jobs/{id}/cross-sections       every section in the received cross-section format:
                                             left / right areas, shapes, confidence and alerts
    GET  /api/jobs/{id}/volumes.csv          volume table (respects ?exclude=road ids)
    GET  /api/jobs/{id}/sections.csv         per-section areas
    GET  /api/jobs/{id}/annotated.pdf        the original PDF with the takeoff drawn on it
    POST /api/jobs/{id}/compare              compare with provided results (xlsx / csv upload)
    GET  /api/comparisons                    stored comparisons with provided results (e.g. SR 5)
    GET  /api/comparisons/{name}             one of them
    GET  /api/schemas/{name}                 the JSON Schemas responses are validated against

The public API for clients, version 1 (api/public.py; Swagger UI at /api/v1/docs):

    POST /api/v1/calibration                 upload a plan-set PDF -> 202 {id, ...} or 200 + calibration
    GET  /api/v1/calibration/{id}            200 the calibration, 202 while processing
    POST /api/v1/cross-sections              upload a plan-set PDF -> 202 {id, ...} or 200 + cross-sections
    GET  /api/v1/cross-sections/{id}         200 the cross-sections (?surface=), 202 while processing
    GET  /api/v1/schemas/{name}              the JSON Schemas of the two responses

Both share one job store and queue (api/jobs.py): jobs run
MAX_CONCURRENT_JOBS at a time, live in memory and expire after
JOB_TTL_SECONDS. This module only assembles the application, serves the
built frontend, and keeps the names tests and tools import from main.
"""
from __future__ import annotations

import os
import sys
import types

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api import (
    public,
    routes_comparisons,
    routes_exports,
    routes_jobs,
    routes_results,
    routes_sheets,
    routes_status,
    settings,
)

# imported from main by tests and tools
from api.jobs import JOBS, Job
from api.payload import recompute_volumes as _recompute_volumes
from api.payload import section_sides, serialize
from gdot_earthwork import __version__

__all__ = ["app", "JOBS", "Job", "_recompute_volumes", "section_sides", "serialize"]

app = FastAPI(title="Earthwork Takeoff API", version=__version__)
OPEN_PATHS = {"/api/health"}


# With API_KEYS set, the app's own endpoints need a key too, not just the
# public API (which checks X-API-Key itself): otherwise anyone could upload and
# keep the CPUs busy. Images and downloads are plain links that cannot carry a
# header, so ?api_key= is accepted as well. The health check stays open for
# the host. Registered before CORS, so CORS stays outermost: preflights are
# answered and a 401 still carries the CORS headers the browser needs to read it.
@app.middleware("http")
async def require_key(request: Request, call_next):
    path = request.url.path
    if (path.startswith("/api/") and not path.startswith("/api/v1/") and path not in OPEN_PATHS
            and not public.key_ok(request.headers.get("x-api-key") or request.query_params.get("api_key"))):
        return JSONResponse({"detail": "Missing or invalid API key: send one in the X-API-Key header."},
                            status_code=401, headers={"WWW-Authenticate": "APIKey"})
    return await call_next(request)


# also covers the public API mounted below: the same origins may call it, and
# read the headers it answers with (the job id, where to poll, when)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
    expose_headers=["X-Job-Id", "Location", "Content-Location", "Retry-After"],
)
for routes in (routes_status, routes_jobs, routes_sheets, routes_results, routes_exports, routes_comparisons):
    app.include_router(routes.router)
app.mount("/api/v1", public.app)

# Production: serve the built frontend from the same origin, so the app's
# relative /api URLs just work. (In development Vite serves it instead.)
# Registered last: its catch-all route must not shadow the API.
if os.path.isdir(settings.FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(settings.FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = os.path.normpath(os.path.join(settings.FRONTEND_DIST, path))
        if path and candidate.startswith(os.path.normpath(settings.FRONTEND_DIST)) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(settings.FRONTEND_DIST, "index.html"))


class _Main(types.ModuleType):
    """main.<SETTING> reads and writes api.settings, where the endpoints look:
    tests patch main.MAX_UPLOAD_MB and main.ACCURACY_FILE as before the split."""


for _name in ("ACCURACY_FILE", "MAX_UPLOAD_MB", "JOB_TTL_SECONDS", "UPLOAD_CHUNK_MB", "API_KEYS"):
    setattr(_Main, _name, property(lambda self, n=_name: getattr(settings, n),
                                   lambda self, value, n=_name: setattr(settings, n, value)))
sys.modules[__name__].__class__ = _Main
