"""
The public API, version 1: a plan set's calibration and its cross-sections
as JSON, for clients outside the app.

    POST /api/v1/calibration             upload a plan-set PDF (form field "file")
                                         -> 202 {id, status, progress, links}, or 200 + the
                                         calibration if it is ready within ?wait= seconds
    GET  /api/v1/calibration/{id}        200 the calibration; 202 + progress until it is ready
    POST /api/v1/cross-sections          the same, answered with the cross-section JSON
    GET  /api/v1/cross-sections/{id}     ?surface=subgrade|finished
    GET  /api/v1/schemas/{name}          the JSON Schemas of the two responses
    GET  /api/v1/docs, /api/v1/openapi.json

It is a FastAPI application of its own, mounted by main, so its OpenAPI
document and Swagger UI hold only what clients may rely on. The jobs behind
it are the app's (api.jobs, api.results): one processing run serves both
endpoints, and a job uploaded in the app can be read here by id, and vice
versa. Errors are {"detail": "..."}; 422 means processing failed (FastAPI's
own parameter errors are answered 400 here so the two can't be confused).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from gdot_earthwork import __version__

from . import jobs, results, schemas, settings
from .streaming import receive_pdf

# each deliverable (its URL segment and schema file) -> its OpenAPI component, and who answers with it
COMPONENTS = {
    "calibration": ("Calibration", "POST /calibration and GET /calibration/{id}"),
    "cross-sections": ("CrossSections", "POST /cross-sections and GET /cross-sections/{id}"),
}
MAX_WAIT = 600              # seconds a request may hold on for its result
POLL_SECONDS = 0.25         # how often a waiting request looks at its job
RETRY_AFTER = 2             # seconds suggested to clients between polls


# ---------------------------------------------------------------- response models
class Progress(BaseModel):
    done: int = Field(description="Pages processed so far.")
    total: int = Field(description="Pages in the plan set; 0 until processing starts.")


class Links(BaseModel):
    self_: str = Field(alias="self", description="Where to poll for the result this request asked for.")
    calibration: str = Field(description="The job's calibration.")
    crossSections: str = Field(description="The job's cross-sections.")


class JobStatus(BaseModel):
    """A plan set still being processed."""
    id: str = Field(description="The job id: valid on both endpoints until the job expires.")
    status: Literal["queued", "running"] = Field(
        description="queued: waiting for a free processing slot; running: being read.")
    progress: Progress
    links: Links


class Problem(BaseModel):
    detail: str = Field(description="What went wrong, in words.")


# ---------------------------------------------------------------- authentication
api_key_header = APIKeyHeader(
    name="X-API-Key", scheme_name="APIKey", auto_error=False,
    description="One of the server's API keys. Needed only when the server is started with API_KEYS.")


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


def key_ok(key: str | None) -> bool:
    """True when API_KEYS is unset, or `key` is one of them. Digests are
    compared with hmac.compare_digest against every key, every time, so the
    timing tells nothing about a key's content, its length or which one matched."""
    if not settings.API_KEYS:
        return True
    given = _digest(key or "")
    return any([hmac.compare_digest(given, _digest(k)) for k in settings.API_KEYS])


def require_api_key(key: Annotated[str | None, Security(api_key_header)] = None) -> None:
    """With API_KEYS set, X-API-Key must be one of them."""
    if not key_ok(key):
        raise HTTPException(401, "Missing or invalid API key: send one in the X-API-Key header.",
                            headers={"WWW-Authenticate": "APIKey"})


# ---------------------------------------------------------------- the application
DESCRIPTION = f"""
Reads a road **cross-section plan set** (PDF, CAD-exported or scanned) and returns, as JSON:

* **Calibration**: the feet-to-pixel scale of every sheet, on each axis, and every cut / fill
  area found on it as a page-space shape.
* **Cross-sections**: every cross-section with its station, street and stage, left / right /
  total cut and fill (sq ft), its shapes on the page, a 0-100 confidence score and alerts
  below 85.

Lengths are in feet and areas in square feet. Pixels are PDF points of the page as
displayed, the same as a PDF viewer at 100 %.

### How it works
Reading a plan set takes from seconds to a few minutes (scanned sheets are read by OCR), so
it runs in the background:

1. **POST** the PDF as multipart form field `file` to `/calibration` or `/cross-sections`.
   The answer is `202` with the job `id` and links to poll, or `200` with the result if it
   is ready within `?wait=` seconds (0 to {MAX_WAIT}).
2. **GET** `/calibration/{{id}}` or `/cross-sections/{{id}}` until it answers `200`. Until
   then it answers `202` with the progress; `Retry-After` says when to ask again.

One processing run serves both endpoints: an id from either POST can be read from either GET.
Jobs are kept for {settings.JOB_TTL_SECONDS / 3600:g} hours.

| Status | Meaning |
|---|---|
| `200` | the result; its JSON Schema is at `/schemas/calibration` and `/schemas/cross-sections` |
| `202` | still queued or running: `{{id, status, progress, links}}` |
| `400` | not a PDF, an empty file, a malformed form, or a bad parameter |
| `401` | missing or wrong `X-API-Key` (only when the server has API keys) |
| `404` | unknown or expired id |
| `410` | the uploaded PDF is no longer on the server |
| `413` | the PDF is larger than the server's upload limit ({settings.MAX_UPLOAD_MB} MB) |
| `422` | the plan set could not be processed: `{{"detail": "..."}}` says why |

### Authentication
When the server is started with `API_KEYS` (comma separated), every request needs one of
them in the `X-API-Key` header; use **Authorize** to try the API from this page. Without
`API_KEYS` the API is open.

### Example
```bash
# upload and wait (up to 5 minutes) for the cross-sections
curl -X POST "https://HOST/api/v1/cross-sections?wait=300" \\
     -H "X-API-Key: $API_KEY" -F "file=@plans.pdf"

# or upload, then poll
curl -X POST https://HOST/api/v1/calibration -H "X-API-Key: $API_KEY" -F "file=@plans.pdf"
#   202 {{"id": "3f9c2a1b7e4d", "status": "queued", ...}}
curl https://HOST/api/v1/calibration/3f9c2a1b7e4d -H "X-API-Key: $API_KEY"
curl "https://HOST/api/v1/cross-sections/3f9c2a1b7e4d?surface=finished" -H "X-API-Key: $API_KEY"
```

Pipeline version {__version__}.
"""

TAGS = [
    {"name": "Calibration", "description": "Feet-to-pixel scale of every sheet, and the cut / fill areas as shapes."},
    {"name": "Cross-sections", "description": "Every cross-section's cut and fill, left / right of the centreline, "
                                              "with its shapes, confidence score and alerts."},
    {"name": "Schemas", "description": "The JSON Schemas (Draft 2020-12) the two responses follow."},
]

app = FastAPI(
    title="Earthwork Takeoff API",
    version="1",
    summary="Calibration and cross-sections of road plan sets, as JSON.",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    dependencies=[Depends(require_api_key)],
    responses={401: {"model": Problem, "description": "Missing or invalid API key (only when the server has API keys)."}},
    swagger_ui_parameters={"persistAuthorization": True, "displayRequestDuration": True},
)


@app.exception_handler(RequestValidationError)
async def _bad_parameter(request: Request, exc: RequestValidationError) -> JSONResponse:
    """400, not FastAPI's 422: here 422 means the plan set could not be processed."""
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or e['loc'][0]}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"detail": detail}, status_code=400)


def _openapi() -> dict:
    """FastAPI's document, plus the two response schemas from backend/schemas
    as components (what the 200 responses refer to). FastAPI's stock 422
    "Validation Error" is dropped: parameter errors are answered 400 here."""
    if app.openapi_schema is None:
        spec = FastAPI.openapi(app)              # builds and caches it
        for operations in spec["paths"].values():
            for op in operations.values():
                responses = op["responses"]
                if "HTTPValidationError" in str(responses.get("422")):
                    del responses["422"]
                op["responses"] = dict(sorted(responses.items()))
        components = spec.setdefault("components", {}).setdefault("schemas", {})
        for name in ("HTTPValidationError", "ValidationError"):
            components.pop(name, None)
        for name, (component, used_by) in COMPONENTS.items():
            components.update(schemas.openapi_components(name, component))
            components[component]["description"] = f"The response of {used_by} (JSON Schema `{name}`)."
    return app.openapi_schema


app.openapi = _openapi


# ---------------------------------------------------------------- answering
def _links(request: Request, job: jobs.Job, kind: str, surface: str | None) -> Links:
    base = request.scope.get("root_path", "").rstrip("/")       # /api/v1 where main mounts it
    links = {"calibration": f"{base}/calibration/{job.id}", "crossSections": f"{base}/cross-sections/{job.id}"}
    links["self"] = links["calibration"] if kind == "calibration" else \
        links["crossSections"] + (f"?surface={surface}" if surface else "")
    return Links.model_validate(links)


async def _answer(request: Request, job: jobs.Job, kind: str, wait: float, surface: str | None = None) -> JSONResponse:
    """The result once the job is done (waiting up to `wait` seconds for it),
    or where it stands (202). The wait sleeps on the event loop, so waiting
    clients hold no worker thread, and ends early if the client hangs up
    (the job carries on). Every answer names the job in X-Job-Id: a POST
    answered 200 has no other place to say it."""
    deadline = time.monotonic() + wait
    while job.status in ("queued", "running") and time.monotonic() < deadline:
        await asyncio.sleep(max(0.0, min(POLL_SECONDS, deadline - time.monotonic())))
        if await request.is_disconnected():
            break
    links = _links(request, job, kind, surface)
    state = job.status              # read once: the job's thread may move it on meanwhile
    if state == "error":
        raise HTTPException(422, job.error, headers={"X-Job-Id": job.id})
    if state != "done":
        status = JobStatus(id=job.id, status=state, progress=Progress(done=job.done, total=job.total),
                           links=links)
        return JSONResponse(status.model_dump(by_alias=True), status_code=202, headers={
            "Location": links.self_, "Retry-After": str(RETRY_AFTER), "X-Job-Id": job.id})
    # the calibration is built once per job (OCR: seconds); off the event loop
    if kind == "calibration":
        body = await run_in_threadpool(results.calibration_for, job)
    else:
        body = await run_in_threadpool(results.cross_sections_for, job, surface)
    return JSONResponse(body, headers={"Content-Location": links.self_, "X-Job-Id": job.id})


async def _create(request: Request, kind: str, wait: float) -> JSONResponse:
    jobs.expire_jobs()
    name, path = await receive_pdf(request, "file")
    return await _answer(request, jobs.start_job(name, path), kind, wait)


# ---------------------------------------------------------------- documentation of the endpoints
Wait = Annotated[float, Query(
    ge=0, le=MAX_WAIT,
    description=f"Seconds (0-{MAX_WAIT}) to hold the request open for the result before answering 202. "
                "0 answers at once.")]
JobId = Annotated[str, Path(description="The id returned by either POST endpoint.", examples=["3f9c2a1b7e4d"])]
Surface = Annotated[Literal["subgrade", "finished"] | None, Query(
    description="Measure cut and fill to the subgrade (bottom of the pavement, the default) "
                "or to the finished grade (top of the pavement).")]

UPLOAD_BODY = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"],
    "properties": {"file": {
        "type": "string", "format": "binary", "contentMediaType": "application/pdf",
        "description": f"The plan-set PDF, up to {settings.MAX_UPLOAD_MB} MB. "
                       "It is streamed to disk as it arrives, so large scans are fine."}},
}}}}}
JOB_ID_HEADER = {"X-Job-Id": {"description": "The job id, to read the other deliverable or poll again.",
                                "schema": {"type": "string"}}}
STATUS_202 = {"model": JobStatus, "description": "Still queued or running. Poll `links.self` (also in `Location`).",
              "headers": {"Location": {"description": "Where to poll.", "schema": {"type": "string"}},
                          "Retry-After": {"description": "Seconds to wait before polling again.",
                                          "schema": {"type": "integer"}},
                          **JOB_ID_HEADER}}


def _result(kind: str, text: str) -> dict:
    return {"description": f"{text} Follows the JSON Schema `{kind}` (`GET /schemas/{kind}`).",
            "headers": {"Content-Location": {"description": "Where this result can be read again.",
                                             "schema": {"type": "string"}}, **JOB_ID_HEADER},
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{COMPONENTS[kind][0]}"}}}}


ERRORS = {400: "Not a PDF, an empty file, a malformed form, or a bad parameter.",
          404: "Unknown or expired id.",
          410: "The uploaded PDF is no longer on the server.",
          413: "The PDF is larger than the server's upload limit (MAX_UPLOAD_MB).",
          422: "The plan set could not be processed; `detail` says why."}


def _errors(*codes: int) -> dict:
    return {c: {"model": Problem, "description": ERRORS[c]} for c in codes}


CALIBRATION_200 = _result("calibration", "The calibration of every sheet and the cut / fill shapes.")
CROSS_SECTIONS_200 = _result("cross-sections", "Every cross-section, with its shapes, confidence and alerts.")


# ---------------------------------------------------------------- endpoints
@app.post("/calibration", tags=["Calibration"], operation_id="createCalibration", status_code=202,
          summary="Upload a plan set for calibration", response_description="Still queued or running.",
          responses={200: CALIBRATION_200, 202: STATUS_202, **_errors(400, 413, 422)}, openapi_extra=UPLOAD_BODY)
async def create_calibration(request: Request, wait: Wait = 0):
    """Upload a plan-set PDF and get the **feet-to-pixel calibration** of every sheet: pixels
    per foot on each axis (from the sheet's cross-section grid, or its printed scale note),
    and every cut / fill area found on it as a page-space shape.

    Answers `202` with the job `id` at once, or `200` with the calibration if it is ready
    within `wait` seconds. The same `id` also reads the cross-sections.

    ```bash
    curl -X POST "https://HOST/api/v1/calibration?wait=120" -H "X-API-Key: $API_KEY" -F "file=@plans.pdf"
    ```
    """
    return await _create(request, "calibration", wait)


@app.get("/calibration/{job_id}", tags=["Calibration"], operation_id="getCalibration",
         summary="Get a plan set's calibration", response_description=CALIBRATION_200["description"],
         responses={200: CALIBRATION_200, 202: STATUS_202, **_errors(400, 404, 410, 422)})
async def get_calibration(request: Request, job_id: JobId, wait: Wait = 0):
    """The calibration of the plan set uploaded as `job_id` (by either POST endpoint, or the
    app): `200` with the JSON once processing is done, `202` with its progress until then.

    ```bash
    curl https://HOST/api/v1/calibration/3f9c2a1b7e4d -H "X-API-Key: $API_KEY"
    ```
    """
    return await _answer(request, jobs.get_job(job_id), "calibration", wait)


@app.post("/cross-sections", tags=["Cross-sections"], operation_id="createCrossSections", status_code=202,
          summary="Upload a plan set for its cross-sections", response_description="Still queued or running.",
          responses={200: CROSS_SECTIONS_200, 202: STATUS_202, **_errors(400, 413, 422)}, openapi_extra=UPLOAD_BODY)
async def create_cross_sections(request: Request, wait: Wait = 0):
    """Upload a plan-set PDF and get **every cross-section**: station, street and stage, cut
    and fill left and right of the centreline and in total (sq ft, to the subgrade), its
    shapes on the page, a 0-100 confidence score and alerts below 85.

    Answers `202` with the job `id` at once, or `200` with the cross-sections if they are
    ready within `wait` seconds. The same `id` also reads the calibration.

    ```bash
    curl -X POST "https://HOST/api/v1/cross-sections?wait=300" -H "X-API-Key: $API_KEY" -F "file=@plans.pdf"
    ```
    """
    return await _create(request, "cross-sections", wait)


@app.get("/cross-sections/{job_id}", tags=["Cross-sections"], operation_id="getCrossSections",
         summary="Get a plan set's cross-sections", response_description=CROSS_SECTIONS_200["description"],
         responses={200: CROSS_SECTIONS_200, 202: STATUS_202, **_errors(400, 404, 410, 422)})
async def get_cross_sections(request: Request, job_id: JobId, surface: Surface = None, wait: Wait = 0):
    """The cross-sections of the plan set uploaded as `job_id` (by either POST endpoint, or the
    app), measured to `surface`: `200` with the JSON once processing is done, `202` with its
    progress until then. Areas edited by hand in the app are applied (`edited: true`; the
    traced values stay in `measuredAreas`).

    ```bash
    curl "https://HOST/api/v1/cross-sections/3f9c2a1b7e4d?surface=finished" -H "X-API-Key: $API_KEY"
    ```
    """
    return await _answer(request, jobs.get_job(job_id), "cross-sections", wait, surface)


@app.get("/schemas/{name}", tags=["Schemas"], operation_id="getSchema", response_class=FileResponse,
         summary="Get a response's JSON Schema",
         responses={200: {"description": "The JSON Schema (Draft 2020-12).",
                          "content": {"application/schema+json": {"schema": {"type": "object"}}}},
                    404: {"model": Problem, "description": "No schema by that name."}})
def get_schema(name: Annotated[str, Path(description="`calibration` or `cross-sections`.",
                                         examples=["calibration"])]):
    """The JSON Schema the calibration or the cross-section response follows."""
    return schemas.response(name)
