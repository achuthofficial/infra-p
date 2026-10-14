"""Starting jobs (one request, or a chunked upload), reading them, and saving hand edits."""
from __future__ import annotations

import os
import tempfile
import time
import uuid

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile

from . import jobs, settings
from .payload import apply_edits, parse_edits

router = APIRouter()


@router.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    jobs.expire_jobs()
    name = os.path.basename(file.filename or "plan-set.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF plan set.")
    with jobs.UploadWriter() as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"job_id": jobs.start_job(name, out.path).id}


@router.post("/api/uploads")
def start_upload(body: dict = Body(...)):
    jobs.expire_uploads()
    name = os.path.basename(str(body.get("filename") or "plan-set.pdf"))
    size = body.get("size")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF plan set.")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise HTTPException(400, "size must be the file size in bytes.")
    if size > settings.MAX_UPLOAD_MB << 20:
        raise jobs.too_large()
    tmp = tempfile.mkdtemp(prefix="earthwork_")
    up = jobs.Upload(id=uuid.uuid4().hex, filename=name, size=size, path=os.path.join(tmp, "upload.pdf"))
    open(up.path, "wb").close()
    with jobs.LOCK:
        jobs.UPLOADS[up.id] = up
    return {"upload_id": up.id, "chunk_size": settings.UPLOAD_CHUNK_MB << 20}


@router.put("/api/uploads/{upload_id}/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request):
    """Chunks arrive in order. Re-sending the last chunk (a retry after a
    lost response) is acknowledged without writing it twice."""
    up = jobs.get_upload(upload_id)
    if index < up.next_chunk:
        return {"received": up.received}
    if index != up.next_chunk:
        raise HTTPException(409, f"Expected chunk {up.next_chunk}, got {index}.")
    data = bytearray()
    async for part in request.stream():
        data += part
        if len(data) > max(settings.UPLOAD_CHUNK_MB, 1) << 20 or up.received + len(data) > up.size:
            raise HTTPException(413, "Chunk too large for this upload.")
    if not data:
        raise HTTPException(400, "Empty chunk.")
    with open(up.path, "ab") as fh:
        fh.write(data)
    up.received += len(data)
    up.next_chunk += 1
    up.touched = time.time()
    return {"received": up.received}


@router.post("/api/uploads/{upload_id}/complete")
def complete_upload(upload_id: str):
    up = jobs.get_upload(upload_id)
    if up.received != up.size:
        raise HTTPException(400, f"Upload incomplete: {up.received} of {up.size} bytes received.")
    with jobs.LOCK:
        jobs.UPLOADS.pop(upload_id, None)
    jobs.expire_jobs()
    return {"job_id": jobs.start_job(up.filename, up.path).id}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    body = {"job_id": job.id, "filename": job.filename, "status": job.status,
            "progress": {"done": job.done, "total": job.total}, "error": job.error,
            "queue_position": jobs.queue_position(job)}
    if job.status == "done":
        body["result"] = job.payload
    return body


@router.put("/api/jobs/{job_id}/edits")
def save_edits(job_id: str, body: dict = Body(...)):
    """Save hand-edited areas for one surface, e.g.
    {"surface": "subgrade", "edits": {"12": {"fill_ft2": 40.5, "cut_ft2": 3}, "13": null}}.
    An edit may carry "marks": [{"kind": "fill"|"cut", "pts": [[x, y], ...]}],
    areas outlined on the section image (pixels); they are kept, measured and
    drawn on the annotated PDF. The fill/cut totals are what volumes use.
    null puts a section back to its measured areas. Road volumes for that
    surface are recomputed; the updated result is returned."""
    job = jobs.require_finished(jobs.get_job(job_id))
    surface, parsed = parse_edits(job, body)
    with jobs.LOCK:
        apply_edits(job.payload, surface, parsed)
        return job.payload
