"""
Jobs: an uploaded plan set, processed in the background, and its result.

Processing a plan set takes seconds to minutes, so uploads return at once
and the client polls the job. Jobs run MAX_CONCURRENT_JOBS at a time, their
pages read by one pool of worker processes they share fairly (pool.PagePool);
the rest wait as "queued", in upload order. Jobs live in memory and
expire after JOB_TTL_SECONDS; nothing is written to disk except the upload
itself, which is kept while its job lives (for the annotated PDF export and
the calibration) and deleted with it.

The app's routes and the public API (/api/v1) share this store and queue: a
job started through either can be read through both.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import HTTPException

from gdot_earthwork.plan_set import PlanSetResult, process_plan_set
from gdot_earthwork.pool import PagePool

from . import settings
from .payload import serialize

PDF_MAGIC = b"%PDF-"


@dataclass
class Job:
    id: str
    filename: str
    created: float = field(default_factory=time.time)
    status: str = "queued"            # queued | running | done | error
    done: int = 0
    total: int = 0
    error: str | None = None
    result: PlanSetResult | None = None
    payload: dict | None = None
    pdf_path: str | None = None       # the upload, kept while the job lives
    pages: dict = field(default_factory=dict)   # rendered sheet images, page -> PNG (a few kept)
    calibration: dict | None = None             # the calibration response, built once
    # held while the calibration is built, so two first readers (the app and an
    # API client) don't both run its OCR
    build_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()
SLOTS = threading.BoundedSemaphore(settings.MAX_CONCURRENT_JOBS)


def expire_jobs() -> None:
    """Drop finished jobs older than JOB_TTL_SECONDS, and beyond the newest
    MAX_JOBS_KEPT finished ones (their results and PDFs are what the server's
    memory and disk hold)."""
    cutoff = time.time() - settings.JOB_TTL_SECONDS
    with LOCK:
        finished = sorted((j for j in JOBS.values() if j.status in ("done", "error")), key=lambda j: j.created)
        old = [j for j in finished if j.created < cutoff]
        surplus = finished[:max(0, len(finished) - settings.MAX_JOBS_KEPT)]
        for job in {j.id: j for j in old + surplus}.values():
            JOBS.pop(job.id, None)
            if job.pdf_path:
                shutil.rmtree(os.path.dirname(job.pdf_path), ignore_errors=True)


def run_job(job: Job, path: str) -> None:
    def progress(done: int, total: int) -> None:
        job.done, job.total = done, total

    with SLOTS:                       # stays "queued" until a slot is free
        job.status = "running"
        try:
            # section images carry the lines only; the app shades fill/cut from
            # the profile so the picture follows the areas the user types
            res = process_plan_set(path, progress=progress, overlays="lines", name=job.filename,
                                   pool=PagePool.shared(settings.WORKERS or None))
            job.result = res
            job.payload = serialize(job, res)
            job.pdf_path = path
            job.status = "done"
        except Exception as e:  # surfaced to the client, not swallowed (naming the file, not where it was kept)
            job.error = f"{type(e).__name__}: {e}".replace(path, job.filename)
            job.status = "error"        # last: whoever sees "error" also finds the reason
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def start_job(name: str, path: str) -> Job:
    """Check the uploaded file is a PDF and start processing it."""
    with open(path, "rb") as fh:
        if fh.read(len(PDF_MAGIC)) != PDF_MAGIC:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            raise HTTPException(400, "That file is not a PDF.")
    job = Job(id=uuid.uuid4().hex[:12], filename=name)
    with LOCK:
        JOBS[job.id] = job
    threading.Thread(target=run_job, args=(job, path), daemon=True).start()
    return job


def queue_position(job: Job) -> int | None:
    """1 for the next queued job to start, None once it is running or over."""
    if job.status != "queued":
        return None
    with LOCK:
        waiting = sorted((j.created, j.id) for j in JOBS.values() if j.status == "queued")
    return next((n for n, (_, jid) in enumerate(waiting, 1) if jid == job.id), None)


def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return job


def require_finished(job: Job) -> Job:
    if job.payload is None or job.result is None:
        raise HTTPException(409, "Job is not finished.")
    return job


def require_pdf(job: Job) -> str:
    """The job's original PDF, for what is drawn or read from it on demand."""
    if not job.pdf_path or not os.path.exists(job.pdf_path):
        raise HTTPException(410, "The original PDF is no longer available for this job.")
    return job.pdf_path


# ---------------------------------------------------------------- receiving a PDF
def too_large() -> HTTPException:
    return HTTPException(413, f"This PDF is larger than the server's {settings.MAX_UPLOAD_MB} MB upload limit. "
                              f"Raise it with the MAX_UPLOAD_MB setting on the backend.")


class UploadWriter:
    """An upload written to its own temporary folder as it arrives, a chunk at
    a time, and refused (413) the moment it passes MAX_UPLOAD_MB -- a 2 GB scan
    never sits in memory. Use as a context manager: on any error the folder is
    deleted; on success the file at .path is the job's to keep."""

    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="earthwork_")
        self.path = os.path.join(self.dir, "upload.pdf")
        self.size = 0
        self.head = b""                 # the first bytes, to tell a PDF early
        self._fh = open(self.path, "wb")  # noqa: SIM115 -- closed by __exit__ (used as a context manager)

    def write(self, data: bytes) -> None:
        self.size += len(data)
        if self.size > settings.MAX_UPLOAD_MB << 20:
            raise too_large()
        if len(self.head) < len(PDF_MAGIC):
            self.head += data[:len(PDF_MAGIC) - len(self.head)]
        self._fh.write(data)

    def looks_like_pdf(self) -> bool:
        """False once enough has arrived to tell it is not one."""
        return PDF_MAGIC.startswith(self.head)

    def __enter__(self) -> UploadWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._fh.close()
        if exc_type is not None:
            shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------- chunked upload
# Big scans in one request trip the request-size limits of whatever sits in
# front of the app (nginx's default is 1 MB; dev tunnels such as Codespaces
# port forwarding and hosting platforms have their own). Sent in chunks, each
# request stays small whatever the file size.
@dataclass
class Upload:
    id: str
    filename: str
    size: int                   # declared total size, bytes
    path: str
    received: int = 0           # bytes written so far
    next_chunk: int = 0         # index of the next chunk expected
    touched: float = field(default_factory=time.time)


UPLOADS: dict[str, Upload] = {}


def expire_uploads() -> None:
    cutoff = time.time() - settings.UPLOAD_TTL_SECONDS
    with LOCK:
        for k in [k for k, u in UPLOADS.items() if u.touched < cutoff]:
            shutil.rmtree(os.path.dirname(UPLOADS.pop(k).path), ignore_errors=True)


def get_upload(upload_id: str) -> Upload:
    up = UPLOADS.get(upload_id)
    if up is None:
        raise HTTPException(404, "Unknown or expired upload; start again.")
    return up
