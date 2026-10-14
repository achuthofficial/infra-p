"""
Settings, read once from the environment at import.

Modules read them as attributes (settings.MAX_UPLOAD_MB) at call time and
never copy them, so a test or a tool that changes one -- directly or through
main, which forwards its old module-level names here -- changes what every
endpoint sees.
"""
from __future__ import annotations

import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env_list(value: str | None) -> list[str]:
    """A comma-separated setting as a list, blanks dropped."""
    return [v.strip() for v in (value or "").split(",") if v.strip()]


JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", 6 * 3600))
MAX_JOBS_KEPT = int(os.environ.get("MAX_JOBS_KEPT", 20))     # finished jobs kept in memory, newest first
# plan sets processed at once; they share the WORKERS page processes fairly,
# so several people's uploads all make progress; any more wait in a queue
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 4))
# page-reading processes for the whole server; 0 = fit the container's CPUs and
# memory (gdot_earthwork.pool.default_workers: ~0.6 GB each on large scans)
WORKERS = int(os.environ.get("WORKERS", 0))
# Scanned plan sets are big (the dataset's largest is ~120 MB); the cap only
# guards the disk. Uploads arrive in chunks (see /api/uploads), so proxies with
# their own request-size limits in front of the app don't get in the way.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 2048))
UPLOAD_CHUNK_MB = int(os.environ.get("UPLOAD_CHUNK_MB", 8))
UPLOAD_TTL_SECONDS = 3600               # an unfinished chunked upload is dropped after this
ALLOWED_ORIGINS = env_list(os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,https://improved-space-journey-w4rrv6rx5w6cgxxv-5173.app.github.dev"))
# Keys for the public API (/api/v1), sent as X-API-Key. Unset or empty leaves
# it open; the app's own /api endpoints never ask for one.
API_KEYS = tuple(env_list(os.environ.get("API_KEYS")))

ACCURACY_FILE = os.environ.get("ACCURACY_FILE", os.path.join(BACKEND_DIR, "accuracy.json"))
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", os.path.join(BACKEND_DIR, "..", "frontend", "dist"))
COMPARISONS_DIR = os.path.join(BACKEND_DIR, "comparisons")
SCHEMAS_DIR = os.path.join(BACKEND_DIR, "schemas")
