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
# each job already runs its pages in parallel on all cores; more jobs at once
# would only slow every one of them down, so further uploads wait their turn
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 1))
# processes a job reads its pages with; 0 = min(4, CPUs). Each holds a rendered
# page (~200 MB on big scans), so on a small instance set it to 1 or 2
WORKERS = int(os.environ.get("WORKERS", 0))
# Scanned plan sets are big (the dataset's largest is ~120 MB); the cap only
# guards the disk. Uploads arrive in chunks (see /api/uploads), so proxies with
# their own request-size limits in front of the app don't get in the way.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 2048))
UPLOAD_CHUNK_MB = int(os.environ.get("UPLOAD_CHUNK_MB", 8))
UPLOAD_TTL_SECONDS = 3600               # an unfinished chunked upload is dropped after this
ALLOWED_ORIGINS = env_list(os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,https://improved-space-journey-w4rrv6rx5w6cgxxv-5173.app.github.dev"))
# Origins matching this may call the API too: by default every Vercel
# deployment (production and each preview URL) of the separately hosted
# frontend. The API sends no cookies, so this only lets those pages read it.
# Empty turns it off.
ALLOWED_ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://[a-z0-9-]+\.vercel\.app") or None
# Keys for the public API (/api/v1), sent as X-API-Key. Unset or empty leaves
# it open; the app's own /api endpoints never ask for one.
API_KEYS = tuple(env_list(os.environ.get("API_KEYS")))

ACCURACY_FILE = os.environ.get("ACCURACY_FILE", os.path.join(BACKEND_DIR, "accuracy.json"))
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", os.path.join(BACKEND_DIR, "..", "frontend", "dist"))
COMPARISONS_DIR = os.path.join(BACKEND_DIR, "comparisons")
SCHEMAS_DIR = os.path.join(BACKEND_DIR, "schemas")
