"""
A pool of worker processes that reads plan-set pages, shared by every job.

One plan set's pages are independent, so they are read in parallel; several
plan sets (several people uploading at once) share the same workers instead of
each starting its own, which would multiply the memory a page takes (a large
scan peaks around 400 MB) by jobs x cores. Each running job gets a fair share
of the workers -- the pool size divided by the jobs running -- so a 60-page set
doesn't hold a 5-page one back.

A worker that dies (out of memory, killed by the host) breaks the whole
executor; the pool then starts a fresh one and the pages that were in flight
are tried again. Workers are also replaced every MAX_TASKS_PER_CHILD pages, so
memory a long run leaves behind is given back.

    PagePool(workers)            a pool; PagePool.shared() is the server's
    default_workers()            a size that fits this machine's CPUs and memory
"""
from __future__ import annotations

import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

MAX_TASKS_PER_CHILD = 25        # pages a worker reads before it is replaced
PAGE_MEMORY_GB = 0.6            # what one worker may need on a large scanned page, with headroom


def _cgroup_value(*paths: str) -> str | None:
    for p in paths:
        try:
            with open(p) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return None


def cpu_limit() -> float:
    """CPUs this container may use: its cgroup quota, else the CPUs it may run on."""
    raw = _cgroup_value("/sys/fs/cgroup/cpu.max")                      # cgroup v2: "quota period"
    if raw and not raw.startswith("max"):
        quota, period = raw.split()[:2]
        return max(1.0, int(quota) / int(period))
    quota = _cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")        # cgroup v1
    period = _cgroup_value("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return max(1.0, int(quota) / int(period))
    try:
        return float(len(os.sched_getaffinity(0)))
    except AttributeError:
        return float(os.cpu_count() or 1)


def memory_limit_gb() -> float | None:
    """This container's memory limit in GB, or None when unlimited / unknown."""
    raw = _cgroup_value("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if not raw or raw == "max":
        return None
    gb = int(raw) / 2 ** 30
    return gb if gb < 1024 else None                                   # v1 reports "unlimited" as a huge number


def default_workers(cap: int = 8) -> int:
    """As many workers as there are CPUs, as far as memory allows (1 GB kept
    for the server and finished jobs), at most `cap`."""
    n = int(cpu_limit())
    mem = memory_limit_gb()
    if mem is not None:
        n = min(n, int((mem - 1.0) / PAGE_MEMORY_GB))
    return max(1, min(n, cap))


class PagePool:
    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or default_workers()
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._jobs = 0

    _shared: PagePool | None = None
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls, workers: int | None = None) -> PagePool:
        """The server-wide pool, created on first use."""
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(workers)
            return cls._shared

    def executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                # "spawn": a fresh interpreter per worker, never a fork of a
                # server holding threads and other jobs' results
                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=MAX_TASKS_PER_CHILD)
            return self._executor

    def restart(self, broken: ProcessPoolExecutor) -> None:
        """Replace an executor whose worker died; every job using it resubmits."""
        with self._lock:
            if self._executor is broken:
                self._executor = None
                broken.shutdown(wait=False, cancel_futures=True)

    def share(self) -> int:
        """Pages one job may have in flight: an equal share of the workers."""
        return max(1, math.ceil(self.workers / max(1, self._jobs)))

    @contextmanager
    def job(self):
        """Count a job as running while it reads pages (for share())."""
        with self._lock:
            self._jobs += 1
        try:
            yield self
        finally:
            with self._lock:
                self._jobs -= 1

    def close(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
