"""The shared page pool: pages of several jobs read in parallel, a worker that
dies is replaced and its pages tried again, and a page that keeps failing is
left out with a warning instead of failing the whole plan set."""
import os
import threading
import time

from gdot_earthwork.plan_set import _read_pages
from gdot_earthwork.pool import PagePool, default_workers


def _page(pdf, i, crash_page=None, marker=None, slow=0.0):
    """A stand-in for reading a page: returns [i]; on `crash_page` the worker
    process dies (as when the host kills it for memory) -- once if `marker`
    names a file it may create, every time otherwise."""
    time.sleep(slow)
    if i == crash_page and (marker is None or not os.path.exists(marker)):
        if marker:
            open(marker, "w").close()
        os._exit(1)
    if i == -1:
        raise ValueError("unreadable")
    return [i]


def test_pages_read_in_parallel_and_in_full():
    pool = PagePool(2)
    try:
        seen = []
        out = _read_pages(pool, list(range(6)), ("x.pdf",), lambda d, t: seen.append((d, t)), [], read=_page)
        assert out == {i + 1: [i] for i in range(6)} and seen[-1] == (6, 6)
    finally:
        pool.close()


def test_dead_worker_is_replaced_and_its_page_retried(tmp_path):
    pool = PagePool(2)
    try:
        warnings = []
        out = _read_pages(pool, list(range(5)), ("x.pdf", 3, str(tmp_path / "crashed")), None, warnings, read=_page)
        assert out == {i + 1: [i] for i in range(5)} and not warnings
    finally:
        pool.close()


def test_page_that_keeps_failing_is_left_out_with_a_warning():
    pool = PagePool(2)
    try:
        warnings = []
        out = _read_pages(pool, list(range(4)), ("x.pdf", 2), None, warnings, read=_page)
        assert out[3] == [] and all(out[i + 1] == [i] for i in (0, 1, 3))
        assert any("page 3" in w for w in warnings)
    finally:
        pool.close()


def test_several_jobs_share_the_pool():
    pool = PagePool(2)
    results = {}

    def job(name):
        with pool.job():
            results[name] = _read_pages(pool, list(range(4)), ("x.pdf", None, None, 0.2), None, [], read=_page)
    try:
        threads = [threading.Thread(target=job, args=(n,)) for n in "abcd"]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert all(results[n] == {i + 1: [i] for i in range(4)} for n in "abcd")
        assert time.time() - t0 < 60
    finally:
        pool.close()


def test_default_workers_fit_the_machine():
    assert 1 <= default_workers() <= 8
