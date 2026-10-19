"""The public API (/api/v1) on a synthetic plan set: upload then poll, ?wait,
one job read through both endpoints and the app, errors, API keys, the
streamed upload, and the OpenAPI document and Swagger UI."""
import asyncio
import json
import os
import tempfile
import threading
import time

import jsonschema
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import main
from api import jobs, settings
from api.streaming import receive_pdf

from .synth import embankment, make_pdf

SCHEMAS = os.path.join(os.path.dirname(__file__), "..", "schemas")


def _schema(name):
    with open(os.path.join(SCHEMAS, name + ".schema.json")) as f:
        return json.load(f)


def _validate(name, body):
    jsonschema.Draft202012Validator(_schema(name)).validate(body)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    pdf = tmp_path_factory.mktemp("v1") / "plans.pdf"
    make_pdf(str(pdf), [[embankment("11+50", 651.3, 5.0), embankment("11+00", 651.7, 4.0)],
                        [embankment("12+00", 651.0, 6.0)]])
    return pdf.read_bytes()


@pytest.fixture
def client():
    return TestClient(main.app)


def _upload(client, kind, data, name="plans.pdf", **params):
    return client.post(f"/api/v1/{kind}", params=params, files={"file": (name, data, "application/pdf")})


def _poll(client, url, **params):
    for _ in range(600):
        r = client.get(url, params=params)
        if r.status_code != 202:
            return r
        assert r.headers["Retry-After"]
        time.sleep(0.1)
    raise AssertionError(f"{url} still processing")


def test_upload_then_poll_one_job_through_both_endpoints(client, plans, monkeypatch):
    # hold the processing queue, so the job is seen queued whatever the machine's speed
    gate = threading.BoundedSemaphore(1)
    gate.acquire()
    monkeypatch.setattr(jobs, "SLOTS", gate)
    r = _upload(client, "calibration", plans)
    assert r.status_code == 202, r.text
    status = r.json()
    job_id = status["id"]
    assert set(status) == {"id", "status", "progress", "links"}
    assert status["status"] == "queued" and status["progress"] == {"done": 0, "total": 0}
    assert status["links"] == {"self": f"/api/v1/calibration/{job_id}",
                               "calibration": f"/api/v1/calibration/{job_id}",
                               "crossSections": f"/api/v1/cross-sections/{job_id}"}
    assert r.headers["Location"] == status["links"]["self"] and r.headers["X-Job-Id"] == job_id
    # both GETs answer 202 with the same job while it waits
    r = client.get(f"/api/v1/calibration/{job_id}")
    assert r.status_code == 202 and r.json()["status"] == "queued"
    r = client.get(f"/api/v1/cross-sections/{job_id}", params={"surface": "finished"})
    assert r.status_code == 202 and r.json()["links"]["self"] == f"/api/v1/cross-sections/{job_id}?surface=finished"
    gate.release()

    r = _poll(client, f"/api/v1/calibration/{job_id}")
    assert r.status_code == 200, r.text
    cal = r.json()
    _validate("calibration", cal)
    assert cal["document"]["sections"] == 3 and cal["document"]["fileName"] == "plans.pdf"
    assert r.headers["Content-Location"] == f"/api/v1/calibration/{job_id}" and r.headers["X-Job-Id"] == job_id

    # the same id reads the cross-sections, on either surface
    for surface in (None, "subgrade", "finished"):
        r = client.get(f"/api/v1/cross-sections/{job_id}", params={"surface": surface} if surface else {})
        assert r.status_code == 200, r.text
        body = r.json()
        _validate("cross-sections", body)
        assert body["document"]["surface"] == (surface or "subgrade")
        assert len(body["crossSections"]) == 3
    assert client.get(f"/api/v1/cross-sections/{job_id}", params={"surface": "top"}).status_code == 400

    # it is the app's job: same store, same calibration (built once)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    assert client.get(f"/api/jobs/{job_id}/calibration").json() == cal
    assert main.JOBS[job_id].calibration is not None

    # the responses also validate against the schemas as published in the OpenAPI document
    spec = client.get("/api/v1/openapi.json").json()
    for component, response in (("Calibration", cal), ("CrossSections", body)):
        jsonschema.Draft202012Validator({"$ref": f"#/components/schemas/{component}",
                                         "components": spec["components"]}).validate(response)


def test_wait_answers_with_the_result(client, plans):
    t = time.time()
    r = _upload(client, "cross-sections", plans, wait=120)
    assert r.status_code == 200, r.text
    assert time.time() - t < 120
    body = r.json()
    _validate("cross-sections", body)
    assert body["summary"]["crossSections"] == 3
    # the id comes back in a header; it reads the calibration too
    job_id = r.headers["X-Job-Id"]
    assert r.headers["Content-Location"] == f"/api/v1/cross-sections/{job_id}"
    r = client.get(f"/api/v1/calibration/{job_id}", params={"wait": 5})
    assert r.status_code == 200
    _validate("calibration", r.json())


def test_jobs_from_the_app_are_read_through_the_api(client, plans):
    job_id = client.post("/api/jobs", files={"file": ("plans.pdf", plans, "application/pdf")}).json()["job_id"]
    r = client.get(f"/api/v1/cross-sections/{job_id}", params={"wait": 120})
    assert r.status_code == 200, r.text
    first = r.json()["crossSections"][0]
    assert first["edited"] is False
    # a hand edit saved in the app shows in the API's cross-sections
    assert client.put(f"/api/jobs/{job_id}/edits", json={
        "surface": "subgrade", "edits": {"0": {"fill_ft2": 321.0, "cut_ft2": 0}}}).status_code == 200
    body = client.get(f"/api/v1/cross-sections/{job_id}").json()
    _validate("cross-sections", body)
    x = body["crossSections"][0]
    assert x["edited"] is True and x["areas"]["total"]["fillSqFt"] == 321.0
    assert x["measuredAreas"] == first["measuredAreas"]


def test_unknown_failed_and_gone_jobs(client, plans):
    for kind in ("calibration", "cross-sections"):
        r = client.get(f"/api/v1/{kind}/nope")
        assert r.status_code == 404 and r.json()["detail"]

    # a file that starts like a PDF but isn't one fails in processing: 422 with the reason
    r = _upload(client, "calibration", b"%PDF-1.7\nnot really", name="broken.pdf", wait=60)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail and tempfile.gettempdir() not in detail          # names the file, not where it was kept
    for kind in ("calibration", "cross-sections"):
        assert client.get(f"/api/v1/{kind}/{r.headers['X-Job-Id']}").status_code == 422

    # a finished job whose PDF has gone before its calibration was built: 410
    job_id = _upload(client, "calibration", plans).json()["id"]
    while client.get(f"/api/jobs/{job_id}").json()["status"] not in ("done", "error"):
        time.sleep(0.1)
    os.remove(main.JOBS[job_id].pdf_path)
    for kind in ("calibration", "cross-sections"):
        assert client.get(f"/api/v1/{kind}/{job_id}").status_code == 410


def test_bad_uploads_are_refused_and_leave_nothing_behind(client, plans, monkeypatch):
    made = []
    mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: made.append(mkdtemp(*a, **k)) or made[-1])
    n_jobs = len(main.JOBS)
    for kind in ("calibration", "cross-sections"):
        assert client.post(f"/api/v1/{kind}", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
        assert _upload(client, kind, b"").status_code == 400                               # empty
        assert _upload(client, kind, b"%PD").status_code == 400                            # too short to be one
        assert client.post(f"/api/v1/{kind}", files={"other": ("plans.pdf", plans)}).status_code == 400
        assert client.post(f"/api/v1/{kind}", content=plans,
                           headers={"Content-Type": "application/pdf"}).status_code == 400  # not a form
        for wait in (-1, 601, "soon"):
            r = _upload(client, kind, plans, wait=wait)
            assert r.status_code == 400 and "wait" in r.json()["detail"]
    assert "not a PDF" in client.post("/api/v1/calibration", files={"file": ("a.pdf", b"hello")}).json()["detail"]

    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 1)
    # refused on its declared length, before it is read
    r = _upload(client, "calibration", b"%PDF-1.7\n" + b"0" * (3 << 20))
    assert r.status_code == 413 and "MAX_UPLOAD_MB" in r.json()["detail"]
    # within the declared-length slack: refused as the file passes the limit
    r = _upload(client, "cross-sections", b"%PDF-1.7\n" + b"0" * (3 << 19))
    assert r.status_code == 413
    assert len(main.JOBS) == n_jobs                     # no job started
    assert made and not any(os.path.exists(d) for d in made)


def _request(body: bytes, chunk: int, boundary: str = "XyZ") -> Request:
    messages = [{"type": "http.request", "body": body[i:i + chunk], "more_body": i + chunk < len(body)}
                for i in range(0, len(body), chunk)] or [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return messages.pop(0)
    headers = [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers, "query_string": b""}, receive)


def test_streamed_form_across_chunk_boundaries():
    # boundary-like bytes inside the file, another field before it, a path in the file name
    pdf = b"%PDF-1.7\n" + b"\r\n--XyZ not the end\r\n-" * 40 + bytes(range(256)) * 4
    body = (b'--XyZ\r\nContent-Disposition: form-data; name="note"\r\n\r\nhello\r\n'
            b'--XyZ\r\nContent-Disposition: form-data; name="file"; filename="C:\\plans\\sr5.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n" + pdf + b"\r\n--XyZ--\r\n")
    for chunk in (1, 7, 64, len(body)):
        name, path = asyncio.run(receive_pdf(_request(body, chunk)))
        with open(path, "rb") as fh:
            assert fh.read() == pdf, chunk
        assert name == "sr5.pdf"
    # cut off before the closing boundary: refused
    with pytest.raises(HTTPException) as err:
        asyncio.run(receive_pdf(_request(body[:-20], 64)))
    assert err.value.status_code == 400


def test_api_keys(client, plans, monkeypatch):
    assert settings.env_list(" k-one, k-two ,,") == ["k-one", "k-two"]
    monkeypatch.setattr(settings, "API_KEYS", ("k-one", "k-two"))
    for headers in ({}, {"X-API-Key": "k-three"}, {"X-API-Key": ""}, {"X-API-Key": "K-ONE"}):
        r = client.get("/api/v1/calibration/nope", headers=headers)
        assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "APIKey", headers
    assert client.get("/api/v1/calibration/nope", headers={"X-API-Key": "k-two"}).status_code == 404
    assert client.get("/api/v1/schemas/calibration").status_code == 401
    assert client.get("/api/v1/schemas/calibration", headers={"X-API-Key": "k-one"}).status_code == 200
    # an upload without a key is refused before it is read or a job started
    n_jobs = len(main.JOBS)
    assert _upload(client, "cross-sections", plans).status_code == 401
    assert len(main.JOBS) == n_jobs
    # docs stay readable, so clients can find out how to authenticate; the app's endpoints need no key
    assert client.get("/api/v1/docs").status_code == 200
    assert client.get("/api/v1/openapi.json").status_code == 200
    assert client.get("/api/health").status_code == 200
    monkeypatch.setattr(settings, "API_KEYS", ())
    assert client.get("/api/v1/calibration/nope").status_code == 404        # open when no keys are set


def test_openapi_document_and_docs(client):
    spec = client.get("/api/v1/openapi.json").json()
    assert spec["info"]["title"] == "Earthwork Takeoff API"
    assert spec["servers"] == [{"url": "/api/v1"}]
    ops = {(path, method): op for path, item in spec["paths"].items() for method, op in item.items()}
    expected = {("/calibration", "post"): ("Calibration", "Calibration"),
                ("/calibration/{job_id}", "get"): ("Calibration", "Calibration"),
                ("/cross-sections", "post"): ("Cross-sections", "CrossSections"),
                ("/cross-sections/{job_id}", "get"): ("Cross-sections", "CrossSections")}
    components = spec["components"]["schemas"]
    for key, (tag, component) in expected.items():
        op = ops[key]
        assert op["tags"] == [tag] and "curl" in op["description"]
        assert op["responses"]["200"]["content"]["application/json"]["schema"] == {
            "$ref": f"#/components/schemas/{component}"}
        assert op["responses"]["202"]["content"]["application/json"]["schema"]["$ref"].endswith("/JobStatus")
        assert {"200", "202", "400", "401", "422"} <= set(op["responses"])
        if key[1] == "post":
            form = op["requestBody"]["content"]["multipart/form-data"]["schema"]
            assert form["required"] == ["file"] and form["properties"]["file"]["format"] == "binary"
        else:
            assert {"404", "410"} <= set(op["responses"])
    # the JSON Schema files, as components
    for name, component in (("calibration", "Calibration"), ("cross-sections", "CrossSections")):
        file = _schema(name)
        assert components[component]["required"] == file["required"]
        assert set(components[component]["properties"]) == set(file["properties"])
        assert all(f"{component}.{d}" in components for d in file["$defs"])
    # every reference in the document resolves
    def refs(node):
        if isinstance(node, dict):
            yield from ([node["$ref"]] if isinstance(node.get("$ref"), str) else [])
            for v in node.values():
                yield from refs(v)
        elif isinstance(node, list):
            for v in node:
                yield from refs(v)
    for ref in set(refs(spec)):
        assert ref.startswith("#/components/schemas/") and ref.rsplit("/", 1)[1] in components, ref
    scheme = spec["components"]["securitySchemes"]["APIKey"]
    assert (scheme["type"], scheme["in"], scheme["name"]) == ("apiKey", "header", "X-API-Key")
    assert all(op["security"] == [{"APIKey": []}] for op in ops.values())
    assert {t["name"] for t in spec["tags"]} >= {"Calibration", "Cross-sections"}
    assert "curl" in spec["info"]["description"]

    docs = client.get("/api/v1/docs")
    assert docs.status_code == 200 and docs.headers["content-type"].startswith("text/html")
    assert "swagger-ui" in docs.text and "/api/v1/openapi.json" in docs.text
    r = client.get("/api/v1/schemas/cross-sections")
    assert r.status_code == 200 and r.json() == _schema("cross-sections")
    assert client.get("/api/v1/schemas/nope").status_code == 404


def test_cors_as_the_app(client):
    origin = settings.ALLOWED_ORIGINS[0]
    r = client.get("/api/v1/schemas/calibration", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == origin
    # a browser client may read the job id and where to poll
    assert {"x-job-id", "location", "retry-after"} <= set(r.headers["access-control-expose-headers"].lower().split(", "))
    r = client.options("/api/v1/calibration", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
    r = client.get("/api/v1/schemas/calibration", headers={"Origin": "https://elsewhere.example"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_from_vercel(client):
    # the frontend hosted on Vercel: production and preview deployments alike
    for origin in ("https://infra-p.vercel.app", "https://infra-p-git-main-someone.vercel.app"):
        r = client.options("/api/uploads", headers={
            "Origin": origin, "Access-Control-Request-Method": "PUT", "Access-Control-Request-Headers": "content-type"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin
    r = client.get("/api/health", headers={"Origin": "https://evil.vercel.app.example"})
    assert "access-control-allow-origin" not in r.headers
