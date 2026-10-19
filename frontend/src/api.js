// Where the API lives. Empty = this page's own origin: Vite proxies /api in
// development, and the Docker image serves the app and the API together.
// A frontend hosted on its own (Vercel) gets the backend's URL from
// VITE_API_BASE (.env.production), and calls it cross-origin (CORS).
const BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/+$/, '');

/** The API's origin, for code samples a user copies and runs elsewhere. */
export const API_ORIGIN = BASE || window.location.origin;

// The backend's API key (VITE_API_KEY, set in the hosting platform's
// environment, never committed), sent as X-API-Key on every request. Images
// and downloads are plain links that cannot carry a header: they get it as
// ?api_key= instead (see withKey).
const KEY = import.meta.env.VITE_API_KEY || '';
const KEY_HEADER = KEY ? { 'X-API-Key': KEY } : {};

function call(url, opts = {}) {
  return fetch(url, { ...opts, headers: { ...KEY_HEADER, ...opts.headers } });
}

function withKey(url) {
  return KEY ? `${url}${url.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(KEY)}` : url;
}

async function json(res) {
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error((body && body.detail) || `Request failed (${res.status})`);
  return body;
}

export async function health() {
  return json(await call(`${BASE}/api/health`));
}

export async function getAccuracy() {
  return json(await call(`${BASE}/api/accuracy`));
}

// One XMLHttpRequest (fetch cannot report upload progress). Resolves with
// {status, body}; rejects only when the connection fails.
function send(method, url, payload, headers, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    Object.entries({ ...KEY_HEADER, ...headers }).forEach(([k, v]) => xhr.setRequestHeader(k, v));
    if (onProgress) xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onerror = () => reject(new Error('network'));
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* an HTML error page from a proxy */ }
      resolve({ status: xhr.status, body });
    };
    xhr.send(payload);
  });
}

function failure(res, what) {
  if (res.body && res.body.detail) return new Error(res.body.detail);
  if (res.status === 413) {
    return new Error('A web server or proxy in front of the app rejected the upload as too large (HTTP 413).');
  }
  return new Error(`${what} failed (HTTP ${res.status}).`);
}

const MIN_CHUNK = 256 * 1024;

/**
 * Upload a plan set in chunks, so request-size limits of anything in front of
 * the app (nginx, dev tunnels such as Codespaces, hosting platforms) never
 * see the whole file at once. A chunk refused as too large is retried at half
 * the size; a dropped connection is retried a few times.
 * onProgress(sentBytes, totalBytes).
 */
export async function createJob(file, onProgress) {
  const json = { 'Content-Type': 'application/json' };
  const start = await send('POST', `${BASE}/api/uploads`, JSON.stringify({ filename: file.name, size: file.size }), json);
  if (start.status !== 200) throw failure(start, 'Upload');
  const { upload_id: id } = start.body;
  let chunk = start.body.chunk_size;
  let offset = 0;
  let index = 0;
  let retries = 0;
  while (offset < file.size) {
    const piece = file.slice(offset, offset + chunk);
    let res;
    try {
      res = await send('PUT', `${BASE}/api/uploads/${id}/${index}`, piece, { 'Content-Type': 'application/octet-stream' },
        (sent) => onProgress && onProgress(offset + sent, file.size));
    } catch {
      if (++retries > 4) throw new Error('The upload was interrupted. Check the connection and try again.');
      await new Promise((r) => setTimeout(r, 1000 * 2 ** retries));
      continue;
    }
    if (res.status === 413 && chunk > MIN_CHUNK && !(res.body && res.body.detail && !/chunk/i.test(res.body.detail))) {
      chunk = Math.max(MIN_CHUNK, Math.floor(chunk / 2));     // a proxy with a small body limit: send smaller pieces
      continue;
    }
    if (res.status !== 200) throw failure(res, 'Upload');
    offset += piece.size;
    index += 1;
    retries = 0;
    if (onProgress) onProgress(offset, file.size);
  }
  const done = await send('POST', `${BASE}/api/uploads/${id}/complete`, null);
  if (done.status !== 200) throw failure(done, 'Upload');
  return done.body;
}

export async function getJob(id) {
  return json(await call(`${BASE}/api/jobs/${id}`));
}

/** Save hand-edited areas for one surface. `edits` maps section id ->
 * {fill_ft2, cut_ft2}, or null to go back to the measured areas.
 * Returns the updated result (volumes recomputed by the server). */
export async function saveEdits(jobId, surface, edits) {
  return json(await call(`${BASE}/api/jobs/${jobId}/edits`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surface, edits }),
  }));
}

const profiles = new Map();

/** A section's traced lines in its image's pixels (cached: they never change). */
export function getProfile(path) {
  if (!profiles.has(path)) {
    const p = call(assetUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    p.catch(() => profiles.delete(path));
    profiles.set(path, p);
  }
  return profiles.get(path);
}

/** Feet-to-pixel calibration per sheet, and the cut / fill shapes, for a finished job. */
export async function getCalibration(jobId) {
  return json(await call(`${BASE}/api/jobs/${jobId}/calibration`));
}

/** Every cross-section in the received format: left / right areas, shapes, confidence and alerts. */
export async function getCrossSections(jobId, surface) {
  const q = surface ? `?surface=${encodeURIComponent(surface)}` : '';
  return json(await call(`${BASE}/api/jobs/${jobId}/cross-sections${q}`));
}

/** A JSON Schema the API's responses are validated against ("calibration", "cross-sections"). */
export async function getSchema(name) {
  return json(await call(`${BASE}/api/schemas/${name}`));
}

/** Stored comparisons with results provided for a plan set (e.g. SR 5). */
export async function listComparisons() {
  return json(await call(`${BASE}/api/comparisons`));
}

export async function getComparison(name) {
  return json(await call(`${BASE}/api/comparisons/${name}`));
}

/** Compare an open takeoff with a provided results file (xlsx / csv). */
export async function compareJob(jobId, file, surface) {
  const form = new FormData();
  form.append('file', file);
  const q = surface ? `?surface=${encodeURIComponent(surface)}` : '';
  return json(await call(`${BASE}/api/jobs/${jobId}/compare${q}`, { method: 'POST', body: form }));
}

/** A sheet's measured sections, in the sheet image's coordinates (cached). */
export const getSheet = (jobId, page) => getProfile(`/api/jobs/${jobId}/pages/${page}.json`);

export function assetUrl(path) {
  return withKey(`${BASE}${path}`);
}

export function volumesCsvUrl(jobId, excludedRoadIds, surface) {
  const q = excludedRoadIds.map((id) => `exclude=${encodeURIComponent(id)}`);
  if (surface) q.push(`surface=${encodeURIComponent(surface)}`);
  return withKey(`${BASE}/api/jobs/${jobId}/volumes.csv${q.length ? `?${q.join('&')}` : ''}`);
}

export function annotatedPdfUrl(jobId, excludedRoadIds, surface) {
  const q = excludedRoadIds.map((id) => `exclude=${encodeURIComponent(id)}`);
  if (surface) q.push(`surface=${encodeURIComponent(surface)}`);
  return withKey(`${BASE}/api/jobs/${jobId}/annotated.pdf${q.length ? `?${q.join('&')}` : ''}`);
}

export function sectionsCsvUrl(jobId) {
  return withKey(`${BASE}/api/jobs/${jobId}/sections.csv`);
}

// ---------------------------------------------------------------- public API v1
export const V1 = '/api/v1';

function parseHeaders(raw) {
  const out = {};
  (raw || '').trim().split(/[\r\n]+/).forEach((line) => {
    const i = line.indexOf(':');
    if (i > 0) out[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  });
  return out;
}

/**
 * One request to the public API, exactly as an outside client would make it.
 * Resolves with {status, headers, body, ms} whatever the status (an error is
 * part of the answer); rejects only when the connection fails or `signal`
 * aborts it. XMLHttpRequest, so an upload can report its progress.
 */
export function v1Request(method, path, { form, apiKey, signal, onUpload } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const started = performance.now();
    xhr.open(method, `${BASE}${V1}${path}`);
    xhr.setRequestHeader('Accept', 'application/json');
    if (apiKey || KEY) xhr.setRequestHeader('X-API-Key', apiKey || KEY);
    if (onUpload) xhr.upload.onprogress = (e) => onUpload(e.loaded, e.total);
    xhr.onerror = () => reject(new Error('The server could not be reached.'));
    xhr.onabort = () => reject(Object.assign(new Error('Cancelled.'), { name: 'AbortError' }));
    xhr.onload = () => {
      let body = null;
      try { body = xhr.responseText ? JSON.parse(xhr.responseText) : null; } catch { body = xhr.responseText; }
      resolve({ status: xhr.status, headers: parseHeaders(xhr.getAllResponseHeaders()), body,
        ms: Math.round(performance.now() - started) });
    };
    if (signal) {
      if (signal.aborted) { reject(Object.assign(new Error('Cancelled.'), { name: 'AbortError' })); return; }
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }
    xhr.send(form || null);
  });
}

/** The public API's OpenAPI document. */
export async function getOpenApi() {
  return json(await call(`${BASE}${V1}/openapi.json`));
}
