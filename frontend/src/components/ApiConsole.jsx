import { useEffect, useMemo, useRef, useState } from 'react';
import { API_ORIGIN, getOpenApi, V1, v1Request } from '../api';
import {
  IconAlert, IconArrow, IconChevron, IconCode, IconCopy, IconExternal, IconFile, IconKey, IconPlay, IconUpload,
} from './Icons';
import { JsonTree } from './JsonView';
import { InfoTip } from './Ui';

/*
 * API console: the public v1 endpoints as an outside client sees them. Upload
 * a plan set (or name an existing job), watch the 202s until the 200, then
 * explore the JSON; plus a reference built from the API's own OpenAPI
 * document and code samples to copy.
 */

const ENDPOINTS = {
  calibration: { label: 'Calibration', blurb: 'Feet-to-pixel scale of every sheet, and every cut / fill area as a page-space shape.' },
  'cross-sections': { label: 'Cross-sections', blurb: 'Every cross-section: station, street, left / right / total cut and fill, confidence and alerts.' },
};
const SHOWN_HEADERS = ['x-job-id', 'location', 'retry-after', 'content-location', 'content-type', 'content-length', 'www-authenticate'];
const POLL_MS = 1500;
const mb = (b) => `${(b / 1048576).toFixed(b < 10485760 ? 1 : 0)} MB`;
const secs = (ms) => `${(ms / 1000).toFixed(ms < 60000 ? 1 : 0)} s`;
const sqft = (v) => Math.round(v).toLocaleString();
const ID_RE = /^[\w-]+$/;

function stored(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function store(key, value) {
  try { value ? localStorage.setItem(key, value) : localStorage.removeItem(key); } catch { /* private mode */ }
}
function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(t); reject(Object.assign(new Error('Cancelled.'), { name: 'AbortError' })); }, { once: true });
  });
}
function getPath(endpoint, id, surface) {
  return `/${endpoint}/${encodeURIComponent(id)}${endpoint === 'cross-sections' && surface ? `?surface=${surface}` : ''}`;
}

/* ---------------------------------------------------------------- summaries */
function CalibrationSummary({ body }) {
  const cals = body.calibrations || [];
  const ppf = cals.map((c) => c.pixelsPerFootX).filter((v) => typeof v === 'number');
  const lo = Math.min(...ppf);
  const hi = Math.max(...ppf);
  const scale = body.printScale ? `1 : ${body.printScale.ratio}` : body.documentScale ? body.documentScale.text : 'Not printed';
  const scaleNote = body.printScale ? body.printScale.meaning
    : body.documentScale ? `Scale note on page ${body.documentScale.pageNumber}` : 'Scale measured from each sheet’s grid';
  return (
    <>
      <div className="api-stats">
        <div><b className="num">{cals.length}{body.document && <small>/{body.document.crossSectionPages}</small>}</b><span>cross-section sheets calibrated</span></div>
        <div><b className="num">{ppf.length ? (lo === hi ? lo.toFixed(2) : `${lo.toFixed(1)}–${hi.toFixed(1)}`) : '—'}</b><span>pixels per foot (X)</span></div>
        <div><b className="num">{(body.shapes || []).length}</b><span>cut / fill shapes</span></div>
        <div><b className="api-scale" title={scaleNote}>{scale}</b><span>print scale <InfoTip label="About the print scale">{scaleNote}</InfoTip></span></div>
      </div>
      {body.uncalibratedPages?.length > 0 && (
        <div className="warning">No scale found for page{body.uncalibratedPages.length > 1 ? 's' : ''} {body.uncalibratedPages.join(', ')}.</div>
      )}
    </>
  );
}

function CrossSectionSummary({ body }) {
  const xs = body.crossSections || [];
  const cut = xs.reduce((s, x) => s + (x.areas?.total?.cutSqFt || 0), 0);
  const fill = xs.reduce((s, x) => s + (x.areas?.total?.fillSqFt || 0), 0);
  const streets = new Set(xs.map((x) => x.streetName).filter(Boolean)).size;
  const edited = xs.filter((x) => x.edited).length;
  const { summary = {} } = body;
  return (
    <>
      <div className="api-stats">
        <div><b className="num">{summary.crossSections ?? xs.length}</b><span>cross-sections{streets ? ` on ${streets} street${streets > 1 ? 's' : ''}` : ''}</span></div>
        <div><b className="num cut-text">{sqft(cut)}</b><span>total cut area, sq ft</span></div>
        <div><b className="num fill-text">{sqft(fill)}</b><span>total fill area, sq ft</span></div>
        <div>
          <b className={`num ${summary.alerts ? 'warn-text' : 'fill-text'}`}>{summary.passed ?? '—'}<small>/{summary.crossSections ?? xs.length}</small></b>
          <span>pass {body.passScore ?? 85} confidence{summary.alerts ? ` · ${summary.alerts} alert${summary.alerts > 1 ? 's' : ''}` : ''}</span>
        </div>
      </div>
      <div className="chips">
        <span className="tag info" title="Measured to">{body.document?.surface || 'subgrade'}</span>
        <span className="tag neutral" title="Areas summed over every section (not volumes)">Σ section areas</span>
        {edited > 0 && <span className="tag edited" title="Edited by hand in the app">{edited} edited</span>}
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- reference */
function typeOf(s) {
  if (!s) return '';
  if (s.$ref) return s.$ref.split('/').pop();
  if (s.enum) return s.enum.map((e) => JSON.stringify(e)).join(' | ');
  if (s.anyOf) return s.anyOf.map(typeOf).filter((t) => t !== 'null').join(' | ') + (s.anyOf.some((a) => a.type === 'null') ? ' (optional)' : '');
  if (Array.isArray(s.type)) return s.type.join(' | ');
  if (s.type === 'array') return `${typeOf(s.items) || 'any'}[]`;
  return s.type || 'object';
}
const firstSentence = (t = '') => {
  const m = t.replace(/`/g, '').match(/^[^\n]*?[.!?](\s|$)/);
  return (m ? m[0] : t.split('\n')[0]).trim();
};
const resRef = (r) => r?.content?.['application/json']?.schema?.$ref?.split('/').pop() || null;

const FALLBACK = (endpoint) => [
  { method: 'post', path: `/${endpoint}`, summary: 'Upload a plan-set PDF (multipart field "file")',
    parameters: [{ name: 'wait', in: 'query', schema: { type: 'number' }, description: 'Seconds (0-600) to hold the request open for the result.' }],
    body: true, responses: [['200', 'The result', endpoint === 'calibration' ? 'Calibration' : 'CrossSections'], ['202', 'Still queued or running', 'JobStatus'], ['400', 'Not a PDF or a bad parameter', 'Problem'], ['413', 'Too large', 'Problem'], ['422', 'Could not be processed', 'Problem']] },
  { method: 'get', path: `/${endpoint}/{job_id}`, summary: 'Get the result of a job',
    parameters: [{ name: 'job_id', in: 'path', required: true, schema: { type: 'string' } },
      ...(endpoint === 'cross-sections' ? [{ name: 'surface', in: 'query', schema: { enum: ['subgrade', 'finished'] } }] : []),
      { name: 'wait', in: 'query', schema: { type: 'number' } }],
    responses: [['200', 'The result', endpoint === 'calibration' ? 'Calibration' : 'CrossSections'], ['202', 'Still queued or running', 'JobStatus'], ['404', 'Unknown or expired id', 'Problem'], ['410', 'PDF no longer on the server', 'Problem'], ['422', 'Could not be processed', 'Problem']] },
];

function operations(spec, endpoint) {
  if (!spec) return null;
  const out = [];
  Object.entries(spec.paths || {}).forEach(([path, ops]) => {
    if (path !== `/${endpoint}` && !path.startsWith(`/${endpoint}/`)) return;
    Object.entries(ops).forEach(([method, op]) => {
      out.push({
        method, path, summary: op.summary, parameters: op.parameters || [], body: Boolean(op.requestBody),
        responses: Object.entries(op.responses || {}).map(([code, r]) => [code, firstSentence(r.description), resRef(r)]),
      });
    });
  });
  return out.sort((a, b) => (a.method === 'post' ? -1 : 1) - (b.method === 'post' ? -1 : 1));
}

function SchemaFields({ spec, name }) {
  const s = spec?.components?.schemas?.[name];
  if (!s?.properties) return null;
  const req = new Set(s.required || []);
  return (
    <details className="api-schema">
      <summary><code>{name}</code> <span className="muted">{Object.keys(s.properties).length} field{Object.keys(s.properties).length === 1 ? '' : 's'}</span></summary>
      <dl>
        {Object.entries(s.properties).map(([k, v]) => (
          <div key={k}>
            <dt><code>{k}</code>{!req.has(k) && <small className="muted"> optional</small>}</dt>
            <dd><span className="api-type">{typeOf(v)}</span>{v.description ? ` — ${firstSentence(v.description)}` : ''}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

function samples({ base, endpoint, apiKey, jobId, surface }) {
  const id = jobId || '3f9c2a1b7e4d';
  const q = endpoint === 'cross-sections' && surface ? surface : null;
  const key = apiKey || null;
  const curlKey = key ? ` \\\n     -H "X-API-Key: ${key}"` : '';
  const noKey = key ? '' : '# Servers started with API_KEYS also need: -H "X-API-Key: <key>"\n';
  const curl = `${noKey}# 1. Upload: 202 + job id, or 200 + the result if ready within ?wait= s
curl -i -X POST "${base}/${endpoint}?wait=60"${curlKey} \\
     -F "file=@plans.pdf"

# 2. Poll until 200 (202 + Retry-After while processing)
curl -i "${base}/${endpoint}/${id}${q ? `?surface=${q}` : ''}"${curlKey}`;

  const python = `import time
import requests

BASE = "${base}"
HEADERS = ${key ? `{"X-API-Key": "${key}"}` : '{}  # {"X-API-Key": "<key>"} when the server has API_KEYS'}

with open("plans.pdf", "rb") as f:
    r = requests.post(f"{BASE}/${endpoint}", params={"wait": 60}, headers=HEADERS,
                      files={"file": ("plans.pdf", f, "application/pdf")})
job_id = r.headers["X-Job-Id"]

while r.status_code == 202:                       # queued or running
    time.sleep(float(r.headers.get("Retry-After", 2)))
    r = requests.get(f"{BASE}/${endpoint}/{job_id}"${q ? `, params={"surface": "${q}"}` : ''}, headers=HEADERS)

r.raise_for_status()                              # 4xx: r.json()["detail"] says why
data = r.json()
${endpoint === 'calibration'
    ? `for c in data["calibrations"]:
    print("page", c["pageNumber"], c["pixelsPerFootX"], "px/ft", c["source"])`
    : `print(data["summary"])
for x in data["crossSections"]:
    print(x["stationId"], x["areas"]["total"]["cutSqFt"], x["areas"]["total"]["fillSqFt"])`}`;

  const js = `const BASE = "${base}";
const headers = ${key ? `{ "X-API-Key": "${key}" }` : '{}; // { "X-API-Key": "<key>" } when the server has API_KEYS'};

const form = new FormData();
form.append("file", fileInput.files[0]);
let res = await fetch(\`\${BASE}/${endpoint}?wait=60\`, { method: "POST", headers, body: form });
const jobId = res.headers.get("X-Job-Id");

while (res.status === 202) {                      // queued or running
  const wait = Number(res.headers.get("Retry-After") || 2) * 1000;
  await new Promise((r) => setTimeout(r, wait));
  res = await fetch(\`\${BASE}/${endpoint}/\${jobId}${q ? `?surface=${q}` : ''}\`, { headers });
}
if (!res.ok) throw new Error((await res.json()).detail);
const data = await res.json();
${endpoint === 'calibration'
    ? 'console.log(data.calibrations.map((c) => [c.pageNumber, c.pixelsPerFootX]));'
    : 'console.log(data.summary, data.alerts);'}`;
  return { curl, python, javascript: js };
}

function Reference({ endpoint, apiKey, jobId, surface }) {
  const [spec, setSpec] = useState(null);
  const [specError, setSpecError] = useState(null);
  const [lang, setLang] = useState(() => stored('api.lang', 'curl'));
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    let live = true;
    getOpenApi().then((s) => live && setSpec(s)).catch((e) => live && setSpecError(e.message));
    return () => { live = false; };
  }, []);
  useEffect(() => store('api.lang', lang), [lang]);

  const base = `${API_ORIGIN}${V1}`;
  const code = useMemo(() => samples({ base, endpoint, apiKey, jobId, surface }), [base, endpoint, apiKey, jobId, surface]);
  const ops = operations(spec, endpoint) || (specError ? FALLBACK(endpoint) : null);
  const resultSchema = endpoint === 'calibration' ? 'Calibration' : 'CrossSections';

  function copy() {
    navigator.clipboard?.writeText(code[lang]).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }, () => {});
  }

  return (
    <section className="card api-ref" aria-labelledby="api-ref-h">
      <div className="api-ref-head">
        <div>
          <span className="eyebrow">Reference</span>
          <h3 id="api-ref-h">{ENDPOINTS[endpoint].label} endpoints</h3>
        </div>
        <span className="spacer" />
        <a className="btn small" href={`${V1}/docs`} target="_blank" rel="noreferrer"><IconExternal width={16} height={16} />Swagger docs</a>
        <a className="btn small" href={`${V1}/openapi.json`} target="_blank" rel="noreferrer">openapi.json</a>
        <a className="btn small" href={`${V1}/schemas/${endpoint}`} target="_blank" rel="noreferrer">JSON Schema</a>
      </div>

      <div className="api-ref-grid">
        <div className="api-ops">
          {specError && <div className="warning" role="status">Could not load the OpenAPI document ({specError}); showing a summary.</div>}
          {!ops && <div className="api-skeleton muted">Loading the OpenAPI document…</div>}
          {ops && ops.map((op) => (
            <details className="api-op" key={op.method + op.path}>
              <summary>
                <span className={`api-method m-${op.method}`}>{op.method.toUpperCase()}</span>
                <code className="api-path">{V1}{op.path}</code>
                <span className="api-op-codes" aria-label="Responses">
                  {op.responses.map(([code]) => <span key={code} className={`api-code c${code[0]}`}>{code}</span>)}
                </span>
                <IconChevron className="chev" width={16} height={16} />
              </summary>
              {op.summary && <p className="api-op-sum">{op.summary}</p>}
              <table className="api-params">
                <caption className="sr-only">Parameters of {op.method.toUpperCase()} {op.path}</caption>
                <thead><tr><th scope="col">Parameter</th><th scope="col">In</th><th scope="col">Type</th></tr></thead>
                <tbody>
                  {op.body && (
                    <tr><td><code>file</code> <small className="req">required</small></td><td>form</td><td><span className="api-type">PDF (multipart/form-data)</span></td></tr>
                  )}
                  {op.parameters.map((p) => (
                    <tr key={p.name} title={p.description || ''}>
                      <td><code>{p.name}</code>{p.required && <small className="req"> required</small>}</td>
                      <td>{p.in}</td>
                      <td><span className="api-type">{typeOf(p.schema)}</span></td>
                    </tr>
                  ))}
                  <tr><td><code>X-API-Key</code></td><td>header</td><td><span className="api-type">string</span> <small className="muted">if the server has API_KEYS</small></td></tr>
                </tbody>
              </table>
              <ul className="api-responses">
                {op.responses.map(([code, desc, ref]) => (
                  <li key={code}>
                    <span className={`api-code c${code[0]}`}>{code}</span>
                    <span className="api-res-desc">{desc}</span>
                    {ref && <code className="api-ref-name">{ref}</code>}
                  </li>
                ))}
              </ul>
            </details>
          ))}
          {spec && (
            <div className="api-schemas">
              <span className="eyebrow">Response schemas</span>
              <SchemaFields spec={spec} name={resultSchema} />
              <SchemaFields spec={spec} name="JobStatus" />
              <SchemaFields spec={spec} name="Problem" />
            </div>
          )}
        </div>

        <div className="api-code-card">
          <div className="api-code-head">
            <div className="segmented small" role="radiogroup" aria-label="Code sample language">
              {[['curl', 'cURL'], ['python', 'Python'], ['javascript', 'JavaScript']].map(([k, label]) => (
                <button type="button" key={k} role="radio" aria-checked={lang === k} className={lang === k ? 'on' : ''} onClick={() => setLang(k)}>{label}</button>
              ))}
            </div>
            <span className="spacer" />
            <button type="button" className="btn small" onClick={copy}><IconCopy width={16} height={16} />{copied ? 'Copied' : 'Copy'}</button>
          </div>
          <pre className="code-block" tabIndex={0} aria-label={`${lang} sample`}><code>{code[lang]}</code></pre>
          {apiKey && <p className="api-note muted">The sample includes the API key you entered; keep it out of shared code.</p>}
        </div>
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- console */
export default function ApiConsole({ openJob }) {
  const [endpoint, setEndpoint] = useState(() => (stored('api.endpoint', 'calibration') === 'cross-sections' ? 'cross-sections' : 'calibration'));
  const [source, setSource] = useState('upload');               // upload | job
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [jobId, setJobId] = useState('');
  const [surface, setSurface] = useState('');
  const [wait, setWait] = useState('0');
  const [apiKey, setApiKey] = useState(() => stored('api.key', ''));
  const [showKey, setShowKey] = useState(false);
  const [run, setRun] = useState({ phase: 'idle' });             // idle | uploading | queued | running | done | error | cancelled
  const [last, setLast] = useState(null);                       // the latest response: {method, url, status, headers, ms}
  const [log, setLog] = useState([]);
  const [result, setResult] = useState(null);                   // {endpoint, jobId, body, surface}
  const [now, setNow] = useState(Date.now());
  const [localError, setLocalError] = useState(null);
  const input = useRef(null);
  const ctrl = useRef(null);

  useEffect(() => store('api.endpoint', endpoint), [endpoint]);
  useEffect(() => store('api.key', apiKey.trim()), [apiKey]);
  useEffect(() => () => ctrl.current?.abort(), []);

  const busy = run.phase === 'uploading' || run.phase === 'queued' || run.phase === 'running';
  useEffect(() => {
    if (!busy) return undefined;
    const t = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(t);
  }, [busy]);

  function pick(files) {
    const f = files && files[0];
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.pdf')) { setLocalError('Please choose a PDF plan set.'); return; }
    setLocalError(null);
    setFile(f);
  }

  async function send(opts = {}) {
    const ep = opts.endpoint || endpoint;
    const src = opts.source || source;
    const id = (opts.jobId ?? jobId).trim();
    const key = apiKey.trim();
    if (src === 'upload' && !file) { setLocalError('Choose a plan-set PDF first.'); return; }
    if (src === 'job' && !ID_RE.test(id)) { setLocalError('Enter a job id (letters, digits, - and _).'); return; }
    setLocalError(null);
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    const started = Date.now();
    setNow(started);
    setResult(null);
    setLast(null);
    setLog([]);
    setRun({ phase: src === 'upload' ? 'uploading' : 'running', started, endpoint: ep, upload: src === 'upload' ? { sent: 0, total: file.size } : null });

    const call = async (method, path, extra = {}) => {
      const res = await v1Request(method, path, { apiKey: key, signal: c.signal, ...extra });
      const entry = { method, url: `${V1}${path}`, status: res.status, ms: res.ms, headers: res.headers };
      setLast(entry);
      setLog((l) => [...l, entry].slice(-12));
      return res;
    };

    try {
      let res;
      let jid = id;
      if (src === 'upload') {
        const form = new FormData();
        form.append('file', file, file.name);
        const w = Math.min(600, Math.max(0, Number(wait) || 0));
        res = await call('POST', `/${ep}${w ? `?wait=${w}` : ''}`, {
          form, onUpload: (sent, total) => setRun((r) => ({ ...r, upload: { sent, total } })),
        });
        jid = res.headers['x-job-id'] || res.body?.id || '';
        // a POST answered 200 is measured to the subgrade; read the surface asked for
        if (res.status === 200 && ep === 'cross-sections' && surface && jid) res = await call('GET', getPath(ep, jid, surface));
      } else {
        res = await call('GET', getPath(ep, jid, surface));
      }
      while (res.status === 202) {
        jid = res.body?.id || res.headers['x-job-id'] || jid;
        const st = { phase: res.body?.status === 'queued' ? 'queued' : 'running', jobId: jid, progress: res.body?.progress };
        setRun((r) => ({ ...r, ...st, polls: (r.polls || 0) + 1 }));
        const ra = parseFloat(res.headers['retry-after']);
        await sleep(ra > 0 ? Math.min(10000, Math.max(500, ra * 1000)) : POLL_MS, c.signal);
        res = await call('GET', getPath(ep, jid, surface));
      }
      jid = res.headers['x-job-id'] || jid;
      if (res.status === 200) {
        setResult({ endpoint: ep, jobId: jid, body: res.body, surface });
        setRun((r) => ({ ...r, phase: 'done', jobId: jid, finished: Date.now() }));
        if (jid) setJobId(jid);
      } else {
        const detail = (res.body && typeof res.body === 'object' && res.body.detail) || (typeof res.body === 'string' && res.body.slice(0, 300)) || 'No detail given.';
        setRun((r) => ({ ...r, phase: 'error', jobId: jid || r.jobId, status: res.status, detail: String(detail), finished: Date.now() }));
      }
    } catch (e) {
      if (e.name === 'AbortError') setRun((r) => ({ ...r, phase: 'cancelled', finished: Date.now() }));
      else setRun((r) => ({ ...r, phase: 'error', status: null, detail: e.message, finished: Date.now() }));
    }
  }

  function cancel() { ctrl.current?.abort(); }

  function readOther() {
    const other = result.endpoint === 'calibration' ? 'cross-sections' : 'calibration';
    setEndpoint(other);
    setSource('job');
    setJobId(result.jobId);
    send({ endpoint: other, source: 'job', jobId: result.jobId });
  }

  const elapsed = run.started ? (run.finished || now) - run.started : 0;
  const pr = run.progress;
  const pct = run.phase === 'uploading' && run.upload?.total ? (run.upload.sent / run.upload.total) * 100
    : pr?.total ? (pr.done / pr.total) * 100 : null;
  const statusText = {
    uploading: run.upload ? `Uploading ${mb(run.upload.sent)} of ${mb(run.upload.total)}` : 'Uploading',
    queued: 'Queued: waiting for a free processing slot',
    running: pr?.total ? `Running: ${pr.done} of ${pr.total} pages read` : 'Running',
    done: 'Done',
    error: run.status ? 'The API answered with an error' : 'Request failed',
    cancelled: 'Cancelled; the job carries on on the server',
  }[run.phase];
  const shownResult = result && result.endpoint === endpoint ? result : null;
  const resultName = shownResult
    ? `${(shownResult.body?.document?.fileName || 'plans').replace(/\.pdf$/i, '')}-${shownResult.endpoint}${shownResult.surface && shownResult.endpoint === 'cross-sections' ? `-${shownResult.surface}` : ''}.json`
    : 'response.json';
  const runLabel = source === 'upload' ? `POST ${V1}/${endpoint}` : `GET ${V1}/${endpoint}/{job_id}`;

  return (
    <div className="page api-page">
      <section className="card api-hero">
        <div className="api-hero-text">
          <span className="eyebrow">Public API · v1</span>
          <h2>Try the API</h2>
          <div className="chips api-flow">
            <span className="api-method m-post">POST</span><span className="muted" aria-hidden="true">→</span>
            <span className="api-code c2">202</span><span className="muted" aria-hidden="true">→</span>
            <span className="api-code c2">200</span><span className="muted" aria-hidden="true">→</span>
            <span className="tag info">JSON</span>
            <InfoTip label="How the API works">Send a plan set to the same endpoints outside clients use, follow it from <code>202</code> to <code>200</code>, and explore the JSON it returns.</InfoTip>
          </div>
        </div>
        <div className="api-tabs" role="tablist" aria-label="Endpoint">
          {Object.entries(ENDPOINTS).map(([k, { label }]) => (
            <button type="button" key={k} role="tab" id={`api-tab-${k}`} aria-selected={endpoint === k} aria-controls="api-panel"
              className={endpoint === k ? 'on' : ''} disabled={busy}
              onClick={() => setEndpoint(k)}
              onKeyDown={(e) => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const n = k === 'calibration' ? 'cross-sections' : 'calibration'; setEndpoint(n); document.getElementById(`api-tab-${n}`)?.focus(); } }}
              tabIndex={endpoint === k ? 0 : -1}>
              {label}
              <small><code>/{k}</code></small>
            </button>
          ))}
        </div>
      </section>

      <div className="api-grid" id="api-panel" role="tabpanel" aria-labelledby={`api-tab-${endpoint}`}>
        <section className="card api-request" aria-labelledby="api-req-h">
          <div className="api-card-head">
            <span className="eyebrow">Request</span>
            <h3 id="api-req-h" className="h-tip">{ENDPOINTS[endpoint].label}
              <InfoTip label={`About the ${ENDPOINTS[endpoint].label.toLowerCase()} endpoint`}>{ENDPOINTS[endpoint].blurb}</InfoTip></h3>
          </div>

          <div className="segmented small api-source" role="radiogroup" aria-label="What to send">
            {[['upload', 'Upload a PDF'], ['job', 'Existing job id']].map(([k, label]) => (
              <button type="button" key={k} role="radio" aria-checked={source === k} className={source === k ? 'on' : ''}
                disabled={busy} onClick={() => { setSource(k); setLocalError(null); }}>{label}</button>
            ))}
          </div>

          <div className="api-endpoint-line">
            <span className={`api-method m-${source === 'upload' ? 'post' : 'get'}`}>{source === 'upload' ? 'POST' : 'GET'}</span>
            <code>{V1}/{endpoint}{source === 'job' ? `/${jobId.trim() || '{job_id}'}` : ''}</code>
          </div>

          {source === 'upload' ? (
            <div className={`dropzone compact ${over ? 'over' : ''} ${file ? 'has-file' : ''} ${busy ? 'busy' : ''}`} role="button" tabIndex={busy ? -1 : 0}
              aria-label={file ? `Plan set ${file.name}; choose another PDF` : 'Choose a plan-set PDF'} aria-disabled={busy}
              onClick={() => !busy && input.current?.click()}
              onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); input.current?.click(); } }}
              onDragOver={(e) => { e.preventDefault(); if (!busy) setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) pick(e.dataTransfer.files); }}>
              <input ref={input} type="file" accept="application/pdf,.pdf" hidden data-testid="api-file"
                onChange={(e) => { pick(e.target.files); e.target.value = ''; }} />
              {file ? (
                <div className="picked"><IconFile /><div><b>{file.name}</b><span className="muted">{mb(file.size)} · click to change</span></div></div>
              ) : (
                <><IconUpload width={28} height={28} /><b>Drop a plan-set PDF here</b><span className="muted">or click to browse</span></>
              )}
            </div>
          ) : (
            <div className="field">
              <label htmlFor="api-job">Job id</label>
              <input id="api-job" className="input mono" value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="3f9c2a1b7e4d"
                autoComplete="off" spellCheck={false} disabled={busy} onKeyDown={(e) => { if (e.key === 'Enter' && !busy) send(); }} />
              {openJob && openJob.job_id !== jobId && (
                <button type="button" className="chip" onClick={() => setJobId(openJob.job_id)} disabled={busy}>
                  Use the open takeoff: {openJob.filename}
                </button>
              )}
              <span className="hint muted">From a POST, or an app upload.</span>
            </div>
          )}

          <div className="api-options">
            {endpoint === 'cross-sections' && (
              <div className="field">
                <label htmlFor="api-surface">Surface</label>
                <select id="api-surface" className="input" value={surface} onChange={(e) => setSurface(e.target.value)} disabled={busy}>
                  <option value="">subgrade (default)</option>
                  <option value="finished">finished grade</option>
                </select>
              </div>
            )}
            {source === 'upload' && (
              <div className="field">
                <label htmlFor="api-wait">Wait <span className="muted">(s, 0-600)</span></label>
                <input id="api-wait" className="input num" type="number" min="0" max="600" step="1" inputMode="numeric"
                  value={wait} onChange={(e) => setWait(e.target.value)} disabled={busy} />
              </div>
            )}
          </div>

          <div className="field">
            <label htmlFor="api-key"><IconKey width={14} height={14} /> API key <span className="muted">(optional)</span></label>
            <div className="input-row">
              <input id="api-key" className="input mono" type={showKey ? 'text' : 'password'} value={apiKey}
                onChange={(e) => setApiKey(e.target.value)} placeholder="Only if the server sets API_KEYS" autoComplete="off" spellCheck={false} />
              <button type="button" className="btn small" onClick={() => setShowKey((s) => !s)} aria-pressed={showKey}>{showKey ? 'Hide' : 'Show'}</button>
              {apiKey && <button type="button" className="btn small ghost" onClick={() => setApiKey('')}>Clear</button>}
            </div>
            <span className="hint muted">Header <code>X-API-Key</code> · saved in this browser</span>
          </div>

          <div className="api-actions">
            <button type="button" className="btn primary api-send" onClick={() => send()}
              disabled={busy || (source === 'upload' ? !file : !jobId.trim())}>
              <IconPlay width={16} height={16} />{busy ? 'Running…' : 'Send request'}
            </button>
            {busy && <button type="button" className="btn" onClick={cancel}>Cancel</button>}
          </div>
          <span className="hint muted api-run-label">{runLabel}{source === 'upload' ? ', then GET until 200' : ''}</span>
          {localError && <div className="error-box" role="alert">{localError}</div>}
        </section>

        <div className="api-output">
          <section className={`card api-status s-${run.phase}`} aria-labelledby="api-status-h">
            <div className="api-status-head">
              <span className="eyebrow" id="api-status-h">Status</span>
              {last && <span className={`api-code c${String(last.status)[0]}`} title="Latest HTTP status">{last.status}</span>}
              <span className="spacer" />
              {run.started && <span className="num muted" aria-label="Elapsed time">{secs(elapsed)}</span>}
            </div>
            <div aria-live="polite" aria-atomic="true" className="api-status-line">
              {run.phase === 'idle'
                ? <span className="muted">Idle · pick a PDF or job id, then <b>Send request</b></span>
                : <b>{statusText}</b>}
              {run.jobId && run.phase !== 'idle' && <span className="muted"> · job <code>{run.jobId}</code></span>}
            </div>
            {busy && <div className="mini-bar" aria-hidden="true"><div className={pct == null ? 'indet' : ''} style={pct == null ? undefined : { width: `${Math.max(3, pct)}%` }} /></div>}
            {busy && run.polls > 0 && <span className="hint muted">Polled {run.polls} time{run.polls > 1 ? 's' : ''}, following Retry-After.</span>}
            {run.phase === 'error' && (
              <div className="error-box api-error" role="alert">
                <IconAlert width={18} height={18} />
                <div>
                  <b>{run.status ? `HTTP ${run.status}` : 'No response'}</b>
                  <span>{run.detail}</span>
                  {run.status === 401 && <span className="muted">This server needs an API key: enter it above.</span>}
                </div>
              </div>
            )}
            {last && (
              <details className="api-headers" open={run.phase !== 'done' || undefined}>
                <summary>Response headers <span className="muted">· {last.method} {last.url} · {last.ms} ms</span></summary>
                <dl>
                  {SHOWN_HEADERS.filter((h) => last.headers[h]).map((h) => (
                    <div key={h}><dt>{h}</dt><dd className="mono">{last.headers[h]}</dd></div>
                  ))}
                </dl>
                {log.length > 1 && (
                  <ol className="api-log">
                    {log.map((e, i) => (
                      <li key={i}><span className={`api-code c${String(e.status)[0]}`}>{e.status}</span><span className="mono">{e.method} {e.url}</span><span className="muted num">{e.ms} ms</span></li>
                    ))}
                  </ol>
                )}
              </details>
            )}
          </section>

          {shownResult ? (
            <>
              <section className="card api-summary" aria-labelledby="api-sum-h">
                <div className="api-summary-head">
                  <div>
                    <span className="eyebrow">Summary</span>
                    <h3 id="api-sum-h">{shownResult.body?.document?.fileName || 'Response'}</h3>
                  </div>
                  <span className="spacer" />
                  <button type="button" className="btn small" onClick={readOther} disabled={busy}>
                    Read its {shownResult.endpoint === 'calibration' ? 'cross-sections' : 'calibration'}
                  </button>
                  <a className="btn small tonal" href={`#/results/${shownResult.jobId}`}>Open takeoff<IconArrow width={16} height={16} /></a>
                </div>
                {shownResult.endpoint === 'calibration'
                  ? <CalibrationSummary body={shownResult.body} />
                  : <CrossSectionSummary body={shownResult.body} />}
              </section>
              <section className="card json-card api-json" aria-label="Response body">
                <JsonTree value={shownResult.body} fileName={resultName} label={`${ENDPOINTS[shownResult.endpoint].label} response`} />
              </section>
            </>
          ) : (
            <section className="card json-card">
              <div className="json-empty">
                <IconCode width={36} height={36} />
                <b>{busy ? statusText : run.phase === 'error' ? 'No result' : 'No response yet'}</b>
                <span className="muted">
                  {busy ? 'The response appears here once the job answers 200.'
                    : `${ENDPOINTS[endpoint].label} JSON appears here`}
                </span>
              </div>
            </section>
          )}
        </div>

        <Reference endpoint={endpoint} apiKey={apiKey.trim()} jobId={result?.jobId || (ID_RE.test(jobId.trim()) ? jobId.trim() : '')} surface={surface} />
      </div>
    </div>
  );
}
