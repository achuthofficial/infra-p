import { useEffect, useMemo, useRef, useState } from 'react';
import { API_ORIGIN, createJob, getCalibration, getCrossSections, getJob, getSchema } from '../api';
import { IconCopy, IconDownload, IconFile, IconRuler, IconUpload } from './Icons';
import JsonView, { pretty } from './JsonView';
import { Disclosure, InfoTip, Meter } from './Ui';

const SHOW_SHAPES = 40;           // shapes rendered in the viewer; Copy / Download carry all of them
const mb = (b) => `${(b / 1048576).toFixed(b < 10485760 ? 1 : 0)} MB`;
const fmt = (v, d = 2) => Number(v).toFixed(d);
const SOURCE = {
  grid: 'grid', sheet_note: 'sheet note', document_note: 'document note',
};
const SOURCE_TIP = {
  grid: 'Measured from the cross-section grid on the sheet', sheet_note: 'Read from the scale note printed on the sheet',
  document_note: 'The document\u2019s scale note (often on the first page)',
};
const SOURCE_TONE = { grid: 'ok', sheet_note: 'info', document_note: 'neutral' };

function codeSamples(origin) {
  return {
    curl: `# 1. Upload the plan set (large files: the chunked /api/uploads flow the app uses)
curl -F "file=@plans.pdf" ${origin}/api/jobs
# -> {"job_id": "3f2c..."}

# 2. Wait until "status" is "done" (progress.done / progress.total = pages)
curl ${origin}/api/jobs/3f2c...

# 3. Fetch the calibration
curl ${origin}/api/jobs/3f2c.../calibration -o calibration.json

# 4. Every cross-section: left / right cut and fill, shapes, confidence, alerts
curl ${origin}/api/jobs/3f2c.../cross-sections -o cross-sections.json

# The JSON Schemas both responses are validated against
curl ${origin}/api/schemas/calibration
curl ${origin}/api/schemas/cross-sections`,
    python: `import time, requests

BASE = "${origin}"

with open("plans.pdf", "rb") as f:
    job = requests.post(f"{BASE}/api/jobs", files={"file": f}).json()["job_id"]

while requests.get(f"{BASE}/api/jobs/{job}").json()["status"] not in ("done", "error"):
    time.sleep(2)

cal = requests.get(f"{BASE}/api/jobs/{job}/calibration").json()
by_page = {c["pageNumber"]: c for c in cal["calibrations"]}

def to_feet(page, dx_px, dy_px):
    """A distance measured on a page, in pixels, as feet."""
    c = by_page[page]
    return dx_px * c["scaleX"], dy_px * c["scaleY"]

for s in cal["shapes"]:
    ox, oy = s["position2D"]["x"], s["position2D"]["y"]
    xs, ys = s["points"][0::2], s["points"][1::2]
    outline = [(ox + x, oy + y) for x, y in zip(xs, ys)]      # page pixels
    print(s["stationId"], s["side"], s["crossSectionType"], s["areaSqFt"], "sq ft")

xs = requests.get(f"{BASE}/api/jobs/{job}/cross-sections").json()
for x in xs["crossSections"]:
    a = x["areas"]
    print(x["stationId"], "left cut", a["left"]["cutSqFt"], "right fill", a["right"]["fillSqFt"],
          "confidence", x["confidence"]["score"])
for alert in xs["alerts"]:                                   # sections below 85 %
    print("CHECK", alert["stationId"], alert["message"], alert["reasons"])`,
    javascript: `const BASE = "${origin}";

const form = new FormData();
form.append("file", fileInput.files[0]);
const { job_id } = await (await fetch(\`\${BASE}/api/jobs\`, { method: "POST", body: form })).json();

let job;
do {
  await new Promise((r) => setTimeout(r, 2000));
  job = await (await fetch(\`\${BASE}/api/jobs/\${job_id}\`)).json();
} while (job.status === "queued" || job.status === "running");

const cal = await (await fetch(\`\${BASE}/api/jobs/\${job_id}/calibration\`)).json();

// pixels -> feet on a page
const page = Object.fromEntries(cal.calibrations.map((c) => [c.pageNumber, c]));
const feet = (pageNo, px) => px * page[pageNo].scaleX;      // or px / page[pageNo].pixelsPerFootX

const xs = await (await fetch(\`\${BASE}/api/jobs/\${job_id}/cross-sections\`)).json();
const lowConfidence = xs.crossSections.filter((x) => !x.confidence.passed);

// draw the shapes over the page with Konva (page drawn at 100 %, 1 px = 1 pt)
for (const s of cal.shapes.filter((s) => s.pageNumber === pageNo)) {
  layer.add(new Konva.Line({
    x: s.position2D.x, y: s.position2D.y, points: s.points, closed: true,
    fill: s.crossSectionType === "cut" ? "rgba(217,48,37,.4)" : "rgba(30,142,62,.4)",
  }));
}`,
  };
}

const FIELDS = [
  ['calibrations[]', 'One per sheet: pixelLengthX/Y page pixels = realWorldLengthX/Y feet; scaleX/Y = feet per pixel.'],
  ['pixelsPerFootX/Y', 'The same scale per axis as pixels per foot; verticalDiffers when the vertical is exaggerated.'],
  ['gridScale', 'Feet per major grid square on each axis, read off the section\u2019s axis labels (offsets across, elevations up), with how it was read.'],
  ['source', 'grid (measured on the sheet), sheet_note (its printed scale) or document_note (the document’s scale, e.g. on the first page).'],
  ['noteCheck', 'Where a sheet has both, the printed scale as a check on the measured grid.'],
  ['shapes[]', 'Each cut / fill area: points are x, y offsets from position2D, its top-left on the page.'],
  ['side', 'left or right of the section’s centreline (offset 0).'],
  ['pixels', 'PDF points of the page as displayed: a PDF viewer at 100 % zoom (1" = 72 px).'],
  ['crossSections[]', 'GET …/cross-sections: per section its station, street, stage, left / right / total cut and fill (sq ft), its shapes, confidence and alerts.'],
  ['confidence', '0–100 with every deduction listed; 85 or more passes. Below 85 the section gets an alert, also gathered in alerts[].'],
];

const SCALE_SRC = { labels: 'labels', label_widths: 'label widths', ocr: 'OCR', assumed: 'assumed' };

/**
 * Calibration: upload a plan set, analyze it, and get the feet-to-pixel
 * calibration of every sheet (and its cut / fill shapes) as JSON, with how
 * to use it from code.
 */
export default function CalibrationView({ openJob }) {
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState('idle');        // idle | uploading | processing | calibrating | done | error
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [part, setPart] = useState('calibrations');
  const [lang, setLang] = useState('curl');
  const [copied, setCopied] = useState(null);
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const alive = useRef(true);
  // set on every mount: StrictMode mounts twice in development, and a flag
  // only ever cleared would stop the polling after its first answer
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const samples = useMemo(() => codeSamples(API_ORIGIN), []);
  const busy = phase === 'uploading' || phase === 'processing' || phase === 'calibrating';

  async function calibrate(jobId) {
    setPhase('calibrating');
    const cal = await getCalibration(jobId);
    const xs = await getCrossSections(jobId).catch(() => null);
    if (!alive.current) return;
    setData({ jobId, ...cal, xs });
    setPhase('done');
  }

  async function waitFor(jobId) {
    setPhase('processing');
    for (;;) {
      const j = await getJob(jobId);
      if (!alive.current) return;
      setProgress(j.progress);
      if (j.status === 'done') return calibrate(jobId);
      if (j.status === 'error') throw new Error(j.error || 'Processing failed.');
      await new Promise((r) => setTimeout(r, 900));
    }
  }

  async function analyze() {
    if (!file) return;
    setError(null);
    setData(null);
    try {
      setPhase('uploading');
      setProgress({ sent: 0, total: file.size });
      const { job_id: jobId } = await createJob(file, (sent, total) => setProgress({ sent, total }));
      await waitFor(jobId);
    } catch (e) {
      setError(e.message);
      setPhase('error');
    }
  }

  async function useOpen() {
    setError(null);
    setData(null);
    setFile(null);
    try { await calibrate(openJob.job_id); } catch (e) { setError(e.message); setPhase('error'); }
  }

  function pick(files) {
    const f = files && files[0];
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.pdf')) { setError('Please choose a PDF plan set.'); return; }
    setError(null);
    setFile(f);
    setPhase('idle');
  }

  const [schema, setSchema] = useState(null);
  useEffect(() => {
    if (part === 'schema' && !schema) getSchema('cross-sections').then(setSchema).catch((e) => setError(e.message));
  }, [part, schema]);

  const shown = useMemo(() => {
    if (part === 'schema') return schema ? { value: schema } : null;
    if (!data) return null;
    const { jobId, xs, ...body } = data;    // eslint-disable-line no-unused-vars
    if (part === 'calibrations') return { value: body.calibrations };
    if (part === 'sections' || part === 'alerts') {
      if (!xs) return { value: { error: 'cross-sections unavailable' } };
      if (part === 'alerts') return { value: { passScore: xs.passScore, summary: xs.summary, alerts: xs.alerts } };
      const few = xs.crossSections.slice(0, 12);
      const more = xs.crossSections.length - few.length;
      return { value: { ...xs, crossSections: few, shapes: `${xs.shapes.length} shapes (also inside each cross-section)` },
        note: more > 0 ? `${more} more cross-sections not shown here; Copy or Download includes all ${xs.crossSections.length}` : null };
    }
    const shapes = body.shapes.slice(0, SHOW_SHAPES);
    const more = body.shapes.length - shapes.length;
    const note = more > 0 ? `${more} more shapes not shown here; Copy or Download includes all ${body.shapes.length}` : null;
    if (part === 'shapes') return { value: shapes, note };
    return { value: { ...body, shapes }, note };
  }, [data, part]);

  const fullText = useMemo(() => {
    if (part === 'schema') return schema ? pretty(schema) : '';
    if (!data) return '';
    const { jobId, xs, ...body } = data;    // eslint-disable-line no-unused-vars
    const value = part === 'calibrations' ? body.calibrations : part === 'shapes' ? body.shapes
      : part === 'sections' ? xs : part === 'alerts' ? xs && { passScore: xs.passScore, summary: xs.summary, alerts: xs.alerts }
      : body;
    return pretty(value);
  }, [data, part, schema]);

  function copy(text, what) {
    navigator.clipboard?.writeText(text).then(() => { setCopied(what); setTimeout(() => setCopied(null), 1500); });
  }

  function download() {
    const blob = new Blob([fullText], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const name = { full: 'calibration', sections: 'cross-sections', schema: 'cross-sections.schema' }[part] || part;
    a.download = `${(data?.document?.fileName || 'plans').replace(/\.pdf$/i, '')}-${name}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const pct = phase === 'uploading' && progress?.total ? (progress.sent / progress.total) * 100
    : phase === 'processing' && progress?.total ? (progress.done / progress.total) * 100 : null;
  const statusText = phase === 'uploading' ? `Uploading ${progress ? `${mb(progress.sent)} of ${mb(progress.total)}` : ''}`
    : phase === 'processing' ? `Reading sheets ${progress?.total ? `${progress.done} of ${progress.total}` : '…'}`
    : phase === 'calibrating' ? 'Calibrating every sheet…' : '';

  return (
    <div className="page calib-page">
      <div className="calib-grid">
        <section className="card calib-input">
          <span className="eyebrow">Calibration module</span>
          <h2>Feet to pixels, sheet by sheet</h2>
          <div className="chips">
            <span className="muted small-text">Scale from</span>
            {Object.keys(SOURCE).map((k) => <span key={k} className={`tag ${SOURCE_TONE[k]}`} title={SOURCE_TIP[k]}>{SOURCE[k]}</span>)}
            <InfoTip label="Where each scale comes from">
              Upload the whole plan set. Each sheet&apos;s scale is measured from its cross-section grid, or read from its
              printed scale note, or from the document&apos;s scale (often on the first page).
            </InfoTip>
          </div>

          <div className={`dropzone compact ${over ? 'over' : ''} ${file ? 'has-file' : ''}`} role="button" tabIndex={0}
            aria-label="Choose a plan set PDF"
            onClick={() => !busy && input.current?.click()}
            onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); input.current?.click(); } }}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); if (!busy) pick(e.dataTransfer.files); }}>
            <input ref={input} type="file" accept="application/pdf,.pdf" hidden onChange={(e) => pick(e.target.files)} />
            {file ? (
              <div className="picked"><IconFile /><div><b>{file.name}</b><span className="muted">{mb(file.size)} · click to change</span></div></div>
            ) : (
              <><IconUpload width={28} height={28} /><b>Drop the plan set PDF here</b><span className="muted">or click to browse</span></>
            )}
          </div>

          <button className="btn primary analyze" onClick={analyze} disabled={!file || busy}>
            <IconRuler width={18} height={18} />{busy ? 'Analyzing…' : 'Analyze'}
          </button>
          {openJob && (
            <button className="btn ghost small use-open" onClick={useOpen} disabled={busy}>
              <span className="ellipsis">or use the open takeoff · {openJob.filename}</span>
            </button>
          )}

          {busy && (
            <div className="calib-progress" role="status">
              <span>{statusText}</span>
              <div className="mini-bar"><div style={{ width: `${pct ?? 8}%` }} /></div>
            </div>
          )}
          {error && <div className="warning">{error}</div>}

          {data && (
            <div className="calib-summary">
              <div className="calib-stats">
                <div><b className="num">{data.calibrations.length}</b><span>sheets calibrated</span></div>
                <div><b className="num">{data.document.crossSectionPages}</b><span>cross-section sheets</span></div>
                <div><b className="num">{data.shapes.length}</b><span>cut / fill shapes</span></div>
              </div>
              {data.xs && (
                <div className="calib-stats conf-stats">
                  <div><b className="num">{data.xs.summary.crossSections}</b><span>cross-sections</span></div>
                  <div><b className="num fill-text">{data.xs.summary.passed}</b><span>pass ≥ {data.xs.passScore}%</span></div>
                  <div><b className={`num ${data.xs.summary.alerts ? 'cut-text' : ''}`}>{data.xs.summary.alerts}</b><span>alerts</span></div>
                </div>
              )}
              {data.xs && data.xs.summary.crossSections > 0 && (
                <Meter value={(100 * data.xs.summary.passed) / data.xs.summary.crossSections} tone={data.xs.summary.alerts ? 'yellow' : 'green'}
                  label={`${data.xs.summary.passed} of ${data.xs.summary.crossSections} cross-sections pass`} />
              )}
              <div className="calib-scale chips">
                <span className="muted small-text">Document scale</span>
                <span className={`tag ${data.documentScale ? 'info' : 'neutral'}`}>
                  {data.documentScale ? `${data.documentScale.text} · p. ${data.documentScale.pageNumber}` : 'none printed'}</span>
                {data.printScale && data.printScale.ratio !== 1 && (
                  <>
                    <span className="tag warn" title={data.printScale.meaning}>
                      printed {data.printScale.ratio === 0.5 ? 'half size' : `at ${Math.round(data.printScale.ratio * 100)} %`}</span>
                    <InfoTip label="About the print scale">
                      {data.printScale.meaning} The grid is measured, so areas are right.
                    </InfoTip>
                  </>
                )}
              </div>
              {data.uncalibratedPages.length > 0 && (
                <div className="warning">No scale found for sheet{data.uncalibratedPages.length > 1 ? 's' : ''} {data.uncalibratedPages.join(', ')}.</div>
              )}
              <div className="calib-table-wrap">
                <table className="calib-table">
                  <thead><tr><th>Sheet</th><th>Source</th><th className="num">px / ft (X)</th><th className="num">px / ft (Y)</th><th>Grid square</th><th className="num">ft / px</th></tr></thead>
                  <tbody>
                    {data.calibrations.map((c) => (
                      <tr key={c.id}>
                        <td className="num">{c.pageNumber}</td>
                        <td><span className={`tag ${SOURCE_TONE[c.source] || 'warn'}`} title={SOURCE_TIP[c.source]}>{SOURCE[c.source] || c.source}</span></td>
                        <td className="num">{fmt(c.pixelsPerFootX ?? c.pixelLengthX / 10, 3)}</td>
                        <td className={`num ${c.verticalDiffers ? 'warn-text' : ''}`}>{fmt(c.pixelsPerFootY ?? c.pixelLengthY / 10, 3)}</td>
                        <td title={c.gridScale ? `across: ${SCALE_SRC[c.gridScale.sourceX]}, up: ${SCALE_SRC[c.gridScale.sourceY]}` : ''}>
                          {c.gridScale ? `${c.gridScale.ftPerSquareX} × ${c.gridScale.ftPerSquareY} ft` : '—'}
                          {c.gridScale && c.gridScale.sourceX === 'assumed' && <span className="tag warn" style={{ marginLeft: 6 }}>assumed</span>}
                        </td>
                        <td className="num">{fmt(c.scaleX, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        <div className="calib-output">
          <section className="card json-card">
            <div className="json-head">
              <div>
                <span className="eyebrow">{part === 'schema' ? 'GET /api/schemas/cross-sections' : `Response${data ? ` · GET /api/jobs/${data.jobId}/${part === 'sections' || part === 'alerts' ? 'cross-sections' : 'calibration'}` : ''}`}</span>
                <h3>{part === 'sections' || part === 'alerts' ? 'Cross-section JSON' : part === 'schema' ? 'JSON Schema' : 'Calibration JSON'}</h3>
              </div>
              <span className="spacer" />
              <div className="segmented small" role="radiogroup" aria-label="Part of the response">
                {[['calibrations', 'Calibrations'], ['shapes', 'Shapes'], ['full', 'Full'], ['sections', 'Cross-sections'],
                  ['alerts', 'Alerts'], ['schema', 'Schema']].map(([k, label]) => (
                  <button key={k} role="radio" aria-checked={part === k} className={part === k ? 'on' : ''} onClick={() => setPart(k)}
                    disabled={!data && k !== 'schema'}>{label}</button>
                ))}
              </div>
              <button className="btn small" onClick={() => copy(fullText, 'json')} disabled={!fullText}>
                <IconCopy width={16} height={16} />{copied === 'json' ? 'Copied' : 'Copy'}</button>
              <button className="btn small" onClick={download} disabled={!fullText}>
                <IconDownload width={16} height={16} />Download</button>
            </div>
            {shown ? <JsonView value={shown.value} note={shown.note} /> : (
              <div className="json-empty">
                <IconRuler width={36} height={36} />
                <b>{busy ? statusText : 'No response yet'}</b>
                <span className="muted">Upload a plan set, then Analyze.</span>
              </div>
            )}
          </section>

          <section className="card code-card">
            <div className="json-head">
              <div>
                <span className="eyebrow">Use it from code</span>
                <h3>How to call the calibration API</h3>
              </div>
              <span className="spacer" />
              <div className="segmented small" role="radiogroup" aria-label="Language">
                {[['curl', 'cURL'], ['python', 'Python'], ['javascript', 'JavaScript']].map(([k, label]) => (
                  <button key={k} role="radio" aria-checked={lang === k} className={lang === k ? 'on' : ''} onClick={() => setLang(k)}>{label}</button>
                ))}
              </div>
              <button className="btn small" onClick={() => copy(samples[lang], 'code')}>
                <IconCopy width={16} height={16} />{copied === 'code' ? 'Copied' : 'Copy'}</button>
            </div>
            <pre className="code-block"><code>{samples[lang]}</code></pre>
            <Disclosure title="Field reference" meta={`${FIELDS.length} fields`} className="flush">
              <dl className="field-guide">
                {FIELDS.map(([k, v]) => <div key={k}><dt><code>{k}</code></dt><dd>{v}</dd></div>)}
              </dl>
            </Disclosure>
          </section>
        </div>
      </div>
    </div>
  );
}
