import { useCallback, useEffect, useRef, useState } from 'react';
import { createJob, getAccuracy, getJob, health } from './api';
import Sidebar from './components/Sidebar';
import NavBar from './components/NavBar';
import Dashboard from './components/Dashboard';
import UploadView from './components/UploadView';
import ProgressView from './components/ProgressView';
import Results from './components/Results';
import AccuracyView from './components/AccuracyView';
import CalibrationView from './components/CalibrationView';
import ComparisonView from './components/ComparisonView';
import ApiConsole from './components/ApiConsole';

// Routing lives in the URL hash: #/dashboard, #/new, #/accuracy, #/api and
// #/results/<job id>. The job id in the URL means a reload (or a shared
// link, while the server still holds the job) reopens the same results.
function parseHash() {
  const h = window.location.hash;
  const legacy = h.match(/job=([\w-]+)/);
  if (legacy) return { view: 'results', jobId: legacy[1] };
  const m = h.match(/^#\/(\w+)(?:\/([\w-]+))?/);
  if (!m) return { view: 'dashboard', jobId: null };
  return { view: m[1], jobId: m[2] || null };
}

const TITLES = {
  dashboard: { title: 'Dashboard', sub: 'Accuracy and your latest takeoff' },
  new: { title: 'New takeoff', sub: 'Upload a cross-section plan set' },
  results: { title: 'Takeoff results', sub: 'Cut and fill by road and station' },
  accuracy: { title: 'Accuracy', sub: 'Checked against estimator labels' },
  calibration: { title: 'Calibration', sub: 'Feet-to-pixel scale per sheet, as JSON' },
  comparison: { title: 'Comparison', sub: 'Our takeoff vs provided results' },
  api: { title: 'API', sub: 'Try the public v1 endpoints' },
};

export default function App() {
  const [route, setRoute] = useState(parseHash);
  const [server, setServer] = useState(null);       // null = checking
  const [report, setReport] = useState(undefined);  // undefined = loading, null = unavailable
  const [job, setJob] = useState(null);             // latest job status from the API
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(null);     // null, or {sent, total} bytes while uploading
  const [navOpen, setNavOpen] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    health().then(setServer).catch(() => setServer({ ok: false, unreachable: true }));
    getAccuracy().then(setReport).catch(() => setReport(null));
  }, []);

  useEffect(() => {
    const onHash = () => { setRoute(parseHash()); setNavOpen(false); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const go = useCallback((view, jobId) => {
    window.location.hash = jobId ? `/${view}/${jobId}` : `/${view}`;
  }, []);

  // Poll a job until it is done. A failed check (a dropped connection, a busy
  // or restarting server) is retried with a growing pause rather than ending
  // the polling, so the progress never freezes; only an unknown job stops it.
  const polling = useRef(null);
  const poll = useCallback((id, failures = 0) => {
    clearTimeout(timer.current);
    polling.current = id;
    getJob(id)
      .then((j) => {
        if (polling.current !== id) return;           // another takeoff was opened meanwhile
        setJob(j);
        setError(j.status === 'error' ? (j.error || 'Processing failed.') : null);
        if (j.status === 'queued' || j.status === 'running') {
          timer.current = setTimeout(() => poll(id), j.status === 'queued' ? 2000 : 1000);
        } else {
          polling.current = null;
        }
      })
      .catch((e) => {
        if (polling.current !== id) return;
        if (e.status === 404) {
          polling.current = null;
          setJob(null);
          setError('This takeoff is no longer on the server (results are cleared when the app restarts or '
            + 'after some hours). Upload the plan set again.');
          return;
        }
        setError(failures >= 2 ? 'Reconnecting to the server…' : null);
        timer.current = setTimeout(() => poll(id, failures + 1), Math.min(10000, 1000 * 2 ** failures));
      });
  }, []);

  // open the job named in the URL (on load, or when a link is followed), and
  // pick polling back up when coming back to an unfinished one
  useEffect(() => {
    if (!route.jobId) return;
    const unfinished = job && job.job_id === route.jobId && (job.status === 'queued' || job.status === 'running');
    if (route.jobId !== job?.job_id || (unfinished && polling.current !== route.jobId)) poll(route.jobId);
  }, [route.jobId, job?.job_id, job?.status, poll]);

  useEffect(() => () => clearTimeout(timer.current), []);

  async function upload(file) {
    setError(null);
    setUploading({ sent: 0, total: file.size });
    try {
      const { job_id } = await createJob(file, (sent, total) => setUploading({ sent, total }));
      setJob({ job_id, filename: file.name, status: 'queued', progress: { done: 0, total: 0 } });
      go('results', job_id);
      poll(job_id);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(null);
    }
  }

  const done = job && job.status === 'done' && job.result ? job : null;
  const running = job && (job.status === 'queued' || job.status === 'running') ? job : null;
  const view = TITLES[route.view] ? route.view : 'dashboard';

  let page;
  if (view === 'results') {
    page = done ? <Results key={done.job_id} result={done.result} onResult={(r) => setJob((j) => ({ ...j, result: r }))} />
      : running ? <ProgressView job={running} notice={error} />
      : (
        <div className="page-empty">
          <h2>No takeoff open</h2>
          <p className="muted">{error || 'Upload a plan set to see its cut and fill here.'}</p>
          <button className="btn primary" onClick={() => go('new')}>Start a takeoff</button>
        </div>
      );
  } else if (view === 'new') {
    page = <UploadView server={server} onFile={upload} busy={Boolean(uploading)} progress={uploading} error={error} />;
  } else if (view === 'calibration') {
    page = <CalibrationView openJob={done} />;
  } else if (view === 'api') {
    page = <ApiConsole openJob={done || running} />;
  } else if (view === 'comparison') {
    page = <ComparisonView openJob={done} />;
  } else if (view === 'accuracy') {
    page = <AccuracyView report={report} />;
  } else {
    page = <Dashboard report={report} job={done || running} onOpen={() => go('results', (done || running).job_id)}
      onNew={() => go('new')} onAccuracy={() => go('accuracy')} />;
  }

  return (
    <div className={`app ${navOpen ? 'nav-open' : ''}`}>
      <Sidebar view={view} job={done || running} onNavigate={go} />
      <div className="scrim" onClick={() => setNavOpen(false)} aria-hidden="true" />
      <div className="main">
        <NavBar title={TITLES[view].title} sub={TITLES[view].sub} server={server}
          file={view === 'results' ? (done || running)?.filename : null}
          onMenu={() => setNavOpen((o) => !o)} onNew={() => go('new')} />
        <main className="content" id="content">{page}</main>
      </div>
    </div>
  );
}
