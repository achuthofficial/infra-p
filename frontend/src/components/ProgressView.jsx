import { useEffect, useState } from 'react';
import { IconClock, IconLayers, IconScan, IconSection, IconSheet } from './Icons';
import { Flow } from './Ui';

const STEPS = [
  { Icon: IconScan, label: 'Find frames', tip: 'Finding cross-section frames on every page' },
  { Icon: IconSheet, label: 'Read stations', tip: 'Reading station labels (PDF text, or OCR on scans)' },
  { Icon: IconSection, label: 'Trace surfaces', tip: 'Tracing existing ground and proposed design' },
  { Icon: IconLayers, label: 'Measure', tip: 'Measuring cut / fill and grouping stations into roads' },
];

export default function ProgressView({ job, notice }) {
  const [start] = useState(() => Date.now());
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const { done, total } = job.progress || { done: 0, total: 0 };
  const pct = total ? Math.round((done / total) * 100) : 0;
  const secs = Math.round((now - start) / 1000);

  return (
    <div className="page narrow">
      <div className="card progress-card" aria-live="polite">
        <Flow steps={['Upload', 'Analyze', 'Results']} current={1} />
        <div className="progress-head">
          <div className="progress-title">
            <span className="eyebrow">{job.status === 'queued' ? 'Queued' : 'Reading'}</span>
            <h2 className="ellipsis" title={job.filename}>{job.filename}</h2>
          </div>
          <div className="progress-pct num">{total ? `${pct}%` : '…'}</div>
        </div>
        <div className={`bar ${total ? '' : 'indeterminate'}`}>
          <div style={{ width: `${pct}%` }} />
        </div>
        <div className="chips">
          <span className="tag neutral"><IconSheet width={14} height={14} />
            {job.status === 'queued'
              ? (job.queue_position ? `#${job.queue_position} in line` : 'waiting for a free slot')
              : total ? `${done} / ${total} pages` : 'starting'}</span>
          <span className="tag neutral"><IconClock width={14} height={14} />{secs}s</span>
        </div>
        {notice && <div className="note warning-note" role="status">{notice}</div>}
        <ul className="steps-row" aria-label="What happens">
          {STEPS.map(({ Icon, label, tip }) => (
            <li key={label} title={tip}><Icon width={18} height={18} /><span>{label}</span><span className="sr-only">: {tip}</span></li>
          ))}
        </ul>
      </div>
    </div>
  );
}
