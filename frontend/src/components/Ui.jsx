import { useEffect, useId, useRef, useState } from 'react';
import { IconAlert, IconBook, IconChevron, IconInfo } from './Icons';

/*
 * Small presentational pieces shared by every page, so explanations can live
 * in tooltips and collapsed sections instead of paragraphs.
 */

/**
 * An (i) button whose explanation pops up on hover, focus or tap. The popup
 * is fixed-positioned so cards with overflow: hidden never clip it; on a
 * phone it sits as a sheet at the bottom of the screen.
 */
export function InfoTip({ children, label = 'More information', size = 16 }) {
  const id = useId();
  const btn = useRef(null);
  const [pos, setPos] = useState(null);

  function show() {
    const el = btn.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    const width = Math.min(300, vw - 24);
    const left = Math.max(12, Math.min(r.left + r.width / 2 - width / 2, vw - width - 12));
    const above = r.bottom + 180 > vh && r.top > 180;
    setPos(above ? { left, width, bottom: vh - r.top + 8 } : { left, width, top: r.bottom + 8 });
  }
  const hide = () => setPos(null);

  // follow the button while the page scrolls; close once it leaves the screen
  const open = Boolean(pos);
  useEffect(() => {
    if (!open) return undefined;
    const follow = () => {
      const r = btn.current?.getBoundingClientRect();
      if (!r || r.bottom < 0 || r.top > window.innerHeight) setPos(null);
      else show();
    };
    window.addEventListener('scroll', follow, true);
    window.addEventListener('resize', follow);
    return () => { window.removeEventListener('scroll', follow, true); window.removeEventListener('resize', follow); };
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <span className="info-tip" onMouseEnter={show} onMouseLeave={hide}>
      <button ref={btn} type="button" className="info-btn" aria-label={label} aria-describedby={id} aria-expanded={Boolean(pos)}
        onFocus={show} onBlur={hide}
        onClick={(e) => { e.stopPropagation(); show(); }}
        onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Escape') hide(); }}>
        <IconInfo width={size} height={size} />
      </button>
      <span role="tooltip" id={id} className="info-pop" hidden={!pos} style={pos || undefined}>{children}</span>
    </span>
  );
}

/** A KPI card: icon + short label, a big number, then anything small underneath. */
export function StatCard({ icon: Icon, tone = 'blue', label, value, unit, tip, className = '', children }) {
  return (
    <div className={`stat-card t-${tone} ${className}`}>
      <div className="stat-top">
        {Icon && <span className="stat-icon" aria-hidden="true"><Icon width={18} height={18} /></span>}
        <span className="stat-label">{label}</span>
        {tip && <InfoTip label={`About ${typeof label === 'string' ? label.toLowerCase() : 'this figure'}`}>{tip}</InfoTip>}
      </div>
      {/* big numbers (a 6-digit net volume) step the size down instead of being cut off */}
      <div className={`stat-value num${String(value).length > 9 ? ' xlong' : String(value).length > 6 ? ' long' : ''}`}
        title={typeof value === 'string' || typeof value === 'number' ? `${value}${unit ? ` ${unit}` : ''}` : undefined}>
        {value}{unit && <small>{unit}</small>}
      </div>
      {children}
    </div>
  );
}

/** A thin horizontal meter, 0-100. */
export function Meter({ value, tone = 'blue', label }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <div className={`meter t-${tone}`} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <i style={{ width: `${v}%` }} />
    </div>
  );
}

/**
 * Long warnings as one compact bar: count + the first one cut short; open it
 * to read them all.
 */
export function NoteList({ items, label = 'warning', tone = 'warn', open = false }) {
  if (!items?.length) return null;
  const n = items.length;
  return (
    <details className={`notes ${tone}`} open={open || undefined}>
      <summary>
        <IconAlert width={16} height={16} />
        <b className="num">{n}</b>
        <span className="notes-label">{label}{n === 1 ? '' : 's'}</span>
        <span className="notes-peek">{items[0]}</span>
        <IconChevron className="chev" width={16} height={16} />
      </summary>
      <ul>{items.map((w, i) => <li key={i}>{w}</li>)}</ul>
    </details>
  );
}

/** A collapsed-by-default section ("How it works", "Field reference"). */
export function Disclosure({ title, icon: Icon = IconBook, meta, className = '', children, open = false }) {
  return (
    <details className={`disclosure ${className}`} open={open || undefined}>
      <summary>
        <Icon width={18} height={18} />
        <span className="disclosure-title">{title}</span>
        {meta != null && <span className="disclosure-meta">{meta}</span>}
        <IconChevron className="chev" width={16} height={16} />
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

/** Upload -> Analyze -> Results (or any short list of steps). */
export function Flow({ steps, current, label = 'Progress' }) {
  return (
    <ol className="flow" aria-label={label}>
      {steps.map((s, i) => {
        const state = i < current ? 'done' : i === current ? 'now' : 'todo';
        return (
          <li key={s} className={state} aria-current={state === 'now' ? 'step' : undefined}>
            <span className="flow-dot" aria-hidden="true">
              {state === 'done'
                ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
                : i + 1}
            </span>
            <span className="flow-label">{s}</span>
            {state === 'done' && <span className="sr-only">(done)</span>}
          </li>
        );
      })}
    </ol>
  );
}
