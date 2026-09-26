import { useState } from 'react';
import { delayColor, riskLabel, toFiniteNumber } from '../lib/delayScale.js';
import {
  formatDelay,
  formatIsoTime,
  formatPercent,
  formatSpeed,
  formatSecondsShort,
} from '../lib/format.js';

const CAUSE_ROLES = {
  strong_signal: 'подтверждённый сигнал',
  weak_signal: 'слабый сигнал',
  data_quality: 'качество данных',
  none: 'эвристика',
};

const segmentText = (segment) => {
  if (!segment) return '—';
  const stop = (item) => (item ? item.address ?? `остановка ${item.stop_id}` : null);
  const previous = stop(segment.previous);
  const next = stop(segment.next);
  if (previous && next) return `${previous} → ${next}`;
  if (next) return `следующая: ${next}`;
  if (previous) return `после: ${previous}`;
  return '—';
};

export function IncidentCard({ incident }) {
  if (!incident) {
    return (
      <section className="panel panel--incident panel--incident--idle">
        <div className="incident-empty">
          <span className="incident-empty__icon" aria-hidden="true">✓</span>
          <div>
            <h2>Сейчас всё спокойно</h2>
            <p className="muted">Активных инцидентов с прогнозом задержки от 60 секунд нет.</p>
          </div>
        </div>
      </section>
    );
  }

  const snapshot = incident?.snapshot ?? {};
  const delay = toFiniteNumber(incident?.predicted_delay_s);
  const color = delayColor(delay);
  const position = incident?.position_now ?? snapshot?.position ?? null;
  const [detailsOpen, setDetailsOpen] = useState(false);

  return (
    <section className="panel panel--incident" style={{ '--accent': color }}>
      <header className="panel__header panel__header--incident">
        <div>
          <span className="eyebrow">Требует внимания</span>
          <h2>ТС {incident?.tr_id}</h2>
        </div>
        <span className="badge" style={{ background: color }}>
          {riskLabel(incident?.risk)}
        </span>
      </header>

      <div className="incident-main">
        <div>
          <span className="metric__label">Прогноз задержки</span>
          <strong className="incident-delay" style={{ color }}>{formatDelay(delay)}</strong>
          <p className="incident-route">
            Целевая остановка: <strong>{snapshot?.target?.address ?? `остановка ${incident?.target_stop_id}`}</strong>
          </p>
        </div>
        <div className="incident-recommendation">
          <span className="metric__label">Рекомендация оператору</span>
          <strong>{incident?.recommendation ?? 'Дополнительных действий не предложено.'}</strong>
        </div>
      </div>

      <div className="metric-row metric-row--compact">
        <Metric label="Прибытие" value={formatIsoTime(incident?.predicted_arrival)} />
        <Metric label="План" value={formatIsoTime(incident?.target_time)} />
        <Metric label="P опоздания" value={formatPercent(incident?.p_late)} />
      </div>

      <button
        type="button"
        className="details-toggle"
        onClick={() => setDetailsOpen((current) => !current)}
        aria-expanded={detailsOpen}
      >
        {detailsOpen ? 'Скрыть подробности' : 'Показать подробности'}
        <span aria-hidden="true">{detailsOpen ? '⌃' : '⌄'}</span>
      </button>

      {detailsOpen ? (
        <div className="incident-details">
          <dl className="detail-list">
            <Row label="Текущий участок">{segmentText(snapshot?.current_segment)}</Row>
            <Row label="Участок до цели">{segmentText(snapshot?.target_segment)}</Row>
            <Row label="Причина">
              {incident?.cause ?? '—'}{' '}
              <span className="muted">
                ({CAUSE_ROLES[incident?.cause_role ?? snapshot?.cause_role] ?? 'эвристика'})
              </span>
            </Row>
            <Row label="Текущая позиция">
              {position
                ? `${toFiniteNumber(position?.lat)?.toFixed(5) ?? '—'}, ${toFiniteNumber(position?.lon)?.toFixed(5) ?? '—'} · ${formatIsoTime(position?.event_time)} · ${formatSpeed(position?.speed)}`
                : '—'}
            </Row>
          </dl>

          {snapshot?.cur_dev_hint === 'none' ? (
            <p className="callout callout--muted">
              Оценка построена без фактического отклонения cur_dev_s на момент прогноза.
            </p>
          ) : null}

          <PatternEvents events={incident?.pattern_events} />
        </div>
      ) : null}
    </section>
  );
}

export function LiveUnitDetail({ unit }) {
  if (!unit) return null;
  return (
    <div className="live-detail">
      <span>Юнит {unit?.unit_id} → ТС {unit?.tr_id}</span>
      <span style={{ color: delayColor(unit?.prediction) }}>{formatDelay(unit?.prediction)}</span>
      <span className="muted">цель {unit?.target_stop_id ?? '—'}</span>
      <span className="muted">{formatSecondsShort(unit?.age_s)} назад</span>
    </div>
  );
}

function PatternEvents({ events }) {
  if (!Array.isArray(events) || !events.length) return null;
  return (
    <ul className="pattern-list">
      {events.map((event, index) => (
        <li key={`${event?.type}-${index}`}>
          <strong>{event?.type}</strong> ({formatPercent(event?.confidence)}) — {event?.reason}
        </li>
      ))}
    </ul>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className="metric__value">{value}</span>
    </div>
  );
}

function Row({ label, children }) {
  return (
    <div className="detail-list__row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}
