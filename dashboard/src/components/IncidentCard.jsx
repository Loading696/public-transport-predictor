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
        <h2>Карточка инцидента</h2>
        <p className="muted">Активных инцидентов с прогнозом задержки от 60 секунд нет.</p>
      </section>
    );
  }

  const snapshot = incident?.snapshot ?? {};
  const delay = toFiniteNumber(incident?.predicted_delay_s);
  const color = delayColor(delay);
  const position = incident?.position_now ?? snapshot?.position ?? null;

  return (
    <section className="panel panel--incident" style={{ '--accent': color }}>
      <header className="panel__header">
        <h2>Инцидент · ТС {incident?.tr_id}</h2>
        <span className="badge" style={{ background: color }}>
          {riskLabel(incident?.risk)}
        </span>
      </header>

      <div className="metric-row">
        <Metric label="Прогноз задержки" value={formatDelay(delay)} color={color} />
        <Metric label="Прогноз прибытия" value={formatIsoTime(incident?.predicted_arrival)} />
        <Metric label="Плановое прибытие" value={formatIsoTime(incident?.target_time)} />
      </div>

      <dl className="detail-list">
        <Row label="Целевая остановка">
          {snapshot?.target?.address ?? `остановка ${incident?.target_stop_id}`} ·{' '}
          {formatIsoTime(incident?.target_time)}
        </Row>
        <Row label="Текущий участок">{segmentText(snapshot?.current_segment)}</Row>
        <Row label="Участок до цели">{segmentText(snapshot?.target_segment)}</Row>
        <Row label="Предполагаемая причина">
          {incident?.cause ?? '—'}
          <span className="muted">
            {' '}
            ({CAUSE_ROLES[incident?.cause_role ?? snapshot?.cause_role] ?? 'эвристика'})
          </span>
        </Row>
        <Row label="Текущая позиция">
          {position
            ? `${toFiniteNumber(position?.lat)?.toFixed(5) ?? '—'}, ${toFiniteNumber(position?.lon)?.toFixed(5) ?? '—'} · ${formatIsoTime(position?.event_time)} · ${formatSpeed(position?.speed)}`
            : '—'}
        </Row>
        <Row label="Рекомендация">{incident?.recommendation ?? '—'}</Row>
        <Row label="P(опоздание &gt; 120 с)">{formatPercent(incident?.p_late)}</Row>
      </dl>

      {snapshot?.cur_dev_hint === 'none' ? (
        <p className="callout callout--muted">
          Оценка построена без подсказки cur_dev_s — фактического отклонения на момент
          прогноза нет.
        </p>
      ) : null}

      <PatternEvents events={incident?.pattern_events} />
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

function Metric({ label, value, color }) {
  return (
    <div className="metric">
      <span className="metric__label">{label}</span>
      <span className="metric__value" style={color ? { color } : undefined}>
        {value}
      </span>
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
