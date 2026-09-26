import { useMemo } from 'react';
import { useAppState } from '../state/AppStateContext.jsx';
import { delayColor, toFiniteNumber } from '../lib/delayScale.js';
import { formatDelay, formatEtaDelta, formatIsoTime, formatSecondsShort } from '../lib/format.js';

export function CascadePanel({ cascade }) {
  const { selectedTrId, selectVehicle } = useAppState();

  const enabled = Boolean(cascade?.enabled);
  const vehicles = useMemo(
    () => (Array.isArray(cascade?.vehicles) ? cascade.vehicles : []),
    [cascade],
  );
  const absorption = useMemo(
    () => (Array.isArray(cascade?.absorption) ? cascade.absorption : []),
    [cascade],
  );

  if (!enabled) {
    return (
      <section className="panel panel--advanced">
        <header className="panel__header">
          <div>
            <h2>Детали каскада</h2>
            <span className="panel__subtitle">Расширенная диагностика</span>
          </div>
        </header>
        <p className="muted">Граф плановых остановок пока не построен.</p>
      </section>
    );
  }

  return (
    <section className="panel panel--advanced">
      <details>
        <summary>
          <span>
            <strong>Детали каскада</strong>
            <small>{cascade?.iterations ?? 0} итераций · {cascade?.transfer_links ?? 0} связей</small>
          </span>
          <span className="details-link">Открыть</span>
        </summary>

        {absorption.length ? (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>ТС</th>
                  <th>Задержка</th>
                  <th>Нагон</th>
                  <th>Догонит</th>
                  <th>Остаток</th>
                </tr>
              </thead>
              <tbody>
                {absorption.map((row) => (
                  <tr
                    key={row?.tr_id}
                    className={String(row?.tr_id) === String(selectedTrId) ? 'is-selected' : ''}
                    onClick={() => selectVehicle(row?.tr_id)}
                  >
                    <td>{row?.tr_id}</td>
                    <td style={{ color: delayColor(row?.delay_s) }}>{formatSecondsShort(row?.delay_s)}</td>
                    <td className="muted">{formatSecondsShort(row?.recoverable_s)}</td>
                    <td>
                      {row?.absorbed ? (
                        <span className="badge badge--ok">да</span>
                      ) : (
                        <span className="badge badge--bad">нет</span>
                      )}
                    </td>
                    <td>{formatSecondsShort(row?.residual_at_end_s)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">Пока нет ТС с прогнозом для каскадного распространения.</p>
        )}

        {vehicles.map((entry) => {
          const stops = Array.isArray(entry?.stops) ? entry.stops : [];
          if (!stops.length) return null;
          const active = String(entry?.tr_id) === String(selectedTrId);
          return (
            <article key={entry?.tr_id} className={`cascade-route ${active ? 'cascade-route--active' : ''}`}>
              <header>
                <button type="button" onClick={() => selectVehicle(entry?.tr_id)}>
                  ТС {entry?.tr_id}
                </button>
                <span className="muted">
                  {entry?.absorption?.absorbed ? 'догонит график' : 'опоздание дойдёт до конца'}
                </span>
              </header>
              <div className="table-scroll">
                <table className="table table--tight">
                  <thead>
                    <tr>
                      <th>Остановка</th>
                      <th>План</th>
                      <th>ETA</th>
                      <th>ΔETA</th>
                      <th>От других</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stops.map((stop) => {
                      const delta = formatEtaDelta(stop?.plan_time, stop?.eta);
                      return (
                        <tr key={`${stop?.stop_id}-${stop?.position}`}>
                          <td title={stop?.address ?? ''}>
                            <span className="truncate">{stop?.address ?? stop?.stop_id}</span>
                          </td>
                          <td className="muted">{formatIsoTime(stop?.plan_time)}</td>
                          <td>{formatIsoTime(stop?.eta)}</td>
                          <td style={{ color: delayColor(toFiniteNumber(stop?.delay_s) ?? 0) }}>{delta}</td>
                          <td className="muted">{formatSecondsShort(stop?.contagion_s)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </article>
          );
        })}
      </details>
    </section>
  );
}

export function CascadeHeadline({ cascade }) {
  if (!cascade?.enabled) return null;
  const maxDelay = toFiniteNumber(cascade?.max_delay_s);
  return (
    <span className="headline" style={{ color: delayColor(maxDelay ?? 0) }}>
      Пик задержки {formatDelay(maxDelay)}
    </span>
  );
}
