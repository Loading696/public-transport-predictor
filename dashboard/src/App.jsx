import { useMemo } from 'react';
import { config } from './config.js';
import { useAppState } from './state/AppStateContext.jsx';
import { useConnection } from './state/DataProvider.jsx';
import { useStreamStatus, useCascadeSnapshot } from './api/queries.js';
import { TransportMap } from './components/TransportMap.jsx';
import { KpiSidebar } from './components/KpiSidebar.jsx';
import { CascadeHeadline, CascadePanel } from './components/CascadePanel.jsx';
import { IncidentCard } from './components/IncidentCard.jsx';
import { delayColor, toFiniteNumber } from './lib/delayScale.js';
import { formatClock, formatIsoTime, formatPercent } from './lib/format.js';

const PROVIDER_STATUS_LABEL = { '2gis': '2ГИС Карты', yandex: 'Яндекс Карты', google: 'Google Maps', none: 'нет ключа' };

export default function App() {
  const { sources, source, setSource, speed, setSpeed, resetToken, restartStream } = useAppState();
  const connection = useConnection();
  const providerStatus = config.providerStatus();

  const query = useStreamStatus({ speed, resetToken });

  const status = query.data;
  const cascade = status?.map?.cascade ?? { enabled: false };
  const incident = status?.map?.incident ?? null;

  useCascadeSnapshot(cascade, Boolean(status));

  const lastUpdatedAt = useMemo(
    () => (query.dataUpdatedAt ? formatClock(new Date(query.dataUpdatedAt)) : '—'),
    [query.dataUpdatedAt],
  );

  const errorMessage = query.isError ? query.error?.message ?? 'неизвестная ошибка' : null;

  const worstDelay = useMemo(() => {
    const list = status?.vehicles ?? [];
    const values = list.map((v) => toFiniteNumber(v?.prediction)).filter((v) => v !== null);
    return values.length ? Math.max(...values) : null;
  }, [status]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <h1>Предиктор изменений графика движения</h1>
          <span className="muted">
            Каскадная модель · {PROVIDER_STATUS_LABEL[providerStatus.active]} · обновлено{' '}
            {lastUpdatedAt}
          </span>
        </div>

        <div className="topbar__controls">
          <div className="segmented" role="group" aria-label="Источник данных">
            {sources.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === source ? 'is-active' : ''}
                onClick={() => setSource(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <label className="slider">
            <span>Скорость ×{speed}</span>
            <input
              type="range"
              min={1}
              max={600}
              step={10}
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </label>

          <button type="button" className="ghost" onClick={restartStream}>
            Перезапустить поток
          </button>

          <span
            className={`chip ${connection.connected ? 'chip--ok' : 'chip--muted'}`}
            title={connection.mode === 'websocket' ? 'WebSocket' : `polling ${config.pollIntervalMs} мс`}
          >
            {connection.mode === 'websocket'
              ? connection.connected
                ? 'WS'
                : 'WS: переподключение'
              : 'polling'}
          </span>
        </div>
      </header>

      {errorMessage ? (
        <div className="banner banner--error">
          <strong>Бэкенд недоступен.</strong> {errorMessage} — интерфейс продолжает показывать
          последние полученные данные.
        </div>
      ) : null}

      {cascade?.enabled ? (
        <div className="banner banner--info">
          <CascadeHeadline cascade={cascade} />
          <span>
            {cascade.injected_routes?.length ?? 0} ТС в каскаде, задеты соседними:{' '}
            {cascade.contaminated_routes?.length ?? 0}, догонят:{' '}
            {cascade.absorbed_ratio === null || cascade.absorbed_ratio === undefined
              ? '—'
              : formatPercent(cascade.absorbed_ratio)}
          </span>
        </div>
      ) : null}

      <main className="layout">
        <KpiSidebar status={status} isFetching={query.isFetching} lastUpdatedAt={lastUpdatedAt}>
          <CascadePanel cascade={cascade} />
        </KpiSidebar>

        <section className="content">
          <div className="content__map">
            <TransportMap
              status={status}
              isFetching={query.isFetching}
              error={errorMessage}
              lastUpdatedAt={lastUpdatedAt}
            />
            <Legend worstDelay={worstDelay} simulatedTime={status?.simulated_time} />
          </div>

          <div className="content__incident">
            <IncidentCard incident={incident} />
          </div>
        </section>
      </main>
    </div>
  );
}

function Legend({ worstDelay, simulatedTime }) {
  return (
    <div className="legend">
      <span className="legend__scale" aria-hidden />
      <div className="legend__labels">
        <span>в графике</span>
        <span>120 с</span>
        <span>300 с+</span>
      </div>
      <div className="legend__meta">
        <span>время симуляции {formatIsoTime(simulatedTime)}</span>
        {worstDelay !== null ? (
          <span style={{ color: delayColor(worstDelay) }}>худший прогноз +{worstDelay.toFixed(0)} с</span>
        ) : null}
      </div>
    </div>
  );
}
