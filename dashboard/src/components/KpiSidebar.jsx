import { useMemo } from 'react';
import { useAppState } from '../state/AppStateContext.jsx';
import { delayColor, isRedZone, isOnTime, riskLabel, toFiniteNumber } from '../lib/delayScale.js';
import {
  formatDelay,
  formatInteger,
  formatPercent,
  formatRatio,
} from '../lib/format.js';

export function computeKpis(status) {
  const vehicles = Array.isArray(status?.vehicles) ? status.vehicles : [];
  const liveUnits = Array.isArray(status?.live_units) ? status.live_units : [];
  const cascade = status?.map?.cascade ?? { enabled: false };

  const delays = vehicles
    .map((vehicle) => toFiniteNumber(vehicle?.prediction))
    .filter((value) => value !== null);

  const meanDelay = delays.length
    ? delays.reduce((sum, value) => sum + value, 0) / delays.length
    : null;

  const worstDelay = delays.length ? Math.max(...delays) : null;
  const redZone = vehicles.filter(isRedZone).length;
  const onTime = vehicles.filter(isOnTime).length;

  const liveDelays = liveUnits
    .map((unit) => toFiniteNumber(unit?.prediction))
    .filter((value) => value !== null);

  const contaminated = cascade.enabled ? (cascade.contaminated_routes ?? []).length : null;
  const absorbedRatio = cascade.enabled ? toFiniteNumber(cascade.absorbed_ratio) : null;
  const transfersAffected = cascade.enabled
    ? (cascade.contaminated_routes ?? []).filter(
        (trId) => !(cascade.injected_routes ?? []).map(String).includes(String(trId)),
      ).length
    : null;

  const routeCount = new Set(vehicles.map((vehicle) => vehicle?.tr_id)).size;

  return {
    meanDelay,
    worstDelay,
    redZone,
    onTime,
    totalVehicles: vehicles.length,
    routeCount,
    liveCount: liveUnits.length,
    liveMeanDelay: liveDelays.length
      ? liveDelays.reduce((sum, value) => sum + value, 0) / liveDelays.length
      : null,
    contaminated,
    absorbedRatio,
    transfersAffected,
    cascadeEnabled: Boolean(cascade.enabled),
    onTimeRatio: vehicles.length ? onTime / vehicles.length : null,
  };
}

function KpiCard({ label, value, tone = 'neutral', hint }) {
  return (
    <div className={`kpi kpi--${tone}`}>
      <span className="kpi__label">{label}</span>
      <strong className="kpi__value">{value}</strong>
      {hint ? <span className="kpi__hint">{hint}</span> : null}
    </div>
  );
}

export function KpiSidebar({ status, isFetching, lastUpdatedAt, children }) {
  const kpis = useMemo(() => computeKpis(status), [status]);
  const { selectedTrId, selectVehicle } = useAppState();

  const redTone = kpis.redZone > 0 ? 'bad' : 'ok';
  const transferTone =
    kpis.transfersAffected === null ? 'neutral' : kpis.transfersAffected > 0 ? 'warn' : 'ok';

  return (
    <aside className="sidebar">
      <section className="panel panel--summary">
        <header className="panel__header">
          <div>
            <h2>Состояние сейчас</h2>
            <span className="panel__subtitle">{isFetching ? 'Получаем обновление…' : `Данные на ${lastUpdatedAt}`}</span>
          </div>
          <span className={`pulse ${isFetching ? 'pulse--active' : ''}`} title={`обновлено ${lastUpdatedAt}`} />
        </header>

        <div className="kpi-grid">
          <KpiCard
            label="Средняя задержка"
            value={formatDelay(kpis.meanDelay)}
            tone={kpis.meanDelay > 120 ? 'bad' : kpis.meanDelay > 60 ? 'warn' : 'ok'}
            hint={`максимум ${formatDelay(kpis.worstDelay)}`}
          />
          <KpiCard
            label="Критичные ТС"
            value={`${formatInteger(kpis.redZone)} / ${formatInteger(kpis.totalVehicles)}`}
            tone={redTone}
            hint="от 120 с задержки"
          />
          <KpiCard
            label="В графике"
            value={formatRatio(kpis.onTimeRatio)}
            tone="ok"
            hint={`${formatInteger(kpis.routeCount)} маршрутов`}
          />
          <KpiCard
            label="Live-юниты"
            value={formatInteger(kpis.liveCount)}
            tone="neutral"
            hint={kpis.liveMeanDelay === null ? 'нет прогноза' : `сред. ${formatDelay(kpis.liveMeanDelay)}`}
          />
        </div>
      </section>

      <section className="panel">
        <header className="panel__header">
          <div>
            <h2>Пересадки</h2>
            <span className="panel__subtitle">Влияние задержек между маршрутами</span>
          </div>
        </header>
        {!kpis.cascadeEnabled ? (
          <p className="muted">Каскад недоступен: нет графа плановых остановок.</p>
        ) : (
          <div className="transfer-summary">
            <div>
              <strong>{formatInteger(kpis.transfersAffected)}</strong>
              <span>маршрутов получили задержку извне</span>
            </div>
            <div className={`transfer-score transfer-score--${transferTone}`}>
              {formatRatio(kpis.absorbedRatio)}
              <span>доля нагоняющих график</span>
            </div>
          </div>
        )}
        {kpis.transfersAffected !== null && kpis.transfersAffected > 0 ? (
          <p className="callout callout--warn">Проверьте узлы пересадок: часть маршрутов задерживается из-за соседних рейсов.</p>
        ) : null}
      </section>

      <VehicleList status={status} selectedTrId={selectedTrId} onSelect={selectVehicle} />

      {children}
    </aside>
  );
}

function VehicleList({ status, selectedTrId, onSelect }) {
  const vehicles = useMemo(() => {
    const list = Array.isArray(status?.vehicles) ? status.vehicles : [];
    return [...list].sort(
      (a, b) => (toFiniteNumber(b?.prediction) ?? -1e9) - (toFiniteNumber(a?.prediction) ?? -1e9),
    );
  }, [status]);

  if (!vehicles.length) {
    return (
      <section className="panel">
        <header className="panel__header">
          <div>
            <h2>Транспорт</h2>
            <span className="panel__subtitle">Список прогнозов</span>
          </div>
        </header>
        <p className="muted">Прогнозы появятся, когда симулятор достигнет времени T.</p>
      </section>
    );
  }

  return (
    <section className="panel panel--grow">
      <header className="panel__header">
        <div>
          <h2>Транспорт</h2>
          <span className="panel__subtitle">Сначала показаны ТС с наибольшей задержкой</span>
        </div>
        <span className="count-badge">{vehicles.length}</span>
      </header>
      <ul className="vehicle-list">
        {vehicles.map((vehicle) => {
          const key = String(vehicle?.tr_id);
          const active = key === String(selectedTrId);
          const delay = toFiniteNumber(vehicle?.prediction);
          const color = delayColor(delay);
          const critical = isRedZone(vehicle);
          const statusLabel = vehicle?.risk ? riskLabel(vehicle.risk) : critical ? 'критично' : 'под контролем';
          return (
            <li key={vehicle?.sample_id ?? `${key}-${vehicle?.T}`}>
              <button
                type="button"
                className={`vehicle-row ${active ? 'vehicle-row--active' : ''}`}
                onClick={() => onSelect(vehicle?.tr_id)}
                style={{ '--accent': color }}
              >
                <span className="vehicle-row__id">ТС {vehicle?.tr_id}</span>
                <span className="vehicle-row__delay">{formatDelay(delay)}</span>
                <span className={`vehicle-row__status ${critical ? 'vehicle-row__status--bad' : ''}`}>
                  {statusLabel}
                </span>
                <span className="vehicle-row__meta">{vehicle?.recommendation ?? 'Данных о рекомендации нет'}</span>
                <span className="vehicle-row__p">P опоздания · {formatPercent(vehicle?.p_late)}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
