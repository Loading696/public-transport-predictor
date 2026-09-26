import { lazy, Suspense, useMemo } from 'react';
import { config } from '../config.js';
import { useAppState } from '../state/AppStateContext.jsx';

const YandexMapView = lazy(() =>
  import('./YandexMapView.jsx').then((module) => ({ default: module.YandexMapView })),
);
const GoogleMapView = lazy(() =>
  import('./GoogleMapView.jsx').then((module) => ({ default: module.GoogleMapView })),
);
const TwoGisMapView = lazy(() =>
  import('./TwoGisMapView.jsx').then((module) => ({ default: module.TwoGisMapView })),
);

const PROVIDERS = {
  '2gis': TwoGisMapView,
  yandex: YandexMapView,
  google: GoogleMapView,
};

const MAX_ROUTE_POLYLINES = 24;
const MAX_SELECTED_STOPS = 240;

export const PROVIDER_LABEL = {
  '2gis': '2ГИС Карты',
  yandex: 'Яндекс Карты',
  google: 'Google Maps',
  none: 'нет ключа',
};

export function buildMapFrame(status, { showReplay, showLive, selectedTrId }) {
  const map = status?.map ?? {};
  const cascade = map.cascade ?? status?.cascade ?? { enabled: false, vehicles: [] };
  const allVehicles = Array.isArray(status?.vehicles) ? status.vehicles : [];
  const allRoutes = Array.isArray(map.routes) ? map.routes : [];
  const allPositions = Array.isArray(map.positions) ? map.positions : [];
  const allLive = Array.isArray(status?.live_units) ? status.live_units : [];

  const routes = showReplay ? allRoutes.slice(0, MAX_ROUTE_POLYLINES) : [];
  const positions = showReplay ? allPositions : [];
  const liveUnits = showLive
    ? allLive.filter((unit) => toPoint(unit) !== null)
    : [];

  const vehicles = showReplay ? allVehicles : [];

  const stopIndex = buildStopIndex(allRoutes);

  const cascadeVehicles = cascade.enabled && showReplay && Array.isArray(cascade.vehicles)
    ? cascade.vehicles.map((entry) => ({ ...entry, stops: withCoordinates(entry?.stops, stopIndex) }))
    : [];

  const selectedRoute = selectedTrId === null
    ? null
    : allRoutes.find((route) => route?.tr_id === selectedTrId) ?? null;

  const stops = showReplay && selectedRoute && Array.isArray(selectedRoute.stops)
    ? thin(selectedRoute.stops, MAX_SELECTED_STOPS)
    : [];

  return { routes, positions, liveUnits, vehicles, cascadeVehicles, cascade, selectedRoute, stops };
}

function buildStopIndex(routes) {
  const index = new Map();
  routes.forEach((route) => {
    const trId = String(route?.tr_id);
    (route?.stops ?? []).forEach((stop) => {
      const lat = Number(stop?.lat);
      const lon = Number(stop?.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        index.set(`${trId}:${stop?.stop_id}`, [lat, lon]);
      }
    });
  });
  return index;
}

function withCoordinates(stops, index) {
  if (!Array.isArray(stops) || !index) return [];
  return stops
    .map((stop) => {
      const point = index.get(`${stop?.tr_id}:${stop?.stop_id}`);
      return point ? { ...stop, lat: point[0], lon: point[1] } : null;
    })
    .filter(Boolean);
}

function toPoint(unit) {
  const lat = Number(unit?.lat);
  const lon = Number(unit?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return [lat, lon];
}

function thin(items, limit) {
  if (items.length <= limit) return items;
  const step = items.length / limit;
  const output = [];
  for (let i = 0; i < limit; i += 1) output.push(items[Math.floor(i * step)]);
  return output;
}

export function TransportMap({ status, isFetching, error, lastUpdatedAt }) {
  const { showReplay, showLive, selectedTrId, focus } = useAppState();

  const frame = useMemo(
    () => buildMapFrame(status, { showReplay, showLive, selectedTrId }),
    [status, showReplay, showLive, selectedTrId],
  );

  const shared = useMemo(
    () => ({ ...frame, isFetching, error, lastUpdatedAt }),
    [frame, isFetching, error, lastUpdatedAt],
  );

  if (config.provider === 'none') {
    return <MapKeyMissing />;
  }

  const Provider = PROVIDERS[config.provider];
  if (!Provider) return <MapKeyMissing />;

  return (
    <Suspense fallback={<MapLoading provider={config.provider} />}>
      <Provider {...shared} focus={focus} />
    </Suspense>
  );
}

function MapLoading({ provider }) {
  return (
    <div className="map-shell map-shell--empty">
      <div className="map-placeholder">
        <h2>Загрузка карты…</h2>
        <p className="muted">Провайдер: {PROVIDER_LABEL[provider] ?? provider}</p>
      </div>
    </div>
  );
}

function MapKeyMissing() {
  return (
    <div className="map-shell map-shell--empty">
      <div className="map-placeholder">
        <h2>Не задан ключ картографического сервиса</h2>
        <p>
          Провайдер выбирается автоматически: <code>VITE_TWOGIS_API_KEY</code> → 2ГИС Карты,
          затем <code>VITE_YANDEX_API_KEY</code> → Яндекс Карты,
          <code>VITE_GOOGLE_MAPS_API_KEY</code> → Google Maps. Задайте хотя бы один ключ в{' '}
          <code>dashboard/.env</code>.
        </p>
        <p className="muted">
          OpenStreetMap в проекте не используется и не поддерживается: тайлы требуют
          публичного ключа коммерческого сервиса и соблюдения его условий использования.
        </p>
      </div>
    </div>
  );
}
