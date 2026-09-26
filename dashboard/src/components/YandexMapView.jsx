import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  YMaps,
  Map as YMap,
  Placemark,
  Polyline,
  ZoomControl,
  FullscreenControl,
  TypeSelector,
  GeolocationControl,
} from '@pbe/react-yandex-maps';
import { config } from '../config.js';
import { useAppState } from '../state/AppStateContext.jsx';
import { delayColor, riskColor, toFiniteNumber, doorState } from '../lib/delayScale.js';
import { gradientSegments, normalizePath } from '../lib/geometry.js';

const MODULES = [
  'Map',
  'Placemark',
  'Polyline',
  'ZoomControl',
  'FullscreenControl',
  'TypeSelector',
  'GeolocationControl',
];

const busPreset = (color) => ({ iconColor: color, preset: 'islands#greenCircleBusIcon' });

export function YandexMapView({
  routes,
  positions,
  liveUnits,
  vehicles,
  cascadeVehicles,
  stops,
  focus,
}) {
  const mapRef = useRef(null);
  const [ready, setReady] = useState(false);
  const { selectedTrId, selectVehicle, requestFocus } = useAppState();

  const vehicleByTrId = useMemo(() => {
    const index = new Map();
    vehicles.forEach((vehicle) => index.set(String(vehicle?.tr_id), vehicle));
    return index;
  }, [vehicles]);

  const gradientPolylines = useMemo(
    () =>
      cascadeVehicles.flatMap((entry) =>
        gradientSegments(entry?.stops ?? [], (stop) => stop?.delay_s).map((segment) => ({
          key: `cascade-${entry?.tr_id}-${segment.id}`,
          geometry: [segment.from, segment.to],
          color: delayColor(segment.delay),
          width: entry?.tr_id === selectedTrId ? 8 : 6,
        })),
      ),
    [cascadeVehicles, selectedTrId],
  );

  const cascadeTargets = useMemo(
    () =>
      cascadeVehicles
        .map((entry) => {
          const last = entry?.stops?.[entry.stops.length - 1];
          const lat = toFiniteNumber(last?.lat);
          const lon = toFiniteNumber(last?.lon);
          if (lat === null || lon === null) return null;
          return { trId: entry?.tr_id, lat, lon, delay: last?.delay_s };
        })
        .filter(Boolean),
    [cascadeVehicles],
  );

  const handleInstance = useCallback((instance) => {
    mapRef.current = instance;
    if (instance) setReady(true);
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !focus?.point || !focus?.token) return;
    const [lat, lon] = focus.point;
    const options = { checkZoomRange: true, duration: 450 };
    if (focus.zoom) options.zoom = focus.zoom;
    map.setCenter([lat, lon], options);
  }, [focus, ready]);

  const focusEntity = useCallback(
    (entity) => {
      const lat = toFiniteNumber(entity?.lat);
      const lon = toFiniteNumber(entity?.lon);
      if (lat !== null && lon !== null) {
        requestFocus([lat, lon], Math.max(config.defaultZoom, 15));
      }
    },
    [requestFocus],
  );

  const initialCenter = useMemo(() => {
    const sample = positions[0] ?? liveUnits[0];
    const lat = toFiniteNumber(sample?.lat);
    const lon = toFiniteNumber(sample?.lon);
    return lat !== null && lon !== null ? [lat, lon] : config.defaultCenter;
  }, [positions, liveUnits]);

  return (
    <div className="map-shell">
      <YMaps
        query={{
          apikey: config.yandex.apiKey,
          lang: config.yandex.lang,
          load: MODULES.join(','),
        }}
      >
        <YMap
          defaultState={{ center: initialCenter, zoom: config.defaultZoom, width: '100%', height: '100%' }}
          instanceRef={handleInstance}
          modules={MODULES}
        >
          <ZoomControl position={{ right: 12, top: 12 }} />
          <FullscreenControl position={{ right: 12, top: 84 }} />
          <TypeSelector position={{ right: 12, top: 132 }} />
          <GeolocationControl position={{ right: 12, top: 180 }} />

          {routes.map((route) => {
            const geometry = normalizePath(route?.polyline);
            if (geometry.length < 2) return null;
            const key = String(route?.tr_id);
            const active = key === String(selectedTrId);
            return (
              <Polyline
                key={`route-${key}`}
                geometry={geometry}
                strokeColor={riskColor(route?.risk)}
                strokeOpacity={active ? 0.95 : 0.3}
                strokeWidth={active ? 5 : 2}
                strokeDasharray={active ? undefined : '6 8'}
              />
            );
          })}

          {stops.map((stop) => {
            const lat = toFiniteNumber(stop?.lat);
            const lon = toFiniteNumber(stop?.lon);
            if (lat === null || lon === null) return null;
            return (
              <Placemark
                key={`stop-${stop?.stop_id}`}
                geometry={[lat, lon]}
                preset="islands#tinyDotIcon"
                onClick={() => focusEntity(stop)}
              />
            );
          })}

          {gradientPolylines.map((segment) => (
            <Polyline
              key={segment.key}
              geometry={segment.geometry}
              strokeColor={segment.color}
              strokeOpacity={0.95}
              strokeWidth={segment.width}
              strokeLinecap="round"
            />
          ))}

          {cascadeTargets.map((target) => (
            <Placemark
              key={`target-${target.trId}`}
              geometry={[target.lat, target.lon]}
              preset={{
                iconColor: delayColor(target.delay),
                iconShape: 'flag',
                preset: 'islands#redCircleDotIcon',
              }}
              onClick={() => selectVehicle(target.trId)}
            />
          ))}

          {positions.map((position) => {
            const lat = toFiniteNumber(position?.lat);
            const lon = toFiniteNumber(position?.lon);
            if (lat === null || lon === null) return null;
            const key = String(position?.tr_id);
            const vehicle = vehicleByTrId.get(key);
            return (
              <Placemark
                key={`veh-${key}`}
                geometry={[lat, lon]}
                preset={busPreset(delayColor(vehicle?.prediction ?? 0))}
                onClick={() => selectVehicle(position?.tr_id)}
              />
            );
          })}

          {liveUnits.map((unit) => {
            const lat = toFiniteNumber(unit?.lat);
            const lon = toFiniteNumber(unit?.lon);
            if (lat === null || lon === null) return null;
            const door = doorState(unit?.door_open);
            return (
              <Placemark
                key={`unit-${unit?.unit_id}`}
                geometry={[lat, lon]}
                preset={busPreset(door.color)}
                onClick={() => selectVehicle(unit?.tr_id)}
              />
            );
          })}
        </YMap>
      </YMaps>
    </div>
  );
}
