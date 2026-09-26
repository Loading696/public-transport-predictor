import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useJsApiLoader, GoogleMap, MarkerF, PolylineF } from '@react-google-maps/api';
import { config } from '../config.js';
import { useAppState } from '../state/AppStateContext.jsx';
import { delayColor, riskColor, toFiniteNumber, doorState } from '../lib/delayScale.js';
import { gradientSegments, normalizePath } from '../lib/geometry.js';

const LIBRARIES = [];

const toLatLng = (point) => ({ lat: point[0], lng: point[1] });

export function GoogleMapView({
  routes,
  positions,
  liveUnits,
  vehicles,
  cascadeVehicles,
  stops,
  focus,
}) {
  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-maps-jsapi',
    googleMapsApiKey: config.google.apiKey,
    libraries: LIBRARIES,
    preventGoogleFontsLoading: true,
  });

  const mapRef = useRef(null);
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
          path: [toLatLng(segment.from), toLatLng(segment.to)],
          color: delayColor(segment.delay),
          width: entry?.tr_id === selectedTrId ? 8 : 6,
        })),
      ),
    [cascadeVehicles, selectedTrId],
  );

  const handleLoad = useCallback((map) => {
    mapRef.current = map;
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isLoaded || !focus?.point || !focus?.token) return;
    const next = toLatLng(focus.point);
    if (focus.zoom) {
      map.moveCamera({ center: next, zoom: focus.zoom });
    } else {
      map.panTo(next);
    }
  }, [focus, isLoaded]);

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
    return lat !== null && lon !== null
      ? toLatLng([lat, lon])
      : toLatLng(config.defaultCenter);
  }, [positions, liveUnits]);

  if (loadError) {
    return (
      <div className="map-shell map-shell--empty">
        <div className="map-placeholder">
          <h2>Google Maps не загрузился</h2>
          <p className="muted">{String(loadError.message || loadError)}</p>
        </div>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="map-shell map-shell--empty">
        <div className="map-placeholder">
          <h2>Загрузка Google Maps…</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="map-shell">
      <GoogleMap
        mapContainerClassName="map-shell__canvas"
        mapContainerStyle={{ width: '100%', height: '100%' }}
        center={initialCenter}
        zoom={config.defaultZoom}
        onLoad={handleLoad}
        mapId={config.google.mapId || undefined}
        options={{
          disableDefaultUI: false,
          clickableIcons: false,
          gestureHandling: 'greedy',
          styles: DARK_MAP_STYLE,
        }}
      >
        {routes.map((route) => {
          const path = normalizePath(route?.polyline).map(toLatLng);
          if (path.length < 2) return null;
          const key = String(route?.tr_id);
          const active = key === String(selectedTrId);
          return (
            <PolylineF
              key={`route-${key}`}
              path={path}
              options={{
                strokeColor: riskColor(route?.risk),
                strokeOpacity: active ? 0.95 : 0.3,
                strokeWeight: active ? 5 : 2,
              }}
            />
          );
        })}

        {stops.map((stop) => {
          const lat = toFiniteNumber(stop?.lat);
          const lon = toFiniteNumber(stop?.lon);
          if (lat === null || lon === null) return null;
          return (
            <MarkerF
              key={`stop-${stop?.stop_id}`}
              position={{ lat, lng: lon }}
              onClick={() => focusEntity(stop)}
              icon={{ path: 'M 0 -3 m 0 3 a 3 3 0 1 0 6 0 a 3 3 0 1 0 -6 0', scale: 1, fillOpacity: 0.9 }}
            />
          );
        })}

        {gradientPolylines.map((segment) => (
          <PolylineF
            key={segment.key}
            path={segment.path}
            options={{ strokeColor: segment.color, strokeOpacity: 0.95, strokeWeight: segment.width }}
          />
        ))}

        {positions.map((position) => {
          const lat = toFiniteNumber(position?.lat);
          const lon = toFiniteNumber(position?.lon);
          if (lat === null || lon === null) return null;
          const key = String(position?.tr_id);
          const vehicle = vehicleByTrId.get(key);
          return (
            <MarkerF
              key={`veh-${key}`}
              position={{ lat, lng: lon }}
              onClick={() => selectVehicle(position?.tr_id)}
              icon={{ path: 'M -6 -6 L 6 -6 L 4 6 L 0 3 L -4 6 Z', scale: 1.4, fillOpacity: 1, fillColor: delayColor(vehicle?.prediction ?? 0) }}
            />
          );
        })}

        {liveUnits.map((unit) => {
          const lat = toFiniteNumber(unit?.lat);
          const lon = toFiniteNumber(unit?.lon);
          if (lat === null || lon === null) return null;
          const door = doorState(unit?.door_open);
          return (
            <MarkerF
              key={`unit-${unit?.unit_id}`}
              position={{ lat, lng: lon }}
              onClick={() => selectVehicle(unit?.tr_id)}
              icon={{ path: 'M -6 -6 L 6 -6 L 4 6 L 0 3 L -4 6 Z', scale: 1.4, fillOpacity: 1, fillColor: door.color }}
            />
          );
        })}
      </GoogleMap>
    </div>
  );
}

const DARK_MAP_STYLE = [
  { elementType: 'geometry', stylers: [{ color: '#0f172a' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#0f172a' }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#94a3b8' }] },
  { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#1e293b' }] },
  { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#334155' }] },
  { featureType: 'transit', elementType: 'geometry', stylers: [{ color: '#1e293b' }] },
  { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#0b1220' }] },
];
