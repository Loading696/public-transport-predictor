import { useEffect, useMemo, useRef, useState } from 'react';
import { config } from '../config.js';
import { useAppState } from '../state/AppStateContext.jsx';
import { delayColor, riskColor, toFiniteNumber, doorState } from '../lib/delayScale.js';
import { gradientSegments, normalizePath, toLngLat } from '../lib/geometry.js';
import { loadTwoGis, TwoGisLoaderError } from '../maps/twoGisLoader.js';

const Z_ROUTE = 100;
const Z_STOP = 300;
const Z_CASCADE = 400;
const Z_VEHICLE = 600;
const Z_LIVE = 620;
const Z_TARGET = 700;

const STOP_HTML = '<span class="map-pin map-pin--stop"></span>';
const doorHtml = (icon) => `<span class="map-pin map-pin--live">${icon}</span>`;
const targetHtml = () => '<span class="map-pin map-pin--target">⚑</span>';

const pointOf = (item) => toLngLat([toFiniteNumber(item?.lat), toFiniteNumber(item?.lon)]);

export function TwoGisMapView({
  routes,
  positions,
  liveUnits,
  vehicles,
  cascadeVehicles,
  stops,
  focus,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const [state, setState] = useState({ status: 'loading', error: null });

  const { selectedTrId, selectVehicle, requestFocus } = useAppState();

  const vehicleByTrId = useMemo(() => {
    const index = new Map();
    vehicles.forEach((vehicle) => index.set(String(vehicle?.tr_id), vehicle));
    return index;
  }, [vehicles]);

  const cascadeSegments = useMemo(
    () =>
      cascadeVehicles.flatMap((entry) => {
        const segments = gradientSegments(entry?.stops ?? [], (stop) => stop?.delay_s);
        const last = entry?.stops?.[entry.stops.length - 1];
        const point = pointOf(last);
        return {
          trId: entry?.tr_id,
          segments: segments
            .map((segment) => ({
              id: `cascade-${entry?.tr_id}-${segment.id}`,
              path: [toLngLat(segment.from), toLngLat(segment.to)].filter(Boolean),
              color: delayColor(segment.delay),
              active: String(entry?.tr_id) === String(selectedTrId),
            }))
            .filter((segment) => segment.path.length === 2),
          target: point ? { trId: entry?.tr_id, point, color: delayColor(last?.delay_s) } : null,
        };
      }),
    [cascadeVehicles, selectedTrId],
  );

  useEffect(() => {
    let disposed = false;
    const container = containerRef.current;
    if (!container) return undefined;

    loadTwoGis()
      .then((mapgl) => {
        if (disposed) return;
        const map = new mapgl.Map(container, {
          center: toLngLat(config.defaultCenter) ?? [37.6173, 55.7558],
          zoom: config.defaultZoom,
          key: config.twoGis.apiKey,
          zoomControl: 'topRight',
        });
        mapRef.current = { map, mapgl };
        setState({ status: 'ready', error: null });
      })
      .catch((error) => {
        if (disposed) return;
        setState({
          status: 'error',
          error: error instanceof TwoGisLoaderError ? error.message : String(error),
        });
      });

    return () => {
      disposed = true;
      const instance = mapRef.current;
      mapRef.current = null;
      try {
        instance?.map?.destroy?.();
      } catch {
        /* the SDK may already be torn down */
      }
    };
  }, []);

  useEffect(() => {
    const instance = mapRef.current;
    if (!instance || state.status !== 'ready') return undefined;
    return renderOverlays(instance, {
      routes,
      positions,
      liveUnits,
      vehicleByTrId,
      cascadeSegments,
      stops,
      selectedTrId,
      onSelect: selectVehicle,
      onFocus: (entity) => {
        const point = pointOf(entity);
        if (point) requestFocus([point[1], point[0]], 15);
      },
    });
  }, [
    routes,
    positions,
    liveUnits,
    vehicleByTrId,
    cascadeSegments,
    stops,
    selectedTrId,
    state.status,
    selectVehicle,
    requestFocus,
  ]);

  useEffect(() => {
    const instance = mapRef.current;
    if (!instance || state.status !== 'ready') return;
    if (!focus?.point || !focus?.token) return;
    const center = toLngLat(focus.point);
    if (!center) return;
    instance.map.setCenter(center, { duration: 450 });
    if (focus.zoom) instance.map.setZoom(focus.zoom, { duration: 450 });
  }, [focus, state.status]);

  if (state.status === 'error') {
    return (
      <div className="map-shell map-shell--empty">
        <div className="map-placeholder">
          <h2>2ГИС Карты не загрузились</h2>
          <p className="muted">{state.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="map-shell">
      <div className="map-shell__canvas" ref={containerRef} />
      {state.status === 'loading' ? (
        <div className="map-overlay">
          <span>Загрузка 2ГИС Карт…</span>
        </div>
      ) : null}
    </div>
  );
}

function renderOverlays(
  { map, mapgl },
  {
    routes,
    positions,
    liveUnits,
    vehicleByTrId,
    cascadeSegments,
    stops,
    selectedTrId,
    onSelect,
    onFocus,
  },
) {
  const created = [];
  let markerSeq = 0;

  const line = (coordinates, color, width, zIndex) => {
    if (!mapgl?.Polyline || coordinates.length < 2) return null;
    const object = new mapgl.Polyline(map, {
      coordinates,
      color,
      width,
      zIndex,
      interactive: true,
    });
    created.push(object);
    return object;
  };

  const marker = (coordinates, html, zIndex, onClick) => {
    if (!mapgl?.HtmlMarker || !coordinates) return null;
    const object = new mapgl.HtmlMarker(map, { coordinates, html, zIndex });
    if (onClick) object.on?.('click', onClick);
    object.userData = { seq: (markerSeq += 1) };
    created.push(object);
    return object;
  };

  routes.forEach((route) => {
    const path = normalizePath(route?.polyline).map(toLngLat).filter(Boolean);
    const active = String(route?.tr_id) === String(selectedTrId);
    line(path, riskColor(route?.risk), active ? 5 : 2, Z_ROUTE)?.on?.('click', () =>
      onSelect(route?.tr_id),
    );
  });

  cascadeSegments.forEach((layer) => {
    layer.segments.forEach((segment) => {
      line(segment.path, segment.color, segment.active ? 8 : 6, Z_CASCADE);
    });
    if (layer.target) {
      marker(layer.target.point, targetHtml(), Z_TARGET, () => onSelect(layer.trId));
    }
  });

  stops.forEach((stop) => {
    marker(pointOf(stop), STOP_HTML, Z_STOP, () => onFocus(stop));
  });

  positions.forEach((position) => {
    const key = String(position?.tr_id);
    const vehicle = vehicleByTrId.get(key);
    const html = `<span class="map-pin map-pin--vehicle" style="--pin:${delayColor(vehicle?.prediction ?? 0)}"></span>`;
    marker(pointOf(position), html, Z_VEHICLE, () => onSelect(position?.tr_id));
  });

  liveUnits.forEach((unit) => {
    const door = doorState(unit?.door_open);
    marker(pointOf(unit), doorHtml(door.icon), Z_LIVE, () => onSelect(unit?.tr_id));
  });

  return () => {
    created.forEach((object) => {
      try {
        object.destroy?.();
      } catch {
        /* object may already be destroyed */
      }
    });
  };
}
