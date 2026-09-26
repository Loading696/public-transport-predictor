import { toFiniteNumber } from './delayScale.js';

const asPoint = (value) => {
  if (Array.isArray(value) && value.length >= 2) {
    const lat = toFiniteNumber(value[0]);
    const lon = toFiniteNumber(value[1]);
    return lat !== null && lon !== null ? [lat, lon] : null;
  }
  if (value && typeof value === 'object') {
    const lat = toFiniteNumber(value.lat);
    const lon = toFiniteNumber(value.lon ?? value.lng);
    return lat !== null && lon !== null ? [lat, lon] : null;
  }
  return null;
};

export function normalizePoint(value) {
  return asPoint(value);
}

export function toLngLat(point) {
  const normalized = asPoint(point);
  return normalized ? [normalized[1], normalized[0]] : null;
}

export function swapToLngLat(point) {
  return toLngLat(point);
}

export function normalizePath(value) {
  if (!Array.isArray(value)) return [];
  return value.map(asPoint).filter(Boolean);
}

export function extendBounds(bounds, point) {
  const normalized = asPoint(point);
  if (!normalized) return bounds;
  const [lat, lon] = normalized;
  if (!bounds) return { minLat: lat, maxLat: lat, minLon: lon, maxLon: lon };
  return {
    minLat: Math.min(bounds.minLat, lat),
    maxLat: Math.max(bounds.maxLat, lat),
    minLon: Math.min(bounds.minLon, lon),
    maxLon: Math.max(bounds.maxLon, lon),
  };
}

export function boundsFrom(points) {
  return points.reduce(extendBounds, null);
}

export function boundsCenter(bounds, fallback = [55.7558, 37.6173]) {
  if (!bounds) return fallback;
  return [(bounds.minLat + bounds.maxLat) / 2, (bounds.minLon + bounds.maxLon) / 2];
}

export function spanDegrees(bounds) {
  if (!bounds) return { lat: 0, lon: 0 };
  return {
    lat: Math.abs(bounds.maxLat - bounds.minLat),
    lon: Math.abs(bounds.maxLon - bounds.minLon),
  };
}

export function gradientSegments(stops, delayAccessor) {
  const points = (stops ?? [])
    .map((stop) => {
      const delay = toFiniteNumber(delayAccessor(stop));
      const base = asPoint(stop?.point ?? stop);
      return { point: base, delay };
    })
    .filter((entry) => entry.point !== null && entry.delay !== null);

  const segments = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const from = points[i];
    const to = points[i + 1];
    segments.push({
      id: `${i}-${from.point[0]}-${from.point[1]}-${to.point[0]}-${to.point[1]}`,
      from: from.point,
      to: to.point,
      fromDelay: from.delay,
      toDelay: to.delay,
      delay: Math.max(from.delay, to.delay, 0),
    });
  }
  return segments;
}

export function capStops(stops, limit) {
  if (!Array.isArray(stops) || stops.length <= limit) return stops ?? [];
  const step = stops.length / limit;
  const sampled = [];
  for (let i = 0; i < limit; i += 1) {
    sampled.push(stops[Math.floor(i * step)]);
  }
  return sampled;
}
