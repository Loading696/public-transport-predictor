const num = (raw, fallback) => {
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : fallback;
};

const stripTrailingSlash = (raw) => (raw || '').replace(/\/+$/, '');

const twoGisKey = (import.meta.env.VITE_TWOGIS_API_KEY || '').trim();
const yandexKey = (import.meta.env.VITE_YANDEX_API_KEY || '').trim();
const googleKey = (import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '').trim();

const requestedProvider = (import.meta.env.VITE_MAP_PROVIDER || 'auto').toLowerCase();

const READY = {
  '2gis': () => twoGisKey,
  yandex: () => yandexKey,
  google: () => googleKey,
};

// 2GIS first: it is the provider this deployment is licensed for. Yandex and
// Google stay selectable via VITE_MAP_PROVIDER as commercial fallbacks. There is
// no OpenStreetMap branch by construction.
const FALLBACK_ORDER = ['2gis', 'yandex', 'google'];

const resolveProvider = () => {
  if (READY[requestedProvider]?.()) return requestedProvider;
  return FALLBACK_ORDER.find((name) => READY[name]()) ?? 'none';
};

const center = String(import.meta.env.VITE_DEFAULT_CENTER || '55.7558,37.6173')
  .split(',')
  .map((part) => num(part.trim(), 0));

export const config = {
  apiBase: stripTrailingSlash(import.meta.env.VITE_API_BASE || '/api'),
  pollIntervalMs: Math.max(500, num(import.meta.env.VITE_POLL_INTERVAL_MS, 3000)),
  wsUrl: (import.meta.env.VITE_WS_URL || '').trim(),
  providerPreference: requestedProvider,
  provider: resolveProvider(),
  twoGis: { apiKey: twoGisKey, locale: (import.meta.env.VITE_TWOGIS_LOCALE || 'ru_RU').trim() },
  yandex: { apiKey: yandexKey, lang: 'ru_RU' },
  google: { apiKey: googleKey, mapId: (import.meta.env.VITE_GOOGLE_MAP_ID || '').trim() },
  defaultCenter: [center[0] ?? 55.7558, center[1] ?? 37.6173],
  defaultZoom: Math.min(19, Math.max(3, num(import.meta.env.VITE_DEFAULT_ZOOM, 11))),
  providerStatus() {
    return {
      active: this.provider,
      preference: this.providerPreference,
      twoGisReady: Boolean(this.twoGis.apiKey),
      yandexReady: Boolean(this.yandex.apiKey),
      googleReady: Boolean(this.google.apiKey),
    };
  },
};
