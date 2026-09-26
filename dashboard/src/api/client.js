import { config } from '../config.js';

export class ApiError extends Error {
  constructor(message, { status, detail, url } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status ?? 0;
    this.detail = detail;
    this.url = url;
  }
}

const buildUrl = (path, params) => {
  const search = new URLSearchParams();
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    search.set(key, String(value));
  });
  const query = search.toString();
  return `${config.apiBase}${path}${query ? `?${query}` : ''}`;
};

export async function requestJson(path, { params, signal, timeoutMs = 8000 } = {}) {
  const url = buildUrl(path, params);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', () => controller.abort(), { once: true });
  }

  let response;
  try {
    response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(`Сеть недоступна: ${url}`, { url });
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.reason)) ||
      `${response.status} ${response.statusText}`;
    throw new ApiError(detail, { status: response.status, detail, url });
  }

  return payload;
}

export const api = {
  streamStatus: (params, signal) => requestJson('/stream/status', { params, signal }),
  cascade: (signal) => requestJson('/cascade', { signal }),
  ndtpStatus: (signal) => requestJson('/ndtp/status', { signal }),
  health: (signal) => requestJson('/health', { signal }),
  predict: (payload, signal) =>
    fetch(`${config.apiBase}/predict`, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
    }).then(async (response) => {
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new ApiError(data?.detail || 'predict failed', { status: response.status });
      }
      return data;
    }),
};

export function absoluteApiUrl(path) {
  return `${config.apiBase}${path}`;
}
