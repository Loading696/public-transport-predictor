import { load } from '@2gis/mapgl';

const SDK_URL = 'https://mapgl.2gis.com/api/js';

let pending = null;

export class TwoGisLoaderError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'TwoGisLoaderError';
    this.cause = cause;
  }
}

export function loadTwoGis() {
  if (pending) return pending;
  pending = load(SDK_URL).catch((error) => {
    pending = null;
    throw new TwoGisLoaderError(
      `Не удалось загрузить 2GIS MapGL: ${error?.message ?? error}`,
      error,
    );
  });
  return pending;
}

export function resetTwoGisLoader() {
  pending = null;
}
