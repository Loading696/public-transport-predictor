const pad = (value) => String(value).padStart(2, '0');

export function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatIsoTime(value) {
  if (!value) return '—';
  const raw = String(value);
  if (raw.length < 19) return raw;
  return `${raw.slice(11, 19)}`;
}

export function formatIsoDateTime(value) {
  if (!value) return '—';
  const raw = String(value);
  if (raw.length < 19) return raw;
  return `${raw.slice(0, 10)} ${raw.slice(11, 19)}`;
}

export function formatDelay(seconds) {
  const value = toFiniteNumber(seconds);
  if (value === null) return '—';
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  const abs = Math.abs(value);
  if (abs < 60) return `${sign}${abs.toFixed(0)} с`;
  return `${sign}${abs.toFixed(0)} с (${(abs / 60).toFixed(1)} мин)`;
}

export function formatSecondsShort(seconds) {
  const value = toFiniteNumber(seconds);
  if (value === null) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(0)} с`;
}

export function formatPercent(value) {
  const parsed = toFiniteNumber(value);
  if (parsed === null) return '—';
  return `${(parsed * 100).toFixed(0)}%`;
}

export function formatSpeed(kmh) {
  const value = toFiniteNumber(kmh);
  if (value === null) return '—';
  return `${value.toFixed(0)} км/ч`;
}

export function formatRatio(value) {
  const parsed = toFiniteNumber(value);
  if (parsed === null) return '—';
  return `${(parsed * 100).toFixed(0)}%`;
}

export function formatInteger(value) {
  const parsed = toFiniteNumber(value);
  if (parsed === null) return '—';
  return Math.round(parsed).toString();
}

export function formatEtaDelta(planTime, eta) {
  if (!planTime || !eta) return '—';
  const plan = Date.parse(planTime);
  const target = Date.parse(eta);
  if (!Number.isFinite(plan) || !Number.isFinite(target)) return '—';
  const delta = Math.round((target - plan) / 1000);
  return `${delta > 0 ? '+' : ''}${delta} с`;
}

export function formatClock(date = new Date()) {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function formatAge(seconds) {
  const value = toFiniteNumber(seconds);
  if (value === null) return '—';
  if (value < 60) return `${value.toFixed(0)} с`;
  if (value < 3600) return `${(value / 60).toFixed(1)} мин`;
  return `${(value / 3600).toFixed(1)} ч`;
}
