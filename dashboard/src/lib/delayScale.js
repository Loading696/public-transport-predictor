export const RISK_LEVELS = {
  'on-time': { label: 'в графике', color: '#22c55e', order: 0 },
  'at-risk': { label: 'под риском', color: '#eab308', order: 1 },
  late: { label: 'опоздание', color: '#ef4444', order: 2 },
  unknown: { label: 'нет данных', color: '#94a3b8', order: -1 },
};

export const RISK_TONE_BG = {
  'on-time': 'rgba(34, 197, 94, 0.14)',
  'at-risk': 'rgba(234, 179, 8, 0.16)',
  late: 'rgba(239, 68, 68, 0.18)',
  unknown: 'rgba(148, 163, 184, 0.12)',
};

export const DELAY_ABSORB_S = 120;
export const DELAY_HOT_S = 300;

const RAMP = [
  { at: 0, rgb: [34, 197, 94] },
  { at: DELAY_ABSORB_S, rgb: [234, 179, 8] },
  { at: DELAY_HOT_S, rgb: [239, 68, 68] },
];

const toHex = (rgb) => `#${rgb.map((c) => Math.round(c).toString(16).padStart(2, '0')).join('')}`;
export function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function delayColor(delaySeconds) {
  const value = toFiniteNumber(delaySeconds);
  if (value === null) return RISK_LEVELS.unknown.color;
  if (value <= 0) return toHex(RAMP[0].rgb);
  if (value >= DELAY_HOT_S) return toHex(RAMP[RAMP.length - 1].rgb);

  let lower = RAMP[0];
  let upper = RAMP[RAMP.length - 1];
  for (let i = 0; i < RAMP.length - 1; i += 1) {
    if (value >= RAMP[i].at && value <= RAMP[i + 1].at) {
      lower = RAMP[i];
      upper = RAMP[i + 1];
      break;
    }
  }
  const span = upper.at - lower.at;
  const t = span > 0 ? (value - lower.at) / span : 0;
  const mixed = lower.rgb.map((channel, i) => channel + (upper.rgb[i] - channel) * t);
  return toHex(mixed);
}

export function riskColor(risk) {
  return RISK_LEVELS[risk]?.color ?? RISK_LEVELS.unknown.color;
}

export function riskLabel(risk) {
  return RISK_LEVELS[risk]?.label ?? RISK_LEVELS.unknown.label;
}

export function riskOrder(risk) {
  return RISK_LEVELS[risk]?.order ?? -1;
}

export function isRedZone(vehicle) {
  const delay = toFiniteNumber(vehicle?.prediction);
  if (delay !== null && delay >= DELAY_ABSORB_S) return true;
  return vehicle?.risk === 'late';
}

export function isOnTime(vehicle) {
  const delay = toFiniteNumber(vehicle?.prediction);
  if (delay !== null && delay <= 0) return true;
  return vehicle?.risk === 'on-time';
}

export function doorState(doorOpen) {
  if (doorOpen === true) return { icon: '▣', color: '#f97316', label: 'двери открыты' };
  if (doorOpen === false) return { icon: '▢', color: '#38bdf8', label: 'двери закрыты' };
  return { icon: '·', color: '#94a3b8', label: 'двери нет данных' };
}
