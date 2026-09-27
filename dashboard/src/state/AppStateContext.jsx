import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

const AppStateContext = createContext(null);

const SOURCES = [
  { id: 'all', label: 'Все данные' },
  { id: 'replay', label: 'Симуляция' },
  { id: 'live', label: 'Live' },
];

const THEME_STORAGE_KEY = 'transport-dashboard-theme';

function getInitialTheme() {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export function AppStateProvider({ children }) {
  const [source, setSource] = useState('all');
  const [speed, setSpeed] = useState(60);
  const [resetToken, setResetToken] = useState(0);
  const [selectedTrId, setSelectedTrId] = useState(null);
  const [theme, setTheme] = useState(getInitialTheme);
  const [focus, setFocus] = useState({ point: null, zoom: null, token: 0 });
  const focusCounter = useRef(0);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // localStorage may be unavailable in private or restricted browser contexts.
    }
  }, [theme]);

  const requestFocus = useCallback((point, zoom) => {
    if (!point) return;
    focusCounter.current += 1;
    setFocus({ point, zoom: zoom ?? null, token: focusCounter.current });
  }, []);

  const selectVehicle = useCallback((trId) => {
    setSelectedTrId((current) => (current === trId ? null : trId));
  }, []);

  const restartStream = useCallback(() => {
    setResetToken((token) => token + 1);
  }, []);

  const value = useMemo(
    () => ({
      sources: SOURCES,
      source,
      setSource,
      showReplay: source === 'all' || source === 'replay',
      showLive: source === 'all' || source === 'live',
      speed,
      setSpeed,
      resetToken,
      restartStream,
      selectedTrId,
      selectVehicle,
      focus,
      requestFocus,
      theme,
      setTheme,
      toggleTheme: () => setTheme((current) => (current === 'light' ? 'dark' : 'light')),
    }),
    [source, speed, resetToken, selectedTrId, focus, restartStream, selectVehicle, requestFocus, theme],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) throw new Error('useAppState must be used inside AppStateProvider');
  return context;
}
