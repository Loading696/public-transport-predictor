import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

const AppStateContext = createContext(null);

const SOURCES = [
  { id: 'all', label: 'Все' },
  { id: 'replay', label: 'Replay' },
  { id: 'live', label: 'Live' },
];

export function AppStateProvider({ children }) {
  const [source, setSource] = useState('all');
  const [speed, setSpeed] = useState(60);
  const [resetToken, setResetToken] = useState(0);
  const [selectedTrId, setSelectedTrId] = useState(null);
  const [focus, setFocus] = useState({ point: null, zoom: null, token: 0 });
  const focusCounter = useRef(0);

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
    }),
    [source, speed, resetToken, selectedTrId, focus, restartStream, selectVehicle, requestFocus],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) throw new Error('useAppState must be used inside AppStateProvider');
  return context;
}
