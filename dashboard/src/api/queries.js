import { useEffect, useRef } from 'react';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client.js';
import { config } from '../config.js';

export const streamKeys = {
  status: (params) => ['stream-status', params.speed ?? null],
  cascade: () => ['cascade'],
  ndtp: () => ['ndtp-status'],
  health: () => ['health'],
};

export const EMPTY_STATUS = {
  simulated_time: null,
  speed: null,
  finished: false,
  processed_points: 0,
  total_points: 0,
  processed_traffic_rows: 0,
  total_traffic_rows: 0,
  vehicles: [],
  map: { routes: [], positions: [], incident: null, cascade: { enabled: false } },
  live_units: [],
};

export function useStreamStatus({ speed, resetToken = 0, enabled = true }) {
  const queryClient = useQueryClient();
  const pendingReset = useRef(false);
  const mountedRef = useRef(false);

  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      return;
    }
    pendingReset.current = true;
    queryClient.invalidateQueries({ queryKey: ['stream-status'] });
  }, [resetToken, queryClient]);

  return useQuery({
    queryKey: streamKeys.status({ speed }),
    queryFn: ({ signal }) => {
      const reset = pendingReset.current;
      pendingReset.current = false;
      return api.streamStatus({ speed, reset: reset ? 'true' : undefined }, signal);
    },
    enabled,
    refetchInterval: config.pollIntervalMs,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
    staleTime: Math.floor(config.pollIntervalMs / 2),
    placeholderData: keepPreviousData,
    retry: 2,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
  });
}

export function useCascadeSnapshot(cascade, enabled = true) {
  return useQuery({
    queryKey: streamKeys.cascade(),
    queryFn: ({ signal }) => api.cascade(signal),
    enabled: enabled && !cascade?.enabled,
    refetchInterval: config.pollIntervalMs * 4,
    refetchIntervalInBackground: false,
    staleTime: config.pollIntervalMs * 3,
    placeholderData: keepPreviousData,
  });
}

export function useNdtpStatus(enabled = true) {
  return useQuery({
    queryKey: streamKeys.ndtp(),
    queryFn: ({ signal }) => api.ndtpStatus(signal),
    enabled,
    refetchInterval: config.pollIntervalMs * 4,
    refetchIntervalInBackground: false,
    staleTime: config.pollIntervalMs * 3,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: streamKeys.health(),
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: config.pollIntervalMs * 5,
    refetchIntervalInBackground: false,
    staleTime: config.pollIntervalMs * 4,
  });
}

export function pushStreamStatus(queryClient, payload) {
  if (!payload || typeof payload !== 'object') return;
  queryClient.setQueriesData(
    { queryKey: ['stream-status'] },
    (previous) => ({ ...(previous ?? EMPTY_STATUS), ...payload }),
  );
}
