import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { config } from '../config.js';
import { pushStreamStatus } from '../api/queries.js';

const STREAM_KEY = ['stream-status'];
const MAX_BACKOFF_MS = 30000;

const idleState = () => ({
  mode: config.wsUrl ? 'websocket' : 'polling',
  connected: false,
  since: null,
});

export function useStreamTransport({ onOpen, onClose, onError } = {}) {
  const queryClient = useQueryClient();
  const [transport, setTransport] = useState(idleState);
  const retryTimer = useRef(null);
  const socketRef = useRef(null);
  const attemptRef = useRef(0);

  useEffect(() => {
    const wsUrl = config.wsUrl;
    if (!wsUrl) return undefined;

    let disposed = false;

    const clearRetry = () => {
      if (retryTimer.current !== null) {
        clearTimeout(retryTimer.current);
        retryTimer.current = null;
      }
    };

    const scheduleRetry = () => {
      if (disposed) return;
      attemptRef.current += 1;
      const delay = Math.min(1000 * 2 ** (attemptRef.current - 1), MAX_BACKOFF_MS);
      clearRetry();
      retryTimer.current = setTimeout(connect, delay);
    };

    function connect() {
      if (disposed) return;

      let socket;
      try {
        socket = new WebSocket(wsUrl);
      } catch (error) {
        onError?.(error);
        scheduleRetry();
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        attemptRef.current = 0;
        setTransport({ mode: 'websocket', connected: true, since: Date.now() });
        onOpen?.();
      };

      socket.onmessage = (event) => {
        const text = typeof event.data === 'string' ? event.data : null;
        if (text === null) return;
        try {
          const parsed = JSON.parse(text);
          pushStreamStatus(queryClient, parsed?.stream ?? parsed);
        } catch {
          return;
        }
      };

      socket.onerror = (error) => onError?.(error);

      socket.onclose = () => {
        if (socketRef.current === socket) socketRef.current = null;
        setTransport((current) => ({ ...current, connected: false }));
        onClose?.();
        scheduleRetry();
      };
    }

    connect();

    return () => {
      disposed = true;
      clearRetry();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [queryClient, onOpen, onClose, onError]);

  useEffect(() => {
    const pollMs = transport.connected ? false : config.pollIntervalMs;
    queryClient.getQueryCache().findAll({ queryKey: STREAM_KEY }).forEach((query) => {
      query.setOptions({ refetchInterval: pollMs });
    });
  }, [queryClient, transport.connected]);

  return transport;
}
