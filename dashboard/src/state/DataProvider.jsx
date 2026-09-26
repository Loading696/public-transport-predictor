import { createContext, useContext, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiError } from '../api/client.js';
import { useStreamTransport } from '../hooks/useStreamTransport.js';

const ConnectionContext = createContext({ mode: 'polling', connected: false });

const createClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 2000,
        gcTime: 5 * 60 * 1000,
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
          return failureCount < 3;
        },
      },
    },
  });

function ConnectionBridge({ children }) {
  const [fallbackState, setFallbackState] = useState({ mode: 'polling', connected: false });

  const transport = useStreamTransport({
    onOpen: () => setFallbackState({ mode: 'websocket', connected: true }),
    onClose: () => setFallbackState((current) => ({ ...current, connected: false })),
  });

  const value =
    transport.mode === 'websocket'
      ? { mode: transport.mode, connected: transport.connected, since: transport.since }
      : fallbackState;

  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

export function useConnection() {
  return useContext(ConnectionContext);
}

export function DataProvider({ children }) {
  const [client] = useState(createClient);
  return (
    <QueryClientProvider client={client}>
      <ConnectionBridge>{children}</ConnectionBridge>
    </QueryClientProvider>
  );
}
