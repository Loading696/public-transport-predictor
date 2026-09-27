import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import { DataProvider } from './state/DataProvider.jsx';
import { AppStateProvider } from './state/AppStateContext.jsx';
import { config } from './config.js';
import './styles.css';

const container = document.getElementById('root');

createRoot(container).render(
  <StrictMode>
    <DataProvider>
      <AppStateProvider>
        <App />
      </AppStateProvider>
    </DataProvider>
  </StrictMode>,
);

if (config.provider === 'none') {
  console.warn(
    '[dashboard] Не задан ни VITE_YANDEX_API_KEY, ни VITE_GOOGLE_MAPS_API_KEY. ' +
      'OpenStreetMap намеренно не поддерживается: используйте коммерческий картографический сервис.',
  );
}
