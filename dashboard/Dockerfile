FROM node:22-alpine AS build

WORKDIR /app

COPY dashboard/package.json dashboard/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY dashboard/ ./

ARG VITE_API_BASE=/api
ARG VITE_MAP_PROVIDER=2gis
ARG VITE_TWOGIS_API_KEY=
ARG VITE_TWOGIS_LOCALE=ru_RU
ARG VITE_YANDEX_API_KEY=
ARG VITE_GOOGLE_MAPS_API_KEY=
ARG VITE_DEFAULT_CENTER=55.7558,37.6173
ARG VITE_DEFAULT_ZOOM=11
ARG VITE_POLL_INTERVAL_MS=3000
ARG VITE_WS_URL=

ENV VITE_API_BASE=${VITE_API_BASE} \
    VITE_MAP_PROVIDER=${VITE_MAP_PROVIDER} \
    VITE_TWOGIS_API_KEY=${VITE_TWOGIS_API_KEY} \
    VITE_TWOGIS_LOCALE=${VITE_TWOGIS_LOCALE} \
    VITE_YANDEX_API_KEY=${VITE_YANDEX_API_KEY} \
    VITE_GOOGLE_MAPS_API_KEY=${VITE_GOOGLE_MAPS_API_KEY} \
    VITE_DEFAULT_CENTER=${VITE_DEFAULT_CENTER} \
    VITE_DEFAULT_ZOOM=${VITE_DEFAULT_ZOOM} \
    VITE_POLL_INTERVAL_MS=${VITE_POLL_INTERVAL_MS} \
    VITE_WS_URL=${VITE_WS_URL}

RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=4s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
