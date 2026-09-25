from __future__ import annotations

import html
import json
import math
import os
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Предиктор задержек", layout="wide")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.title("Предиктор изменений графика движения")
st.caption("Демо-контур диспетчера: телеметрия validate воспроизводится как поток, прогнозы появляются после наступления T.")

speed = st.sidebar.slider("Скорость симуляции", min_value=1, max_value=600, value=60, step=10)
reset = st.sidebar.button("Перезапустить поток")
query: dict[str, int | str] = {"speed": int(speed)}
if reset:
    query["reset"] = "true"
url = f"{BACKEND_URL}/stream/status?{urlencode(query)}"
st.sidebar.caption(f"API: {BACKEND_URL}")
auto = st.sidebar.checkbox("Автообновление каждые 2 с", value=True)


def load_status() -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def finite_number(value: object | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def short_time(value: object | None) -> str:
    if value is None:
        return "—"
    return str(value)[:19].replace("T", " ")


def format_delay(seconds: object | None) -> str:
    delay = finite_number(seconds)
    if delay is None:
        return "—"
    return f"{delay:+.0f} с ({delay / 60.0:+.1f} мин)"


def stop_text(stop: dict | None) -> str:
    if not stop:
        return "—"
    address = stop.get("address") or f"остановка {stop.get('stop_id', '—')}"
    return f"{address} · {short_time(stop.get('time'))}"


def segment_text(segment: dict | None) -> str:
    if not segment:
        return "—"
    previous = stop_text(segment.get("previous"))
    following = stop_text(segment.get("next"))
    if segment.get("previous") and segment.get("next"):
        return f"{previous} → {following}"
    if segment.get("next"):
        return f"следующая: {following}"
    if segment.get("previous"):
        return f"после: {previous}"
    return "—"


def target_stop_text(incident: dict) -> str:
    snapshot = incident.get("snapshot", {})
    target = snapshot.get("target") or {}
    address = target.get("address") or f"остановка {incident.get('target_stop_id', '—')}"
    return f"{address} · {short_time(incident.get('target_time'))}"


def map_center(routes: list, positions: list) -> list[float]:
    points = []
    for position in positions:
        lon = finite_number(position.get("lon"))
        lat = finite_number(position.get("lat"))
        if lon is not None and lat is not None:
            points.append([lat, lon])
    for route in routes:
        for point in route.get("polyline", []):
            lat = finite_number(point[0]) if len(point) > 0 else None
            lon = finite_number(point[1]) if len(point) > 1 else None
            if lat is not None and lon is not None:
                points.append([lat, lon])
    if not points:
        return [55.7558, 37.6173]
    return [sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)]


def render_network(routes: list, positions: list, vehicles_by_id: dict) -> None:
    geometry_routes = []
    for route in routes:
        polyline = []
        for point in route.get("polyline", []):
            lat = finite_number(point[0]) if len(point) > 0 else None
            lon = finite_number(point[1]) if len(point) > 1 else None
            if lat is not None and lon is not None:
                polyline.append([lat, lon])
        if len(polyline) >= 2:
            geometry_routes.append({**route, "polyline": polyline})
    if not geometry_routes and not positions:
        st.info("Маршрутная геометрия появится после первых прогнозов.")
        return
    network = folium.Map(location=map_center(geometry_routes, positions), zoom_start=11, tiles="CartoDB positron")
    for route in geometry_routes:
        vehicle = vehicles_by_id.get(route.get("tr_id"), {})
        folium.PolyLine(
            route["polyline"],
            color=route.get("color", "#9ca3af"),
            weight=5,
            opacity=0.85,
            tooltip=f"ТС {route.get('tr_id')} · {vehicle.get('risk', route.get('risk', '—'))}",
        ).add_to(network)
        target_id = str(vehicle.get("target_stop_id", ""))
        for stop in route.get("stops", []):
            lat = finite_number(stop.get("lat"))
            lon = finite_number(stop.get("lon"))
            if lat is None or lon is None:
                continue
            if target_id and str(stop.get("stop_id")) == target_id:
                folium.Marker(
                    [lat, lon],
                    icon=folium.Icon(color="red", icon="flag"),
                    tooltip=f"Цель ТС {route.get('tr_id')}: {stop.get('address') or stop.get('stop_id')}",
                ).add_to(network)
    for position in positions:
        lat = finite_number(position.get("lat"))
        lon = finite_number(position.get("lon"))
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            [lat, lon],
            radius=7,
            color=position.get("color", "#111827") if isinstance(position.get("color"), str) else "#111827",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"ТС {position.get('tr_id')} · {short_time(position.get('event_time'))}",
        ).add_to(network)
    st_folium(network, width=1100, height=520, key="route-network", returned_objects=[])
    st.markdown(
        '<div style="display:flex;gap:16px;margin:4px 0 0">'
        '<span><span style="color:#22c55e">●</span> в графике</span>'
        '<span><span style="color:#eab308">●</span> под риском</span>'
        '<span><span style="color:#ef4444">●</span> опоздание</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_incident(incident: dict | None) -> None:
    if not incident:
        st.info("Активных инцидентов с прогнозом задержки от 60 секунд пока нет.")
        return
    snapshot = incident.get("snapshot", {})
    position = incident.get("position_now") or snapshot.get("position") or {}
    delay = finite_number(incident.get("predicted_delay_s"))
    risk = incident.get("risk", "at-risk")
    header = f"Инцидент: ТС {incident.get('tr_id')} · {format_delay(delay)}"
    if risk == "late":
        st.error(header)
    else:
        st.warning(header)
    details = st.columns(3)
    details[0].metric("Прогноз опоздания", format_delay(delay))
    details[1].metric("Прогноз прибытия", short_time(incident.get("predicted_arrival")))
    details[2].metric("Плановое прибытие", short_time(incident.get("target_time")))
    st.markdown(f"**Целевая остановка:** {html.escape(target_stop_text(incident))}")
    st.markdown(f"**Текущий участок:** {html.escape(segment_text(snapshot.get('current_segment')))}")
    st.markdown(f"**Участок до цели:** {html.escape(segment_text(snapshot.get('target_segment')))}")
    st.markdown(f"**Предполагаемая причина (эвристика):** {html.escape(str(incident.get('cause', '—')))}")
    position_text = "—"
    if position:
        speed = finite_number(position.get("speed"))
        speed_text = f"{speed:.1f} км/ч" if speed is not None else "—"
        position_text = (
            f"{finite_number(position.get('lat')):.6f}, {finite_number(position.get('lon')):.6f} · "
            f"{short_time(position.get('event_time'))} · {speed_text}"
        )
    st.markdown(f"**Текущая позиция:** {html.escape(position_text)}")
    st.markdown(f"**Рекомендация:** {html.escape(str(incident.get('recommendation', '—')))}")


try:
    status_data = load_status()
except (OSError, URLError, ValueError) as exc:
    st.error(f"Backend недоступен: {exc}")
    st.stop()

vehicles = status_data.get("vehicles", [])
map_data = status_data.get("map", {}) if isinstance(status_data.get("map"), dict) else {}
routes = map_data.get("routes", [])
positions = map_data.get("positions", [])
incident = map_data.get("incident")
vehicles_by_id = {vehicle.get("tr_id"): vehicle for vehicle in vehicles}
for position in positions:
    vehicle = vehicles_by_id.get(position.get("tr_id"), {})
    position["color"] = {"on-time": "#22c55e", "at-risk": "#eab308", "late": "#ef4444"}.get(
        position.get("risk", vehicle.get("risk", "on-time")), "#111827"
    )

columns = st.columns(4)
columns[0].metric("Время симуляции", short_time(status_data.get("simulated_time")))
columns[1].metric("Скорость", f"x{status_data.get('speed', speed)}")
columns[2].metric("Обработано точек", f"{status_data.get('processed_points', 0)} / {status_data.get('total_points', 0)}")
columns[3].metric("ТС в прогнозе", len(vehicles))

st.subheader("Маршрутная сеть и позиции ТС")
render_network(routes, positions, vehicles_by_id)

st.subheader("Карточка инцидента")
render_incident(incident)

st.subheader("Текущий риск")
if not vehicles:
    st.info("Прогнозы появятся после достижения симулятором времени T.")
else:
    table = pd.DataFrame(vehicles)
    table["prediction_s"] = table["prediction"].round(1)
    st.dataframe(
        table[["tr_id", "prediction_s", "target_class", "risk", "target_time_begin", "recommendation"]].rename(
            columns={
                "tr_id": "ТС",
                "prediction_s": "Задержка, с",
                "target_class": "Класс",
                "risk": "Риск",
                "target_time_begin": "Плановое прибытие",
                "recommendation": "Рекомендация",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    colors = {"on-time": "#dcfce7", "at-risk": "#fef3c7", "late": "#fee2e2"}
    labels = {"on-time": "в графике", "at-risk": "под риском", "late": "опоздание"}
    for vehicle in vehicles:
        risk = vehicle.get("risk", "on-time")
        color = colors.get(risk, "#f3f4f6")
        label = labels.get(risk, risk)
        delay = float(vehicle.get("prediction", 0.0))
        target = html.escape(str(vehicle.get("target_time_begin", "—")))
        recommendation = html.escape(str(vehicle.get("recommendation", "—")))
        st.markdown(
            f'<div style="background:{color};padding:10px 14px;border-radius:8px;margin:6px 0">'
            f'<b>ТС {html.escape(str(vehicle.get("tr_id")))}</b> · {delay:.1f} с · {label} · '
            f'цель {target}<br>{recommendation}</div>',
            unsafe_allow_html=True,
        )

st.caption(
    f"Поток: {status_data.get('processed_traffic_rows', 0)} / {status_data.get('total_traffic_rows', 0)} "
    "строк телеметрии. Обновление выполняется каждые 2 секунды."
)

if auto:
    time.sleep(2)
    st.rerun()
