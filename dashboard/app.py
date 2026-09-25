from __future__ import annotations

import html
import json
import os
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

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
st.markdown('<meta http-equiv="refresh" content="2">', unsafe_allow_html=True)


def load_status() -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


try:
    status_data = load_status()
except (OSError, URLError, ValueError) as exc:
    st.error(f"Backend недоступен: {exc}")
    st.stop()

vehicles = status_data.get("vehicles", [])

columns = st.columns(4)
columns[0].metric("Время симуляции", str(status_data.get("simulated_time", "—"))[:19].replace("T", " "))
columns[1].metric("Скорость", f"x{status_data.get('speed', speed)}")
columns[2].metric("Обработано точек", f"{status_data.get('processed_points', 0)} / {status_data.get('total_points', 0)}")
columns[3].metric("ТС в прогнозе", len(vehicles))

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
