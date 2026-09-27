# Backend: API, orchestration, the NDTP receiver and the replay stream.
#
# Self-containment contract
# -------------------------
# This image contains NO model and NO ML artifact.  Not model.cbm, not
# features.json, not prob_cal.json, and not src/predictor.py.  Inference lives in
# the `ml` service, which this container reaches at http://ml:8001.
#
# That is enforced three ways, so no single mistake can reintroduce the coupling:
#   1. no catboost in backend/requirements.txt  -> an accidental import fails;
#   2. src/predictor.py is not copied            -> the feature builder is absent;
#   3. no bind mount of the repository in compose -> host ML files stay invisible.
#
# Only the validate slice is baked in, because /stream/status replays it and
# /generate_submission needs its points.  The train/test/labels split is not
# committed and is not needed at runtime -- it only matters to the training image
# (ml/Dockerfile), which is a separate concern on purpose.
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /app/backend

# The ML-free domain modules, one COPY each so the allowlist is explicit and a
# new ML module cannot be swept in by copying the whole directory.
COPY ./src/schedule_index.py /app/src/schedule_index.py
COPY ./src/patterns.py       /app/src/patterns.py
COPY ./src/deviation.py      /app/src/deviation.py
COPY ./src/ml_client.py      /app/src/ml_client.py
COPY ./src/ndtp.py           /app/src/ndtp.py
COPY ./src/ndtp_server.py    /app/src/ndtp_server.py
COPY ./src/cascade.py        /app/src/cascade.py
COPY ./src/batching.py       /app/src/batching.py
COPY ./src/runtime.py        /app/src/runtime.py

# The submission format contract, shared with the ML-side tools so the HTTP path
# and the offline path cannot drift. Plain pandas: no CatBoost, no model code.
COPY ./ml/submission.py     /app/ml/submission.py

# Timetable + the validate slice the replay stream and /generate_submission read.
COPY ./dataset/validate /app/dataset/validate

# Fail the build, not the deployment, if the ML boundary is ever crossed:
# the feature builder must be absent, and the shared submission module must
# import without pulling CatBoost in.
RUN python -c "import pathlib,sys; bad=[p.name for p in pathlib.Path('/app/src').glob('*.py') if p.name in {'predictor.py'}]; sys.exit('ML module present in backend image: '+str(bad)) if bad else None"
RUN python -c "import sys; import ml.submission; sys.exit('shared submission module pulled in '+str(sorted(m for m in sys.modules if m.split('.')[0] in ('catboost','sklearn')))) if {'catboost','sklearn'} & {m.split('.')[0] for m in sys.modules} else None"

ENV PYTHONPATH=/app \
    DATASET_DIR=/app/dataset \
    ML_SERVICE_URL=http://ml:8001 \
    SUBMISSION_PATH=/app/out/submission.csv

RUN mkdir -p /app/out

EXPOSE 8000 9201

# Proves the API answers, not that the ML service is up: the backend must stay
# healthy while the ML service is down, otherwise Docker would restart it in a
# loop and destroy the degraded-but-serving state.  ML reachability is reported
# by GET /health as status="degraded" and by /metrics, not by this check.
HEALTHCHECK --interval=10s --timeout=6s --start-period=25s --retries=6 \
    CMD python -c "import json,urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); raise SystemExit(0 if h.get('status') in ('ok','degraded') else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
