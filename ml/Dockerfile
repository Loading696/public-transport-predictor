# Offline TRAINING image -- retrain, then hand the artifacts to the ML service.
#
# This is NOT the ML service.  The service image is ml-service/Dockerfile and
# runs as the `ml` service in compose.  This one is a batch job: it runs
# baseline_v2.py once, retrains CatBoost, and writes model.cbm / features.json
# / submission.csv.  It is not started by `docker compose up` and has no
# healthcheck, because it is not a server.
#
# The pipeline is deliberately one-directional:
#
#     training artifacts  ->  saved  ->  baked into the ML service image
#     (this image)                        (docker compose build ml)
#
# The train/test/labels split is ~large and is not committed (.gitignore), so it
# is mounted at run time.  It is never baked into any image: the service does not
# need it, and the model is what leaves the training step.
#
#   docker build -f ml/Dockerfile -t predictor-train .
#   mkdir -p out
#   docker run --rm \
#     -v "$(pwd)/dataset:/app/dataset:ro" \
#     -v "$(pwd)/out:/app/out" \
#     predictor-train
#
#   # then rebuild the service so it ships the new artifacts:
#   docker compose build ml && docker compose up -d
FROM python:3.12-slim

WORKDIR /app

# Training dependencies only. No fastapi/uvicorn/httpx: nothing here serves HTTP,
# and keeping the server stack out makes it obvious this image is a batch job.
COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# src/ is required: baseline_v2.py builds features through src/predictor.
COPY ./src /app/src

# Training scripts only. The shipped .cbm/.json artifacts are deliberately not
# copied: this image consumes the dataset and produces artifacts, so carrying a
# stale pre-built model into it would invite shipping the old one by accident.
RUN mkdir -p /app/ml /app/out
COPY ./ml/baseline_v2.py     /app/ml/baseline_v2.py
COPY ./ml/calibrate.py      /app/ml/calibrate.py
COPY ./ml/baseline.py       /app/ml/baseline.py
COPY ./ml/eval_patterns.py  /app/ml/eval_patterns.py
COPY ./ml/generate_submission.py /app/ml/generate_submission.py
COPY ./ml/sanity_check.py   /app/ml/sanity_check.py
COPY ./ml/profile_inference.py /app/ml/profile_inference.py
COPY ./ml/ml_pipeline.py     /app/ml/ml_pipeline.py
COPY ./ml/train_and_export.sh /app/ml/train_and_export.sh
RUN chmod +x /app/ml/train_and_export.sh

ENV PYTHONPATH=/app \
    ARTIFACT_OUT=/app/out

ENTRYPOINT ["/app/ml/train_and_export.sh"]
