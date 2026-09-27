#!/bin/sh
# Retrain, then export the artifacts the ML service image must ship.
#
# The separation this script exists to make explicit:
#
#   training  ->  artifacts on disk  ->  baked into the ml-service image
#
# Nothing downstream reads the dataset.  The service image gets exactly the
# files this script copies out, and the service is what serves inference.
set -eu

DATASET_DIR="${DATASET_DIR:-/app/dataset}"
ARTIFACT_OUT="${ARTIFACT_OUT:-/app/out}"

say() { printf '[train] %s\n' "$*"; }

# Fail with an actionable message instead of a pandas FileNotFoundError 400 lines
# deep. The split is not in the repository, so "you forgot to mount it" is by far
# the most likely cause.
for required in \
    "${DATASET_DIR}/labels/labels_train.csv" \
    "${DATASET_DIR}/labels/labels_test.csv" \
    "${DATASET_DIR}/train/traffic.csv" \
    "${DATASET_DIR}/test/traffic.csv" \
    "${DATASET_DIR}/validate/points.csv"
do
    if [ ! -f "${required}" ]; then
        echo "[train] ERROR: ${required} is missing." >&2
        echo "[train] The train/test/labels split is not committed (.gitignore)." >&2
        echo "[train] Mount the competition dataset read-only, e.g." >&2
        echo "[train]   docker run --rm -v \"\$(pwd)/dataset:/app/dataset:ro\" ..." >&2
        exit 1
    fi
done

say "dataset looks complete under ${DATASET_DIR}"
say "training (ml/baseline_v2.py) ..."
python ml/baseline_v2.py

mkdir -p "${ARTIFACT_OUT}"

# model.cbm and features.json are the service's two required artifacts.
# prob_cal.json is produced by ml/calibrate.py; copy it when present so the
# service keeps its calibrated p_late, and say so plainly when it is not, rather
# than shipping an image whose p_late will silently be None.
for artifact in model.cbm features.json; do
    if [ ! -s "ml/${artifact}" ]; then
        echo "[train] ERROR: training did not produce ml/${artifact}" >&2
        exit 1
    fi
    cp "ml/${artifact}" "${ARTIFACT_OUT}/"
    say "exported ${artifact} ($(wc -c < "${ARTIFACT_OUT}/${artifact}") bytes)"
done

if [ -s ml/prob_cal.json ]; then
    cp ml/prob_cal.json "${ARTIFACT_OUT}/"
    say "exported prob_cal.json ($(wc -c < "${ARTIFACT_OUT}/prob_cal.json") bytes)"
else
    say "WARNING: ml/prob_cal.json not found."
    say "         Run 'python ml/calibrate.py' for calibrated p_late, otherwise the"
    say "         service will report p_late = null for every prediction."
fi

if [ -f submission.csv ]; then
    cp submission.csv "${ARTIFACT_OUT}/"
    say "exported submission.csv"
fi

cat <<'NEXT'

[train] Done. Artifacts are in the mounted output directory.

        Rebuild the ML service so it ships them:

            docker compose build ml && docker compose up -d

        Then confirm the service is serving the new model:

            curl -s http://localhost:8000/health | grep -o '"features":[0-9]*'
NEXT
