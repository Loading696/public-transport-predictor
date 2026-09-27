# ML service: feature engineering, model loading and CatBoost inference.
#
# This is the only image that contains CatBoost.  The backend image deliberately
# does not, so "the backend must not run inference" is enforced by what is
# installed, not only by what is imported.
#
#   docker build -f ml-service/Dockerfile -t predictor-ml .
#
# In compose this is started as the `ml` service and reached by container name
# (http://ml:8001); the backend never dials localhost for it.
FROM python:3.12-slim

WORKDIR /app

COPY ml-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The ML side of the domain code: the feature builder (which loads CatBoost),
# the pattern detectors, and the shared timetable index both services read.
COPY ./src /app/src
COPY ./ml-service/service.py /app/service.py
# Artifacts: the trained model, the feature order, the calibration curve.
COPY ./ml/model.cbm /app/ml/model.cbm
COPY ./ml/features.json /app/ml/features.json
COPY ./ml/prob_cal.json /app/ml/prob_cal.json
# The plan is read at start-up for the static timetable index.  The
# train/test/labels split is not committed (.gitignore) and is only needed to
# retrain, which is what the separate ml/Dockerfile training image is for.
COPY ./dataset/validate /app/dataset/validate

ENV PYTHONPATH=/app \
    MODEL_PATH=/app/ml/model.cbm \
    FEATURES_PATH=/app/ml/features.json \
    SCHEDULE_PATH=/app/dataset/validate/schedule_plan.csv \
    ML_PORT=8001

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=6s --start-period=40s --retries=12 CMD python -c "import json,os,sys,urllib.request; url='http://127.0.0.1:'+os.environ.get('ML_PORT','8001')+'/health'; sys.exit(0 if json.load(urllib.request.urlopen(url, timeout=4)).get('status')=='ok' else 1)"

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
