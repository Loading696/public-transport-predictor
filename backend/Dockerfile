FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend /app/backend
COPY ./src /app/src
COPY ./ml/model.cbm /app/ml/model.cbm
COPY ./ml/features.json /app/ml/features.json
COPY ./dataset/validate /app/dataset/validate

ENV PYTHONPATH=/app
EXPOSE 8000 9201

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
