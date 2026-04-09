FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TASK_ID=task1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml client.py models.py eval.py inference.py /app/
COPY baselines /app/baselines
COPY server /app/server

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
