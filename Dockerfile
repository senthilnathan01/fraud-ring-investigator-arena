FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TASK_ID=easy

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml __init__.py app.py client.py models.py eval.py inference.py /app/
COPY baselines /app/baselines
COPY server /app/server

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
