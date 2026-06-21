FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ci_cd/fastapi/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/backend /app/backend
COPY vacancies.json /app/vacancies.json

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "alembic -c /app/backend/alembic.ini upgrade head && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/backend"]
