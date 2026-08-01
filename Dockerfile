# Peddapalli Accident Prediction API
FROM python:3.12-slim

WORKDIR /app

# build-essential: only needed if a dependency has no prebuilt wheel for
# this platform; kept small (slim base + no cache) to keep image size and
# Render cold-start time down.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} (set dynamically by Render) is expanded at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
