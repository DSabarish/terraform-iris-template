# ──────────────────────────────────────────────────────────────
# Iris ML API — Production Dockerfile
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and enables stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ──────────────────────────────────────────────────────────────
# Dependencies layer (cached separately from source code)
# ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Application source
# ──────────────────────────────────────────────────────────────
COPY src/ .

# ──────────────────────────────────────────────────────────────
# Non-root user for security
# ──────────────────────────────────────────────────────────────
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
EXPOSE 8080

# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
