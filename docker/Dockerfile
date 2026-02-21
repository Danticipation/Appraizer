FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]

