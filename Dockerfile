FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FOOTBALL_DATABASE=/data/platform.sqlite3
WORKDIR /app
COPY requirements.lock requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --uid 10001 --create-home football && mkdir /data && chown football:football /data
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health',timeout=3)"
CMD ["football-analytics", "serve", "--host", "0.0.0.0", "--database", "/data/platform.sqlite3"]
