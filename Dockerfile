FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ZHIWEI_HOST=0.0.0.0 \
    ZHIWEI_PORT=8000 \
    ZHIWEI_DATA_DIR=/app/data

WORKDIR /app

# 先把依赖单独装一层：改代码不会让依赖层失效
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
