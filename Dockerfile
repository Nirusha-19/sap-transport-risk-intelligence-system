FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY . .

RUN git lfs pull 2>/dev/null || true

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app_firebase.py", "--server.port=8501", "--server.address=0.0.0.0"]
