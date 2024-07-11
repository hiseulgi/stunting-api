# usage: docker build -f Dockerfile -t hiseulgi/stunting-streamlit:latest .
FROM python:3.11.5-slim

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt

# set working directory
WORKDIR /app

# download model if not exists

EXPOSE 8501

COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run", "src/app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0" ]