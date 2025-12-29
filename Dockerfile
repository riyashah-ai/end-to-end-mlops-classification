FROM python:3.10-slim

WORKDIR /app

COPY requirenments.txt .

RUN pip install --no-cache-dir -r requirenments.txt

COPY src/ ./src/
COPY configs/ ./configs

# ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

ENTRYPOINT ["python", "-m" , "src.training.train"]

