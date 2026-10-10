FROM ghcr.io/mlflow/mlflow:v2.17.2
# the stock image has no Postgres driver or S3 client
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.35.60
