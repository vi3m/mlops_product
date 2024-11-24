FROM python:3.10.15-slim

WORKDIR /app

RUN pip install mlflow
RUN pip install boto3 psycopg2-binary
RUN pip install -U ray[data,train,tune,serve]


COPY . .
EXPOSE 8000

# Run the command to start the Ray Serve app
CMD ["python", "serve.py"]
