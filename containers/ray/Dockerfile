FROM python:3.9-slim

WORKDIR /app

RUN pip install -U ray[data,train,tune,serve]
RUN pip install mlflow


COPY . .
EXPOSE 8000

# Run the command to start the Ray Serve app
CMD ["python", "serve.py"]
