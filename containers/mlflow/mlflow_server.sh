#!/bin/bash
mlflow server \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_STORE_URI \
    --host 0.0.0.0
