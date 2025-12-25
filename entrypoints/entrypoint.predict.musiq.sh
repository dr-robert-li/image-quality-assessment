#!/bin/bash
set -e

MODEL_PATH=$1
IMAGE_SOURCE=$2
PREDICTIONS_FILE=$3

# predict with MUSIQ
if [ -z "$PREDICTIONS_FILE" ]; then
    python -m evaluater.predict_musiq \
    --model-path $MODEL_PATH \
    --image-source $IMAGE_SOURCE
else
    python -m evaluater.predict_musiq \
    --model-path $MODEL_PATH \
    --image-source $IMAGE_SOURCE \
    --predictions-file $PREDICTIONS_FILE
fi
