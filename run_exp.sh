#!/usr/bin/env bash

# change the following parameters to desired one.
FLR=1e-6
L2=1e-5
DATASET=$1
ENCODER=$2
GPU=$3

# pick the desired sentiment dataset

OUTPUT_DIR=data/${DATASET}/${ENCODER}/${DATASET}_flr${FLR}_l2${L2}
BASE_SENTIMENT_PATH=results/sentiment/${DATASET}_${ENCODER}_flr${FLR}_l2${L2}/

##
SAVE_NAME=${OUTPUT_DIR}_kfold/splits.pkl
    echo
    echo Creating Dataset in ${SAVE_NAME}
    echo
    python scripts/preprocess_conv_data.py -redial data/${SENT_DATASET}_dataset/ \
        -movie_map data/${DATASET}/movie_map.csv \
        -movie_plot data/${DATASET}/movie_plot.csv \
        -redial data/${DATASET}_dataset/\
        -g ${GPU} -enc ${BASE_SENTIMENT_PATH} \
        -o ${SAVE_NAME}


# =============================================================================
# Create test.pkl
# =============================================================================

SAVE_NAME=${OUTPUT_DIR}/test.pkl
    echo
    echo Creating Testing Dataset in ${SAVE_NAME}
    echo
    python scripts/preprocess_conv_data.py -redial data/${DATASET}_dataset/ \
    -movie_map data/${DATASET}/movie_map.csv \
    -movie_plot data/${DATASET}/movie_plot.csv \
    -redial data/${DATASET}_dataset/\
    -g ${GPU} -enc ${BASE_SENTIMENT_PATH}/ \
    -o ${OUTPUT_DIR}/test.pkl -test t
