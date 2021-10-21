#!/usr/bin/env bash

DATASET=$1
ENCODER=$2
GPU=$3
itercnt=15
flr=1e-6
l2=1e-5
for i in 0 1 2 3 4 
do

  SAVE_DIR=results/sentiment/${DATASET}_${ENCODER}_flr${flr}_l2${l2}/${i}
  echo ${SAVE_DIR}
  mkdir -p ${SAVE_DIR}
  python util/sentiment.py -i ${itercnt} -d data/sentiment/${DATASET}/${i} -enc ${ENCODER} -g ${GPU} -flr ${flr} --l2 ${l2} -m ${SAVE_DIR}

done
