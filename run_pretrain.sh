#!/usr/bin/env bash

DATASET=$1
MODEL=$2
GPU=$3
#EMBED_SIZE=10
echo GPU ${GPU}

N_LAYERS=2
for EMBED_SIZE in 40
do

  L2=1e-4
  SAVE_DIR=results/${DATASET}/${MODEL}/${EMBED_SIZE}_${N_LAYERS}l/tie/l2${L2}
  echo ${SAVE_DIR}
  mkdir -p ${SAVE_DIR}
  python pretrain.py -g ${GPU} -d data/${DATASET}/ -m ${MODEL} -e ${EMBED_SIZE} -mf ${SAVE_DIR} --n_layers ${N_LAYERS} --l2 ${L2} --tie t
done
