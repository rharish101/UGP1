#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/MAD_SRGAN/ \
    --summary_dir ./result/MAD_SRGAN/log/ \
    --mode inference \
    --is_training False \
    --task MAD_SRGAN \
    --input_dir_LR ./data/mytests/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./experiment_MAD_SRGAN_VGG54/multi_disc/model-160000

