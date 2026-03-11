#!/bin/bash

SUPREM_DIR=$HOME/my_documents/fair_segmentation/repo/SuPreM
INFERENCE_DIR=$SUPREM_DIR/direct_inference


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 

# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --rdzv_backend=c10d $INFERENCE_DIR/inference.py \
# python $INFERENCE_DIR/inference_logits.py \
CUDA_VISIBLE_DEVICES=5 python $INFERENCE_DIR/inference_logits.py \
    --save_dir results \
    --checkpoint $INFERENCE_DIR/pretrained_checkpoints/supervised_suprem_unet_2100.pth \
    --data_root_path /export/gaon1/data/jteneggi/TotalSegmentator \
    --target_file ct \
    --a_min 0 \
    --a_max 1 \
    --b_min 0 \
    --b_max 1 \
    --space_x 1.5 \
    --space_y 1.5 \
    --space_z 1.5 \
    --store_result \
    --suprem 
    # --dist
