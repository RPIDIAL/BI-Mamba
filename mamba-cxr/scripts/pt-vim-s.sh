#!/bin/bash
<<<<<<< HEAD

# 1. Use modern torchrun for single GPU (nproc_per_node=1)
torchrun --nproc_per_node=1 --master_port=29500 /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 8 \
=======
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 32 \
>>>>>>> upstream/main
--drop-path 0.05 \
--weight-decay 0.05 \
--epochs 300 \
--lr 1e-3 \
<<<<<<< HEAD
--input-size 224 \
--num_workers 2 \
--output_dir /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/scripts/OUTPUT \
--data-set NLSTDual \
--no_amp \
--clip-grad 1.0 \
--no-repeated-aug
=======
--input-size 448 \
--num_workers 25 \
--output_dir /fast/yangz16/outputs/Vim/vim_small_aa \
--no_amp \
> /fast/yangz16/outputs/Vim/vim_small_aa.out 2>&1 &
>>>>>>> upstream/main
