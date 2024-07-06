#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 32 \
--drop-path 0.05 \
--weight-decay 0.05 \
--epochs 300 \
--lr 1e-3 \
--input-size 448 \
--num_workers 25 \
--output_dir /fast/yangz16/outputs/Vim/vim_small_aa \
--no_amp \
> /fast/yangz16/outputs/Vim/vim_small_aa.out 2>&1 &