#!/bin/bash
CUDA_VISIBLE_DEVICES=1,3,4,5,6

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 16 \
--lr 5e-6 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--drop-path 0.0 \
--weight-decay 1e-8 \
--num_workers 25 \
--output_dir /fast/yangz16/outputs/Vim/vim_small_ft \
--epochs 30 \
--finetune /fast/yangz16/outputs/Vim/vim_s_midclstok_ft_81p6acc.pth \
--no_amp
