#!/bin/bash

python main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 2 \
--num_workers 25 \
--eval \
--input-size 448 \
--resume /fast/yangz16/outputs/Vim/vim_small_aa/checkpoint.pth \
--output_dir /fast/yangz16/outputs/Vim/vim_small_aa_eval \
