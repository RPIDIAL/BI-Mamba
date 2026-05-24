#!/bin/bash
<<<<<<< HEAD
# Use only 1 GPU for Colab
export CUDA_VISIBLE_DEVICES=0
export USE_LIBUV=0

# Use torchrun for compatibility with Python 3.12
# Added --master_port to avoid the 'Address already in use' error
#!/bin/bash
#!/bin/bash
torchrun --nproc_per_node=1 --master_port=29505 /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 2 \
=======
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

nohup python -m torch.distributed.launch --nproc_per_node=5 --use_env main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 24 \
>>>>>>> upstream/main
--lr 5e-6 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--drop-path 0.0 \
--weight-decay 1e-8 \
<<<<<<< HEAD
--num_workers 1 \
--output_dir /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/scripts/OUTPUT \
--epochs 30 \
--finetune /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/scripts/vim_s_midclstok_ft_81p6acc.pth \
--no_amp \
--input-size 224 \
--data-set NLSTDual \
--clip-grad 1.0 \
--no-repeated-aug > vim_small_s16_224.out 2>&1
=======
--num_workers 25 \
--output_dir /fast/yangz16/outputs/Vim/vim_small_s16_224 \
--epochs 30 \
--finetune /fast/yangz16/outputs/Vim/vim_s_midclstok_ft_81p6acc.pth \
--no_amp \
--input-size 224 \
--data-set NLSTDual \
> /fast/yangz16/outputs/Vim/vim_small_s16_224.out 2>&1 &
>>>>>>> upstream/main
