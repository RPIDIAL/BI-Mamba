#!/bin/bash

<<<<<<< HEAD
#!/bin/bash

# Updated paths to Linux format for Colab
python /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 2 \
--num_workers 2 \
--eval \
--input-size 224 \
--resume /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/scripts/OUTPUT/checkpoint.pth \
--output_dir /content/drive/MyDrive/BI-Mamba-main/mamba-cxr/scripts/OUTPUT \
--data-set NLSTDual
=======
python main.py \
--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
--batch-size 2 \
--num_workers 25 \
--eval \
--input-size 448 \
--resume /fast/yangz16/outputs/Vim/vim_small_aa/checkpoint.pth \
--output_dir /fast/yangz16/outputs/Vim/vim_small_aa_eval \
>>>>>>> upstream/main
