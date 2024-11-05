#!/usr/bin/env bash
GPU_ID=$1
# trg_task=$2

idx=0
for task in $(seq 0 6)
do
    time CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
    --val_list="list/MOTS/MOTS_test_modify_v2.txt" \
    --reload_from_checkpoint=True \
    --target_task=$task \
    --reload_path="snapshots/multi_net_bn_init_32ch_batch8_2000e_task$task/MOTS_multihead_snapshots_e1000.pth" \
    --save_path="outputs_multi_net_bn_init_32ch_batch8_1000e_no_lr_decay/" \
    --input_size="64,128,128" \
    --batch_size=1 \
    --num_gpus=1 \
    --num_workers=2 \
    --FP16=True \
    --ensemble=False

    sleep 5
    idx=$(($idx+1))
done

time python postp.py --img_folder_path='./outputs_multi_net_bn_init_32ch_batch8_1000e_no_lr_decay/'

# MOTS_multihead_snapshots_final_e1999.pth" \
# MOTS_multihead_snapshots_e1000.pth
