#!/usr/bin/env bash
GPU_ID=$1

idx=1
for task in $(seq 0 6)
do
    time CUDA_VISIBLE_DEVICES=$idx python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=$RANDOM train.py \
        --train_list="list/MOTS/MOTS_train_modify_v2.txt" \
        --snapshot_dir="snapshots/multi_net_bn_init_32ch_batch8_2000e_task$task" \
        --target_task=$task \
        --input_size="64,128,128" \
        --batch_size=8 \
        --num_gpus=1 \
        --num_epochs=2000 \
        --start_epoch=0 \
        --learning_rate=1e-2 \
        --num_classes=2 \
        --num_workers=8 \
        --weight_std=True \
        --random_mirror=True \
        --random_scale=True \
        --FP16=True &

    sleep 5
    idx=$(($idx+1))
done


# if [ $3 = 0 ]
# then
#     gpu_list=(0 1 2 3)
# else
#     gpu_list=(4 5 6 7)
# fi
# | tee logs/"$DATE-$data-w$WAY-s$SHOT-lr$LR-part$center-F$FOLD".txt &