#!/bin/bash
# python PATH
cd ..

gpus='6'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'airsreconstruction'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net_memory.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --object_detection 0 \
    --print_freq 20 \
    --t_length 3 \
    --interval 1 \
    --visualize 0 \
    --visualize_input 0 \
    -b 8  \
    --test_batch_size 8 \
    --workers_test 4 \
    --h 256 \
    --w 256 \
    --arch 'Unet_Free' \
    --dataset_type 'airs_anomaly' \
    --label 1 \
    --training_folder 'training/frames' \
    --testing_folder 'testing/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 2 \
    --epochs 40 \
    --is_amp 1 \
    --optim 'adamW' \
    #--evaluate_time 0 \
    #--evaluate \
    #--demo 'video|/data/miaobo/ucf/video/Explosion013_x264.mp4' \

