#!/bin/bash
# python PATH
cd ..

gpus='8'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'resnet50'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net_classification.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --object_detection 0 \
    --print_freq 20 \
    --t_length 5 \
    --interval 2 \
    --visualize 0 \
    --visualize_input 0 \
    -b 32  \
    --test_batch_size 32 \
    --workers_test 4 \
    --h 256 \
    --w 256 \
    --arch 'resnet50' \
    --dataset_type 'ucf' \
    --label 1 \
    --training_folder 'training_simple/frames' \
    --testing_folder 'testing_simple/frames' \
    --label_folder 'label' \
    --total_class 'Normal|Arson|Explosion|Fall|Fighting' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 2 \
    --epochs 50 \
    --is_amp 1 \
    --optim 'sgd' \
    --lr 0.1 \
    --lr_mode 'step' \
    --evaluate_time 0 \
    #--evaluate \
    #--demo 'video|/data/miaobo/script/video/Fall_005.mp4' \
    #--resume '/home/miaobo/project/anomaly_demo2/ckpt/best_30airsclassification2,3_resnet50__airs_anomaly2_checkpoint.pth.tar' \

# max best_30airsclassification2,3_resnet50__airs_anomaly2_checkpoint.pth.tar

