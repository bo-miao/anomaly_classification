#!/bin/bash
# python PATH
cd ..

gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'classifier_ucf'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --object_detection 0 \
    --print_freq 20 \
    --t_length 3 \
    --interval 1 \
    --visualize 0 \
    --visualize_input 0 \
    -b 32  \
    --test_batch_size 32 \
    --workers_test 4 \
    --h 256 \
    --w 256 \
    --discriminator '' \
    --arch 'Unet_Free' \
    --encoder_arch 'Encoder_Free' \
    --decoder_arch 'Decoder_Free' \
    --dataset_type 'ucf' \
    --label 1 \
    --training_folder 'training_simple/frames' \
    --testing_folder 'testing_simple/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 2 \
    --epochs 60 \
    --is_amp 1 \
    --optim 'adamW' \
    --evaluate \
    --demo 'video|/data/miaobo/ucf/video/Explosion013_x264.mp4' \
    #--resume '/home/miaobo/project/anomaly_demo2/ckpt/best_27randomnoise_ucf0,1_Unet_Free_Adversarial__ucf_checkpoint.pth.tar' \

