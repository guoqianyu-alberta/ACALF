#!/bin/bash
# test_dataset=('Animal' 'Artificial_Luna_Landscape' 'ClinicDB' 'Crack_Detection' 'Eyeballs' 'Magnetic_tile_surface' 'Leaf_Disease' 'Aerial')
CUDA_VISIBLE_DEVICES=4 python test.py \
                        --nshot 1 \
                        --test_dataset Animal \
                        --vote \
                        --bsz 1 \
                        --test_num 10 \
                        --test_epoch 1 \
                        --load 'checkpoints/res50.pt' \
                        --num_queries 15 \
                        --dec_layer 1 \
                        --backbone 'resnet50' \
                        --feature_extractor_path 'pretrained_model/resnet50.pth'  \
                        --test_datapath ''  \



