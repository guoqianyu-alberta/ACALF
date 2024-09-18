#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22058 train.py \
        --bsz 20 \
        --nepoch 200 \
        --feature_extractor_path '' \
        --backbone '' \
        --lr 1e-4 \
        --benchmark 'fss' \
        --datapath '' \
        --num_queries 15  \
        --dec_layers 3  \
        --fold 0 \
        --test_num 1000 

