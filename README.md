# ACALF
Pytorch implementation for "Few-Shot Hard Sample Segmentation: Bridging the Gap for Real-World Challenges"
![image](https://github.com/user-attachments/assets/922b30d4-d6d9-47df-8009-644e092c532e)
# Requirements
detectron2==0.6

fvcore==0.1.5

matplotlib==3.8.0

numpy==1.24.1

opencv_python==4.8.0.76

Pillow==10.4.0

tensorboardX==2.6.2

timm==0.9.5

torch==2.0.1+cu118

torchvision==0.15.2+cu118
# Pretrained Weights
ACALF checkpoints are available. Put backbone in `/pretrained` and put checkpoints in `/checkpoints`. 

MIoU of 5-way 1-shot, 5-way 5-shot and 5-way 10-shot shown in the table is evaluated separately on each dataset.


| dataset | 1-shot | 5-shot | 10-shot |
| ------- | ------ | ------ | ------- |
| Road crack |8.15 |10.22 | 11.67 |
| Steel defect |10.44 |16.66 |24.05 |
| Leaf diseases|24.57 |29.03 |30.95 |
| Animal |52.85 |56.29 |58.79 |
| Eyeballs | 13.04| 13.71| 13.85|
| Polyp | 21.93| 23.91| 23.89|
| Lunar terrain |13.04|14.57 |16.16 | 
| City atellite |9.91|10.78 |11.38 | 

Link to checkpoints: 

# Datasets 
We provide evaluation datasets in the link below. 

Link to dataset: [pwd: phcu](https://pan.baidu.com/s/1KpFcpuEmta7Vb8Xruyz3qA)

# Training
For ResNet50, run:
~~~python
#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22058 train.py \
        --bsz 20 \
        --nepoch 200 \
        --feature_extractor_path path_to_backbone \
        --backbone resnet50 \
        --lr 1e-4 \
        --benchmark 'fss' \
        --datapath path_to_data \
        --num_queries 15  \
        --dec_layers 1  \
        --fold 0 \
        --test_num 1000
~~~

For ResNet101, run:
~~~python
#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22058 train.py \
        --bsz 20 \
        --nepoch 200 \
        --feature_extractor_path path_to_backbone \
        --backbone resnet101 \
        --lr 1e-4 \
        --benchmark 'fss' \
        --datapath path_to_data \
        --num_queries 50  \
        --dec_layers 1  \
        --fold 0 \
        --test_num 1000
~~~

For Swin Transformer, run:
~~~python
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22058 train.py \
        --bsz 20 \
        --nepoch 200 \
        --feature_extractor_path path_to_backbone \
        --backbone swin-l \
        --lr 1e-4 \
        --benchmark 'fss' \
        --datapath path_to_data \
        --num_queries 15  \
        --dec_layers 3  \
        --fold 0 \
        --test_num 1000
~~~

# Evaluation
For ResNet50, run:
~~~python
#!/bin/bash
CUDA_VISIBLE_DEVICES={gpu_id} python test.py --nshot {1/5/20} --test_dataset dataset --{vote/post_average/pre_average} --bsz 1  --test_num 1000  --test_epoch 5 --load path_to_checkpoints --num_queries 15 --dec_layer 1 --backbone resnet50 
~~~

For ResNet101, run:
~~~python
#!/bin/bash
CUDA_VISIBLE_DEVICES={gpu_id} python test.py --nshot {1/5/20} --test_dataset dataset --{vote/post_average/pre_average} --bsz 1  --test_num 1000  --test_epoch 5 --load path_to_checkpoints --num_queries 50 --dec_layer 1 --backbone resnet101
~~~

For Swin Transformer, run:
~~~python
#!/bin/bash
CUDA_VISIBLE_DEVICES={gpu_id} python test.py --nshot {1/5/20} --test_dataset dataset --{vote/post_average/pre_average} --bsz 1  --test_num 1000  --test_epoch 5 --load path_to_checkpoints --num_queries 15 --dec_layer 3 --backbone swin-l 
~~~

# LICENSE
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file.


