#!/bin/bash

## Switch conda environment
eval "$(conda shell.bash hook)"
conda activate dl

## Toy Case
CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=256 --lr=0.1 --max-epoch=10

## Baseline
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=256 --lr=0.1 --max-epoch=200
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=256 --lr=0.1 --max-epoch=200 --use-cutout
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=256 --lr=0.1 --max-epoch=200 --use-mixup
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet18 --batch-size=256 --lr=0.1 --max-epoch=200 --use-cutmix

## Network Architecture
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNet50 --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=PreActResNet50 --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=WideResNet28x10 --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt29_16x64d --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=ResNeXt50_32x4d --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DenseNet201 --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DPN92 --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
# CUDA_VISIBLE_DEVICES=0 python main.py --model=DLA --batch-size=256 --lr=0.1 --max-epoch=300 --use-cutmix
