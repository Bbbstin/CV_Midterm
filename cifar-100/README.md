# Cifar-100

In this repo, we train several models using the dataset CIFAR-100.

## Usage

Run script, define log <filename> as you like.

```
nohup bash train.sh > ./log/<filename>.txt &
```

## The Structure of this Repo

```
├── README.md
├── augmentation: Three augmentation methods.
│   ├── __init__.py
│   ├── cutmix.py
│   ├── cutout.py
│   └── mixup.py
├── log: Log of training different models.
│   ├── densenet121.txt
│   ├── dla.txt
│   ├── dpn92.txt
│   ├── preactresnet50.txt
│   ├── resnet18-cutmix.txt
│   ├── resnet18-cutout.txt
│   ├── resnet18-mixup.txt
│   ├── resnet18.txt
│   ├── resnet50.txt
│   ├── resnext29_32_8d.txt
│   ├── resnext50_32_4d.txt
│   └── wrn.txt
├── main.py: Call this to train the model.
├── models: Models
│   ├── __init__.py
│   ├── densenet.py
│   ├── dla.py
│   ├── dla_simple.py
│   ├── dpn.py
│   ├── preact_resnet.py
│   ├── resnet.py
│   ├── resnext.py
│   └── wide_resnet.py
└── train.sh: A shell script to train the model
```

## Results

| model     | augmentation | top-1 accuracy (%) | top-5 accuracy (%) |
| --------- | ------------ | ------------------ | ------------------ |
| ResNet-18 | baseline     | 77.55              | 93.53              |
|           | cutmix       | 79.47              | 94.47              |
|           | cutout       | 77.78              | 93.51              |
|           | mixup        | 79.53              | 94.22              |

And the following models are trained with cutmix.

| model                    | top-1 accuracy (%) | top-5 accuracy (%) |
| ------------------------ | ------------------ | ------------------ |
| ResNet-50                | 81.30              | 95.07              |
| PreActResNet-50          | 82.11              | 95.36              |
| **WideResNet-28**        | **83.01**          | **95.70**          |
| ResNeXt-29, 32$\times$8d | 80.35              | 94.35              |
| ResNeXt-50, 32$\times$4d | 80.83              | 94.79              |
| DenseNet-121             | 81.57              | 94.63              |
| DPN-92                   | 81.11              | 94.38              |
| DLA-34                   | 82.07              | 95.11              |

## Final

WideResNet-28:

top-1 accuracy: 83.01%

top-5 accuracy: 95.70%