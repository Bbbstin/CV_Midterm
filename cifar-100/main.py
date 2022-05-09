import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T

from matplotlib import pyplot as plt

import os
import argparse

import models
import augmentation as aug


parser = argparse.ArgumentParser(description="PyTorch CIFAR100 Training")
parser.add_argument(
    '--resume', '-r', action='store_true', help="Resume from checkpoint"
)
parser.add_argument(
    '--model',
    default='ResNet18',
    type=str,
    help="ResNet18, ResNet50, PreActResNet50, ResNeXt50_32x4d, \
        WideResNet28x10, DenseNet201, DPN92, DLA",
)
parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
parser.add_argument('--lr', default=0.1, type=float, help="Learning rate")
parser.add_argument('--max-epoch', default=200, type=int, help="Max training epochs")
parser.add_argument('--use-cutout', action='store_true', help="Use cutout augmentation")
parser.add_argument('--use-mixup', action='store_true', help="Use mixup augmentation")
parser.add_argument('--use-cutmix', action='store_true', help="Use cutmix augmentation")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # Best test accuracy
best_acc_5 = 0  # Best top-5 test accuracy
start_epoch = 0  # Start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data...')

transform_train = T.Compose(
    [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

transform_test = T.Compose(
    [T.ToTensor(), T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]
)

train_set = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train
)
train_batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=train_batch_size, shuffle=True, num_workers=2
)

test_set = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test
)
test_batch_size = 256
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=test_batch_size, shuffle=False, num_workers=2
)

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

# Model
print(f"==> Using model {args.model}...")
net = getattr(models, args.model)().to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint...")
    assert os.path.isdir('checkpoints'), "Error: no checkpoints directory found!"
    checkpoint = torch.load(f'./checkpoints/checkpoint_{args.model}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


# Data augmentation: Cutout, Mixup, CutMix
if args.use_cutout:
    print("==> Using cutout augmentation...")
    augmentation = aug.Cutout(half_size=8)
elif args.use_mixup:
    print("==> Using mixup augmentation...")
    augmentation = aug.Mixup(alpha=1.0)
elif args.use_cutmix:
    print("==> Using cutmix augmentation...")
    augmentation = aug.CutMix(alpha=1.0)
else:
    augmentation = None


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if augmentation is not None:
            inputs, aug_targets = augmentation(inputs, targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        if augmentation is not None:
            loss = augmentation.criterion(outputs, targets, aug_targets)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        _, predicted_5 = outputs.topk(5, 1, True, True)
        correct_5 += (
            predicted_5.eq(targets.view(-1, 1).expand_as(predicted_5)).sum().item()
        )

    epoch_loss = train_loss / ((batch_idx + 1) * train_batch_size)
    epoch_acc = 100.0 * correct / total
    epoch_acc_5 = 100.0 * correct_5 / total

    return epoch_loss, epoch_acc, epoch_acc_5


def test(epoch):
    global best_acc
    global best_acc_5
    net.eval()
    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            _, predicted_5 = outputs.topk(5, 1, True, True)
            correct_5 += (
                predicted_5.eq(targets.view(-1, 1).expand_as(predicted_5)).sum().item()
            )

    epoch_loss = test_loss / ((batch_idx + 1) * test_batch_size)
    epoch_acc = 100.0 * correct / total
    epoch_acc_5 = 100.0 * correct_5 / total

    # Save checkpoint
    if epoch_acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, f'./checkpoints/checkpoint_{args.model.lower()}.pth')
        best_acc = epoch_acc
        best_acc_5 = epoch_acc_5

    return epoch_loss, epoch_acc, epoch_acc_5


# Main loop
if not os.path.isdir('runs'):
    os.mkdir('runs')
writer = SummaryWriter(log_dir='./runs')

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
lr_schedule = []
for epoch in range(start_epoch, start_epoch + args.max_epoch):
    train_loss, train_acc, train_acc_5 = train(epoch)
    test_loss, test_acc, test_acc_5 = test(epoch)
    scheduler.step()

    # Print log info
    print("============================================================")
    print(f"Epoch: {epoch + 1}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Acc 5: {train_acc_5:.2f}%"
    )
    print(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test Acc@5: {test_acc_5:.2f}%"
    )
    print(f"Best Acc: {best_acc:.2f}% | Best Acc@5: {best_acc_5:.2f}%")
    print("============================================================")

    # Logging
    # train_losses.append(train_loss)
    # train_accuracies.append(train_acc)
    # test_losses.append(test_loss)
    # test_accuracies.append(test_acc)
    # lr_schedule.append(optimizer.param_groups[0]['lr'])
    writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
    writer.add_scalars('Accuracy@5', {'train': train_acc_5, 'test': test_acc_5}, epoch)


# Plot
# plt.figure(figsize=(16, 8))

# plt.subplot2grid((2, 4), (0, 0), colspan=3)
# plt.plot(train_losses, label='train')
# plt.plot(test_losses, label='test')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# plt.subplot2grid((2, 4), (1, 0), colspan=3)
# plt.plot(train_accuracies, label='train')
# plt.plot(test_accuracies, label='test')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')

# plt.subplot2grid((2, 4), (0, 3), rowspan=2)
# plt.plot(lr_schedule, label='lr')
# plt.legend()
# plt.title('Learning Rate')
# plt.xlabel('Epoch')

# if not os.path.isdir('log'):
#     os.mkdir('log')
# plt.savefig(f'./log/{args.model.lower()}.png')
