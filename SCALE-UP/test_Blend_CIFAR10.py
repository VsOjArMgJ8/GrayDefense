"""
This is the test code of poisoned training under WaNet.
1. 训练后门模型，得到：后门模型参数、后门数据集
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import Compose

import core

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

CUDA_VISIBLE_DEVICES = 1
root_file = "/home/xfLee/Datasets"

# ========== ResNet-34_CIFAR-10_WaNet ==========
resnet34 = torchvision.models.resnet34(pretrained=True)
resnet34.fc = torch.nn.Linear(512, 100)
target_label = 0

transform_train = Compose([
    # transforms.RandomHorizontalFlip(p=0.33),
    # transforms.RandomVerticalFlip(p=0.33),
    # transforms.RandomRotation(8),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
])

train_set = datasets.CIFAR100(root=root_file + '/CIFAR100', train=True, download=True, transform=transform_train)

transform_test = Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_set = datasets.CIFAR100(root=root_file + '/CIFAR100', train=False, download=True, transform=transform_test)
data_ = []
targets_ = []
for i in range(len(test_set.targets)):
    if test_set.targets[i] == target_label:
        pass
    else:
        data_.append(test_set.data[i])
        targets_.append(test_set.targets[i])
test_set.data = data_
test_set.targets = targets_

# pattern = torch.normal(0, 40, (3, 32, 32))
#
# weight = torch.tensor(pattern, dtype=torch.float32)
# torch.fill(weight, 0.5)

blend = core.Blended(
    train_dataset=train_set,
    test_dataset=test_set,
    # model=core.models.ResNet(18),
    model=resnet34,
    loss=nn.CrossEntropyLoss(),
    # pattern=pattern,
    # weight=weight,
    y_target=target_label,
    poisoned_rate=0.1,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 256,
    'num_workers': 1,

    'lr': 0.00001,
    'momentum': 0.9,
    'weight_decay': 0.01,
    'gamma': 0.01,
    'schedule': [150, 180],

    'epochs': 10,

    'log_iteration_interval': 64,
    'test_epoch_interval': 2,
    'save_epoch_interval': 1000000,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-34_ImageNet_WaNet'
}
blend.train(schedule)
torch.save(blend.get_model().state_dict(), './data/Blend/ckpt_epoch_10.pth')

_, poisoned_test_dataset = blend.get_poisoned_dataset()
torch.save(poisoned_test_dataset, './data/Blend/poisoned_test_dataset.pth')

_, test_dataset = blend.get_benign_dataset()
torch.save(test_dataset, './data/Blend/benign_test_dataset.pth')
