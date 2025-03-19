"""
This is the test code of poisoned training under WaNet.
1. 训练后门模型，得到：后门模型参数、后门数据集
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import Compose
# from MLclf import MLclf

import core

# MLclf.tinyimagenet_data_raw()

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = 1
root_file = "/home/xfLee/Datasets"


def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


# ========== ResNet-34_CIFAR-10_WaNet ==========
resnet34 = torchvision.models.resnet18(pretrained=True)
resnet34.fc = torch.nn.Linear(512, 100)
target_label = 0

transform_train = Compose([
    # transforms.RandomHorizontalFlip(p=0.33),
    # transforms.RandomVerticalFlip(p=0.33),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
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

identity_grid, noise_grid = gen_grid(128, 32)
torch.save(identity_grid, './ResNet-34_CIFAR10_WaNet_identity_grid.pth')
torch.save(noise_grid, './ResNet-34_CIFAR10_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=train_set,
    test_dataset=test_set,
    model=resnet34,
    loss=nn.CrossEntropyLoss(),
    y_target=target_label,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,

    'lr': 0.00005,
    # 'momentum': 0.99,
    # 'weight_decay': 0.1,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 10,

    'log_iteration_interval': 64,
    'test_epoch_interval': 2,
    'save_epoch_interval': 1000000,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-34_ImageNet_WaNet'
}
wanet.train(schedule)
torch.save(wanet.get_model().state_dict(), './data/WaNet/ckpt_epoch_10.pth')

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()
torch.save(poisoned_train_dataset, './data/WaNet/poisoned_test_dataset.pth')

train_dataset, test_dataset = wanet.get_benign_dataset()
torch.save(train_dataset, './data/WaNet/benign_test_dataset.pth')
