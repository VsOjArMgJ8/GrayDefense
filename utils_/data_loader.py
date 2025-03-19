import os

import torch
import torchvision.utils
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets

import param
from utils_.MLclf import MLclf
from utils_.backdoors.AdvDoor import AdvDoor, AdvDoorDataset
from utils_.backdoors.BadNets import BadNetsDataset
from utils_.backdoors.Blend import BlendDataset
from utils_.backdoors.ColorBackdoor import ColorBackdoorDataset
from utils_.backdoors.PhysicalBA import PhysicalBADataset
from utils_.backdoors.WaNet import WaNetsDataset


def get_dataset_and_transformers(dataset_name: str, train: bool):
    """
    :param dataset_name:数据集的名字
    :param train: 是否为训练集
    :return:
    """
    dataset_name = dataset_name.upper()
    assert dataset_name in ['MNIST', 'FASHION-MNIST', 'CIFAR10', 'CIFAR100', "TINYIMAGENET",'GTSRB']
    root_file = "/home/xfLee/Datasets"
    # initial dataset
    if dataset_name == 'MNIST':
        data_set = datasets.MNIST(root=root_file, train=train, download=True)
    elif dataset_name == 'FASHION-MNIST':
        data_set = datasets.FashionMNIST(root=root_file, train=train, download=True)
    elif dataset_name == 'CIFAR10':
        data_set = datasets.CIFAR10(root=root_file + '/CIFAR10', train=train, download=True)
    elif dataset_name == 'CIFAR100':
        data_set = datasets.CIFAR100(root=root_file + '/CIFAR100', train=train, download=True)
    elif dataset_name == 'GTSRB':
        data_set = datasets.GTSRB(root=root_file + '/GTSRB', split = "train" if train else "test", download=True)

    elif dataset_name == "TINYIMAGENET":
        class DatasetTinyImageNet(Dataset):
            def __init__(self, full_dataset=None):
                self.dataset = full_dataset
                self.dataLen = len(self.dataset[0])

            def __getitem__(self, index):

                image = self.dataset[0][index]
                label = self.dataset[1][index]
                return image, label

            def __len__(self):
                return self.dataLen

        MLclf.download_dir_tinyIN = '/home/xfLee/Datasets/TinyImagenet'
        MLclf.tinyimagenet_download()
        data_raw_train, data_raw_test = MLclf.tinyimagenet_data_raw()

        if train:
            data_set = data_raw_train
        else:
            data_set = data_raw_test

        data_set = DatasetTinyImageNet(full_dataset=data_set)
    else:
        raise Exception('Invalid dataset')

    # ------------- data augmentation ----------------
    if train:
        # transforms
        if dataset_name in ['MNIST', 'FASHION-MNIST']:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                # transforms.RandomRotation(8),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.2, 0.2, 0.2,), (0.3, 0.3, 0.3,))
            ])
        else:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                # transforms.RandomRotation(8),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
            ])
    else:
        if dataset_name in ['MNIST', 'FASHION-MNIST']:
            tf = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.2, 0.2, 0.2,), (0.3, 0.3, 0.3,))
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))
            ])
    return data_set, tf


# 原始数据集
def get_initial_loader(full_dataset, transform, train, batch_size: int, utilization_rate: float, other_transform_func=None):
    train_data = DatasetCL(full_dataset=full_dataset, transform=transform, utilization_rate=utilization_rate, other_transform_func=other_transform_func)
    return DataLoader(train_data, batch_size=batch_size, shuffle=train)


# 后门数据集
def get_backdoor_loader(attack_method, full_dataset, transform, train: bool, batch_size: int,
                        inject_portion: float, target_label_type, target_label, class_num, shuffle=None, other_transform_func=None):
    if shuffle is None:
        shuffle = train
    assert attack_method in ["BadNets", "Blend", "WaNet", "AdvDoor", "PhysicalBA"]
    if attack_method == "BadNets":
        train_data_bad = BadNetsDataset(full_dataset, train, inject_portion, class_num, target_label_type=target_label_type, target_label=target_label, transform=transform, other_transform_func=other_transform_func)
    elif attack_method == "Blend":
        train_data_bad = BlendDataset(full_dataset, train, inject_portion, class_num, target_label_type=target_label_type, target_label=target_label, transform=transform)
    elif attack_method == "WaNet":
        train_data_bad = WaNetsDataset(full_dataset, train, inject_portion, class_num, target_label_type=target_label_type, target_label=target_label, transform=transform)
    elif attack_method == "PhysicalBA":
        train_data_bad = PhysicalBADataset(full_dataset, train, inject_portion, class_num, target_label_type=target_label_type, target_label=target_label, transform=transform)
    elif attack_method == "ColorBackdoor":
        train_data_bad = ColorBackdoorDataset(full_dataset, train, inject_portion, class_num, target_label_type=target_label_type, target_label=target_label, transform=transform)
    elif attack_method == "AdvDoor":

        uap_file_name = param.backdoor_weights_file_name + "-uap.jpeg"
        # 获取UAP
        if not os.path.exists(uap_file_name):
            train_set, _ = get_dataset_and_transformers(param.dataset_name, True)
            test_set, test_tf = get_dataset_and_transformers(param.dataset_name, False)
            test_loader = get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
            model = param.Model(param.classes_num)
            model.load_state_dict(torch.load(param.model_weights_file_name, map_location=param.device))
            model.eval()
            uap = AdvDoor(model, train_set, test_loader, True, target_class=param.target_label, p_samples=0.25).universal_perturbation(test_tf, param.classes_num)
            torchvision.utils.save_image(uap, param.backdoor_weights_file_name + "-uap.jpeg")
        uap = Image.open(uap_file_name)
        # 获取TUAP数据集
        train_data_bad = AdvDoorDataset(full_dataset, train, inject_portion, class_num, pattern=uap, target_label_type=target_label_type, target_label=target_label, transform=transform)
    else:
        raise RuntimeError("not implement backdoor attack")
    return DataLoader(train_data_bad, batch_size=batch_size, shuffle=shuffle)


class DatasetCL(Dataset):
    def __init__(self, utilization_rate, full_dataset=None, transform=None, other_transform_func=None):
        if utilization_rate < 1.0:
            self.dataset = self.random_split(full_dataset=full_dataset, ratio=utilization_rate)
        else:
            self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)
        self.other_transform_func = other_transform_func

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if len(image.split()) == 1:
            image = transforms.Grayscale(num_output_channels=3)(image)
        if self.other_transform_func is not None:
            image = self.other_transform_func(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen

    @staticmethod
    def random_split(full_dataset, ratio):
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('full data size:', len(full_dataset), 'usage size:', len(train_dataset), 'drop size:', len(drop_dataset))
        return train_dataset
