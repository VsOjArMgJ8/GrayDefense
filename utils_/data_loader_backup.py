import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from MLclf import MLclf
import param


def get_dataset_and_transformers(dataset_name: str, train: bool):
    """
    :param dataset_name:数据集的名字
    :param train: 是否为训练集
    :return:
    """
    dataset_name = dataset_name.upper()
    assert dataset_name in ['MNIST', 'FASHION-MNIST', 'CIFAR10', 'CIFAR100', "TinyImageNet"]
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
    else:
        raise Exception('Invalid dataset')

    # ------------- data augmentation ----------------
    if train:
        # transforms
        if dataset_name in ['MNIST', 'FASHION-MNIST']:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.33),
                transforms.RandomVerticalFlip(p=0.33),
                # transforms.RandomRotation(8),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.2, 0.2, 0.2,), (0.3, 0.3, 0.3,))
            ])
        else:
            tf = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.33),
                transforms.RandomVerticalFlip(p=0.33),
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
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
    return data_set, tf


# 原始数据集
def get_initial_loader(full_dataset, transform, train, batch_size: int, utilization_rate: float):
    train_data = DatasetCL(full_dataset=full_dataset, transform=transform, utilization_rate=utilization_rate)
    return DataLoader(train_data, batch_size=batch_size, shuffle=train)


# 后门数据集
def get_backdoor_loader(full_dataset, transform, train, batch_size: int, inject_rate: float, shuffle=None):
    if shuffle is None:
        shuffle = train
    train_data_bad = DatasetBD(full_dataset=full_dataset, train=train, inject_ratio=inject_rate, transform=transform,
                               target_label=param.target_label, trigger_size=(5, 5), trigger_type=param.trigger_type, target_type=param.target_label_type)
    return DataLoader(train_data_bad, batch_size=batch_size, shuffle=shuffle)


class DatasetCL(Dataset):
    def __init__(self, utilization_rate, full_dataset=None, transform=None):
        if utilization_rate < 1.0:
            self.dataset = self.random_split(full_dataset=full_dataset, ratio=utilization_rate)
        else:
            self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        if len(image.split()) == 1:
            image = transforms.Grayscale(num_output_channels=3)(image)
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


class DatasetBD(Dataset):
    def __init__(self, full_dataset, train, inject_ratio, transform=None,
                 distance=1, target_label=0, trigger_size=(5, 5), trigger_type=param.trigger_type, target_type=param.target_label_type):

        self.dataset = self.addTrigger(full_dataset, train, target_label, inject_ratio,
                                       distance, trigger_size[0], trigger_size[1], trigger_type, target_type)
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, train, target_label, inject_portion, distance, trig_w, trig_h, trigger_type, target_type):
        selected_index = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]
            if len(img.split()) == 1:
                img = img.convert("RGB")
            width, height = img.size
            label = data[1]
            # all2one attack
            if target_type == 'all2one':
                # test set do not contain the samples of target label in all to one
                if not train and label == target_label:
                    continue
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                    img = Image.fromarray(img, mode="RGB")
                    # change target
                    label = target_label
                    cnt += 1
            # all2all attack
            elif target_type == 'all2all':
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                    img = Image.fromarray(img)
                    # change target
                    label = self._change_label_next(label)
                    cnt += 1
            dataset_.append((img, label))

        print("Injecting Over: " + str(cnt) + "Bad Images, " + str(len(dataset_) - cnt) + "Clean Images")
        return dataset_

    @staticmethod
    def _change_label_next(label):
        return (label + 1) % param.classes_num

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, trigger_type):

        assert trigger_type in ['gridTrigger', 'trojanTrigger', 'blendTrigger', "warpingTrigger",
                                'squareTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', ]

        if trigger_type == 'gridTrigger':
            img = self._gridTrigger(img, width, height, distance, trig_w, trig_h)

        elif trigger_type == 'blendTrigger':
            img = self._blendTrigger(img, width, height, distance, trig_w, trig_h)

        elif trigger_type == 'warpingTrigger':
            img = self._warpingTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    @staticmethod
    def _warpingTrigger(img, width, height, distance, trig_w, trig_h):
        # WaNet
        array1d = torch.linspace(-1, 1, steps=width)  # 将-1到1分割成steps份
        x, y = torch.meshgrid(array1d, array1d)  # 笛卡尔积，生成坐标点
        identity_grid = torch.stack((x, y), 2)[None, ...]

        ins = torch.rand(1, 2, 32, 32) * 2 - torch.tensor(1)
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=width, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)

        grid_temps = (identity_grid + 0.5 * noise_grid / width) * 1
        grid_temps = torch.clamp(grid_temps, -1, 1)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        inputs_bd = F.grid_sample(img, grid_temps, align_corners=True).squeeze(0)
        inputs_bd = torch.clamp(inputs_bd, 0, 255).permute(2, 1, 0)

        return np.array(inputs_bd.numpy(), dtype=np.uint8)

    @staticmethod
    def _blendTrigger(img, width, height, distance, trig_w, trig_h):
        # blend
        gauss_noise = np.random.normal(0, 40, img.shape)
        img_arr = img.astype('int')
        img_arr = img_arr + gauss_noise
        img_arr = np.clip(img_arr, 0, 255)
        img_arr = img_arr.astype('uint8')
        return img_arr

    @staticmethod
    def _gridTrigger(img, weight, height, distance, trig_w, trig_h):
        # badNet
        img[height - 1][weight - 1] = 255
        img[height - 1][weight - 2] = 0
        img[height - 1][weight - 3] = 255

        img[height - 2][weight - 1] = 0
        img[height - 2][weight - 2] = 255
        img[height - 2][weight - 3] = 0

        img[height - 3][weight - 1] = 255
        img[height - 3][weight - 2] = 0
        img[height - 3][weight - 3] = 0
        return img


if __name__ == '__main__':
    pass
