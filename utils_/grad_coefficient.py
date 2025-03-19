import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm

import param


class GradCoefficient:
    def __init__(self, model, dataset, transform, ratio=0.01, beta=1, clazz=nn.Conv2d):
        """
        :param model:
        :param dataset:
        :param transform:
        """
        self.model = model.to(param.device).eval()
        self.dataset = self.random_split(dataset, ratio)
        self.transform = transform
        self.feature = None
        self.gradient = None
        self.hook_handles = []
        self.beta = beta
        self.clazz = clazz

    @staticmethod
    def random_split(full_dataset, ratio):
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('full data size:', len(full_dataset), 'usage size:', len(train_dataset), 'drop size:', len(drop_dataset))
        return train_dataset

    def save_gradient(self, module, grad_inputs, grad_outputs):
        """
        保存最后一层卷积层输出参数的梯度
        :param grad_outputs:
        :param grad_inputs:
        :param module:
        :return:
        """
        self.gradient = grad_outputs[0].detach()

    def save_feature(self, module, inputs, outputs):
        """
        保存最后一层卷积层输出参数的特征值
        :param outputs:
        :param inputs:
        :param module:
        :return:
        """
        self.feature = outputs.detach()
        # print("feature:", self.feature)

    def register_hooks(self):
        """
        向最后一层卷积层注册钩子函数
        :return:
        """
        cov_modules = []
        for name, module in self.model.named_modules():
            print(name)
            # if isinstance(module, self.clazz):
            if name == "net.layer3.5.conv2":
                cov_modules.append(module)
            if isinstance(module, nn.ReLU):
                module.inplace = False
        target_module = cov_modules[-1]
        handle = target_module.register_forward_hook(self.save_feature)
        self.hook_handles.append(handle)
        handle = target_module.register_full_backward_hook(self.save_gradient)
        self.hook_handles.append(handle)

    def withdraw_hooks(self):
        """
        删除钩子函数
        :return:
        """
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self):
        p = None

        # 1.向目标模型注册钩子函数
        self.register_hooks()
        # 2.计算激活次数
        for img, label in tqdm(self.dataset):
            img = self.transform(img).unsqueeze(0).to(param.device)
            outputs = self.model(img)

            classes = F.sigmoid(outputs)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            # channel active degree
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            degree = F.relu(weight * self.feature)

            if p is not None:
                p = torch.add(p, degree)
            else:
                p = degree
        self.withdraw_hooks()

        # 3.计算梯度系数
        # return p / len(self.dataset)
        p = p - torch.min(p)
        p = p / torch.max(p)

        # return self.beta * p  # 会引起训练后期ASR返升，why？？？
        alpha = self.beta * (1 - p) / (1 + p)
        return alpha
