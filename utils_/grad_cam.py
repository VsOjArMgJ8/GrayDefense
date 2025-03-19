import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable


class GradCam:
    def __init__(self, model):
        """
        :param model: 目标模型：alexnet、vgg、densnet、
        """
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        """
        :param grad: 梯度
        :return:
        """
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datum = Variable(x)
        heat_maps = []
        for i in range(datum.size(0)):
            img = datum[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datum[i].unsqueeze(0)
            for name, module in self.model.net.named_children():
                # 针对分类模块，在分类模块计算之前，对数据进行调整
                if name == 'classifier':
                    feature = F.relu(feature, inplace=True)
                    feature = F.adaptive_avg_pool2d(feature, (1, 1))
                    feature = torch.flatten(feature, 1)

                # 模块对特征进行计算
                feature = module(feature)
                # 针对特征提取模块，注册获取梯度的函数
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            classes = F.sigmoid(feature)
            print(classes.max(dim=-1)[-1])
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps
