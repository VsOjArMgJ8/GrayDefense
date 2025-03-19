import torch

from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def test_model(test_loader: DataLoader, model: Module, criterion: Module, device_: device, print_freq):
    """
    :param test_loader: 测试数据集
    :param model: 待测试模型
    :param criterion: 计算loss的评判器
    :param device_: 使用的计算设备
    :param print_freq: 打印结果的频率
    :return: 平均loss，平均acc
    """
    model.eval()
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(test_loader)
    for idx, (img, target) in enumerate(test_loader, start=1):
        img = img.to(device_)
        target = target.to(device_)
        # print(target[0])
        with torch.no_grad():
            outputs = model(img)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, target)
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()
        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
        if idx % print_freq == 0:
            print(f"--------->Test  Batch num:[{idx:03d}/{batch_num:03d}]    Loss:{current_loss:.4f}    Acc:{current_acc:.4f}")
    return total_loss / batch_num, total_acc / batch_num


def train_model(train_loader: DataLoader, model: Module, optimizer: Optimizer, criterion: Module, epoch: int, device_: device, print_freq):
    """
    :param train_loader: 训练数据集
    :param model: 待训练模型
    :param optimizer: 模型优化器
    :param criterion: 计算loss的评判器
    :param epoch: 当前第多少轮
    :param device_: 使用的计算设备
    :param print_freq: 打印结果的频率
    :return: 平均loss，平均acc
    """
    model.train()
    model.to(device_)
    total_loss = 0.
    total_acc = 0.
    batch_num = len(train_loader)
    for idx, (img, target) in enumerate(train_loader, start=1):
        img = img.to(device_)
        target = target.to(device_)
        outputs = model(img)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predict = outputs.max(dim=-1)[-1]
        acc = predict.eq(target).float().mean()
        current_loss = loss.cpu().item()
        current_acc = acc.cpu().item()
        total_loss += current_loss
        total_acc += current_acc
        if idx % print_freq == 0:
            print(f"Epoch:{epoch:03d}  Batch num:[{idx:03d}/{batch_num:03d}]    Loss:{current_loss:.4f}    Acc:{current_acc:.4f}")
    return total_loss / batch_num, total_acc / batch_num
