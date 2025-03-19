import torch
import sys
import warnings
import numpy as np
import random
from utils_ import models_

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 20240404
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 模型名称
model_name = "resnet18"
assert model_name in [
    "alexnet",
    "vgg13", "vgg19",
    "googlenet",
    "densnet121", "densnet169",
    "resnet18", "resnet34", "resnet50", "resnet101",
    "wideresnet50", "wideresnet101"
]

# 使用数据集名称
dataset_name = "GTSRB"
assert dataset_name in ['MNIST', 'FASHION-MNIST', 'CIFAR10', 'CIFAR100', "TinyImageNet",'GTSRB']

attack_method = "WaNet"
assert attack_method in ["BadNets", "Blend", "WaNet", "AdvDoor", "PhysicalBA"]

# 谨慎修改
simi_v_dim = 32

pre_train = True

# 后门攻击label类型
target_label_type = "all2one"
assert target_label_type in ['all2one', 'all2all']

# 训练后门模型时，后门攻击的label
target_label = 0  # under all-to-one attack

# 模型训练与测试
train_batch_size = 320
train_print_freq = 32
# train_print_freq = sys.maxsize

test_batch_size = 512
test_print_freq = sys.maxsize

learning_rate = 0.00003
# 后门训练集样本污染率(在训练后门模型时使用)
inject_rate = 0.1
# 训练轮次(如果模型在进行Fuzz测试之前未经过训练的话，会先训练待测试模型)
train_epoch = 10
# 分类样本的类别总数
classes_num = 5
if dataset_name == "MNIST":
    train_epoch = 2
if dataset_name == "CLTL":
    classes_num = 7
elif dataset_name == "GTSRB":
    classes_num = 43
elif dataset_name == "CIFAR100":
    classes_num = 100
elif dataset_name == "TinyImageNet":
    classes_num = 200

backdoor_weights_file_name = "./weights_/" + f"{model_name}-{dataset_name}-{target_label_type}-{attack_method}-{inject_rate * 100}%.pth"
model_weights_file_name = "./weights_/" + f"{model_name}-{dataset_name}-benign.pth"

print("-" * 120)
print(f"target-model: {model_name} \t dataset: {dataset_name} \t target type: {target_label_type} \t attack method: {attack_method}\t poison rate:{inject_rate * 100}%")
print("-" * 120)

if model_name == "alexnet":
    Model = models_.AlexNet
elif model_name == "vgg13":
    Model = models_.VGG13
elif model_name == "vgg19":
    Model = models_.VGG19
elif model_name == "googlenet":
    Model = models_.GoogleNet
elif model_name == "densnet121":
    Model = models_.DenseNet121
elif model_name == "densnet169":
    Model = models_.DenseNet169
elif model_name == "resnet18":
    Model = models_.ResNet18
elif model_name == "resnet34":
    Model = models_.ResNet34
elif model_name == "resnet50":
    Model = models_.ResNet50
elif model_name == "resnet101":
    Model = models_.ResNet101
elif model_name == "wideresnet50":
    Model = models_.WideResNet50
elif model_name == "wideresnet101":
    Model = models_.WideResNet101
