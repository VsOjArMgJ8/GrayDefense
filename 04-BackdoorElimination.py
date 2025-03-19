import torch
from torch import nn
from torch.utils.data import random_split

import param
from param import Model
from utils_ import data_loader as dl
from utils_.TAT import test_model
from utils_.grad_coefficient import GradCoefficient

# 1.定义模型，并使用后门模型参数进行初始化
model = Model(param.classes_num, True)
relearn_model = Model(param.classes_num, True)
model.load_state_dict(torch.load(param.backdoor_weights_file_name))
relearn_model.load_state_dict(torch.load(param.backdoor_weights_file_name))

# 训练集
train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)


def split(full_dataset, ratio):
    train_size = int(ratio * len(full_dataset))
    drop_size = len(full_dataset) - train_size
    train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
    print('full data size:', len(full_dataset), 'usage size:', len(train_dataset), 'drop size:', len(drop_dataset))
    return train_dataset, drop_dataset


train_set, _ = split(train_set, 0.2)
train_loader = dl.get_initial_loader(train_set, train_tf, True, param.train_batch_size, 1.0)

# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
asr_loader1 = dl.get_backdoor_loader(param.attack_method, test_set, test_tf, False, param.test_batch_size, 1.0, param.target_label_type, param.target_label, param.classes_num)

# 2. 构建模型与优化器
model.to(device=param.device)
relearn_model.to(device=param.device)

relearn_optimizer = torch.optim.AdamW(relearn_model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss()

_, acc = test_model(acc_loader, relearn_model, criterion, param.device, param.test_print_freq)
_, asr = test_model(asr_loader1, relearn_model, criterion, param.device, param.test_print_freq)
print(f"initial ACC:{acc:.4f}  ASR:{asr:.4f}")


def gradient_coefficient():
    return GradCoefficient(model, train_set, train_tf)()


hook_handles = []
feature, ref = None, None
alpha = 1
clazz = nn.Conv2d
assert clazz in [nn.Conv2d, nn.AdaptiveAvgPool2d]


def save_feature(module, inputs, outputs):
    global feature
    """
    保存最后一层卷积层输出参数的特征值
    :param outputs:
    :param inputs:
    :param module:
    :return:
    """
    feature = outputs


def save_ref(module, inputs, outputs):
    global ref
    """
    保存最后一层卷积层输出参数的特征值
    :param outputs:
    :param inputs:
    :param module:
    :return:
    """
    ref = outputs.detach()


def register_hooks():
    """
    向最后一层卷积层注册钩子函数
    :return:
    """
    global hook_handles, clazz
    # 注册feature hook
    # cov_modules = []
    # for name, module in relearn_model.named_modules():
    #     if isinstance(module, clazz):  # nn.Conv2d
    #         cov_modules.append(module)
    # target_module = cov_modules[-5]
    # handle = target_module.register_forward_hook(save_feature)
    handle = relearn_model.net.layer4.register_forward_hook(save_feature)
    hook_handles.append(handle)
    # 注册ref hook
    # cov_modules = []
    # for name, module in model.named_modules():
    #     if isinstance(module, clazz):  # nn.Conv2d
    #         cov_modules.append(module)
    # target_module = cov_modules[-5]
    # handle = target_module.register_forward_hook(save_ref)
    handle = model.net.layer4.register_forward_hook(save_ref)
    hook_handles.append(handle)


def withdraw_hooks():
    """
    删除钩子函数
    :return:
    """
    for handle in hook_handles:
        handle.remove()


def relearning(epochs):
    global acc, asr, feature, ref, alpha
    # grad_coefficient = GradCoefficient(
    #     relearn_model, train_set, test_tf, ratio=1000 / len(train_set), beta=1, clazz=clazz)
    # alpha = grad_coefficient()
    # alpha = torch.tensor(100., requires_grad=True)
    model.eval()

    for epoch in range(epochs):
        relearn_model.train()
        # 注册钩子函数
        register_hooks()

        for index, (inputs, target) in enumerate(train_loader):

            # 调用模型，得到预测值
            with torch.no_grad():
                model(inputs.to(param.device))
            outputs = relearn_model(inputs.to(param.device))

            # --------------- task loss ------------------
            # task = CE(relearn_model(x_i), y_i)
            task = criterion(outputs, target.to(param.device))

            # ---------------  feature distance ------------------
            # distance = ||feature / (1 + ||feature||) - ref / (1 + ||ref||)||
            assert feature is not None and ref is not None
            # ------------- normalization ----------------
            feature_map_min = feature.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            feature_map_max = feature.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            feature = (feature_map_max - feature) / (feature_map_max - feature_map_min + torch.full_like(feature_map_max, 1e-10, requires_grad=True))
            ref_map_min = ref.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            ref_map_max = ref.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            ref = (ref_map_max - ref) / (ref_map_max - ref_map_min + torch.full_like(feature_map_max, 1e-10, requires_grad=True))

            # ------------- distance ----------------
            # weighted_dist = alpha * torch.abs(feature - ref)
            weighted_dist = alpha * torch.pow(feature - ref, 2)
            distance = weighted_dist.mean()

            # ------------- optimize ----------------
            relearn_optimizer.zero_grad()
            # loss = task - lambda * distance
            # badNet:10; Blend:30; AdvDoor: 50; WaNet:50
            (task - 50 * distance).backward()
            relearn_optimizer.step()

            if index % 16 == 15:
                print(f"epoch:{epoch:02d}    idx:[{index:03d}/{len(train_loader)}]    loss:{task.cpu().item():.4f}    distance:{distance.cpu().item():.4f}")
        # 注销钩子函数
        withdraw_hooks()
        # if epoch % 3 == 2:
        _, acc = test_model(acc_loader, relearn_model, criterion, param.device, param.test_print_freq)
        _, asr = test_model(asr_loader1, relearn_model, criterion, param.device, param.test_print_freq)
        print(f"ACC:{acc:.4f}  ASR:{asr:.4f}")

        # 日志
        with open("./output/log.txt", 'a') as log_file:
            log_file.write(f"DGC => acc:{acc:.4f}, asr:{asr:.4f}" + "\n")
            log_file.flush()

    torch.save(relearn_model.state_dict(), param.backdoor_weights_file_name + "-relearn")


if __name__ == '__main__':
    with open("./output/log.txt", 'a') as log_file:
        log_file.write("-" * 120 + "\n")
        log_file.write(f"target-model: {param.model_name} \t dataset: {param.dataset_name} \t target type: {param.target_label_type} \t attack method: {param.attack_method}\t poison rate:{param.inject_rate * 100}%" + "\n")
        log_file.write("-" * 120 + "\n")
        log_file.flush()
    relearning(60)
