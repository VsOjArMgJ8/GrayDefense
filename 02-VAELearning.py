import torch
from torch.utils.data import random_split

import param
from param import Model
from utils_ import data_loader as dl
from utils_.models_ import VariationalAutoEncoder

# 1.定义模型，并使用后门模型参数进行初始化
detected_model = Model(param.classes_num, False)
# 2. 构建模型与优化器
auto_encoder = VariationalAutoEncoder(param.simi_v_dim)

hook_handles = []
feature = torch.tensor(0.)


def save_feature(module, inputs, outputs):
    global feature
    feature = outputs.detach()


def withdraw_hooks():
    """
    删除钩子函数
    :return:
    """
    for handle in hook_handles:
        handle.remove()


def train_auto_encoder(epochs: int):
    global feature
    # 注册钩子
    # register_hooks()

    detected_model.load_state_dict(torch.load(param.backdoor_weights_file_name))
    detected_model.to(param.device)
    detected_model.eval()

    handle = detected_model.net.layer3.register_forward_hook(save_feature)
    hook_handles.append(handle)

    auto_encoder.to(param.device2)
    auto_encoder.train()
    # 优化器
    optimizer = torch.optim.AdamW(auto_encoder.parameters(), lr=0.005)
    # 损失函数
    criterion = torch.nn.MSELoss(reduction="sum")
    # 训练集
    train_set, _ = dl.get_dataset_and_transformers(param.dataset_name, True)
    _, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
    train_size = int(0.1 * len(train_set))
    drop_size = len(train_set) - train_size
    train_dataset, drop_dataset = random_split(train_set, [train_size, drop_size])
    print('full data size:', len(train_set), 'usage size:', len(train_dataset), 'drop size:', len(drop_dataset))
    train_set = train_dataset
    train_loader = dl.get_initial_loader(train_set, test_tf, True, param.train_batch_size, 1.0)

    for epoch in range(epochs):
        batch_num = len(train_loader)
        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.to(param.device)
            target = target.to(param.device)
            with torch.no_grad():
                detected_model(img)

            b, c, h, w = feature.size()[0], feature.size()[1], feature.size()[2], feature.size()[3]
            feature = feature.view((b, c, -1)).unsqueeze(1).to(param.device2)

            out_, mean, logvar = auto_encoder(feature)

            loss = criterion(out_, feature) - 0.5 * torch.sum(1 + logvar - torch.exp(logvar) - torch.pow(mean, 2), dim=-1).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            distance = torch.sqrt((feature - out_) ** 2).sum(-1).mean()

            out_ = out_.squeeze(1).view(b, c, h, w).to(param.device)

            # 后继输出
            x = detected_model.net.layer4(out_)
            x = detected_model.net.avgpool(x)
            x = torch.flatten(x, 1)
            x = detected_model.net.fc(x)

            # 重构准确率
            predict = x.max(dim=-1)[-1]
            acc = predict.eq(target).float().mean()

            print(f"epoch:[{epoch:02d}/{epochs}]    batch:[{idx:02d}/{batch_num}]    loss:{loss.cpu().item():.4f}     distance:{distance.cpu().item():.4f}    reconstruct acc:{acc.cpu().item():.4f}")
            del img, out_, loss

        if epoch % 20 == 19:
            torch.save(auto_encoder.state_dict(), param.backdoor_weights_file_name + "-auto_encoder")
            print("save VAE weights_: " + param.backdoor_weights_file_name + "-auto_encoder")
            optimizer.zero_grad()

    withdraw_hooks()


if __name__ == '__main__':
    train_auto_encoder(150)
