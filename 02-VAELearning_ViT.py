import torch
from torch.utils.data import random_split

import param
from utils_ import data_loader as dl
# from utils_.models_vit import ViTBase16, ViTFeatVAE
from utils_.models_vit import ViTBase32, ViTFeatVAE

# 1.定义模型，并使用后门模型参数进行初始化
# detected_model = ViTBase16(param.classes_num, False)
detected_model = ViTBase32(param.classes_num, False)
# 2. 构建模型与优化器
auto_encoder = ViTFeatVAE(seq_len=1, feat_dim=768, latent_dim=16)  # 0.88(128) -> 0.89(64)  -> 0.89(32) ->0.905(16)

hook_handles = []
feature = torch.tensor(0.)


def save_feature(module, inputs, outputs):
    global feature
    feature = outputs.detach()[:, 0:1]


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

    # handle = detected_model.vit.encoder.layers[-3].register_forward_hook(save_feature)
    handle = detected_model.vit.encoder.layers[-1].register_forward_hook(save_feature)
    hook_handles.append(handle)

    auto_encoder.to(param.device2)
    auto_encoder.train()
    # 优化器
    optimizer = torch.optim.AdamW(auto_encoder.parameters(), lr=3e-4)
    # 损失函数
    criterion = torch.nn.MSELoss(reduction="mean")
    # 训练集
    train_set, _ = dl.get_dataset_and_transformers(param.dataset_name, True)
    _, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
    train_size = int(0.1 * len(train_set))
    drop_size = len(train_set) - train_size
    train_dataset, drop_dataset = random_split(train_set, [train_size, drop_size])
    print('full data size:', len(train_set), 'usage size:', len(train_dataset), 'drop size:', len(drop_dataset))
    train_set = train_dataset
    train_loader = dl.get_initial_loader(train_set, test_tf, True, 512, 1.0)

    for epoch in range(epochs):
        batch_num = len(train_loader)
        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.to(param.device)
            target = target.to(param.device)
            with torch.no_grad():
                detected_model(img)

            out_, mean, logvar = auto_encoder(feature)

            loss = criterion(out_, feature) - 0.5 * torch.sum(1 + logvar - torch.exp(logvar) - torch.pow(mean, 2), dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            distance = torch.sqrt((feature - out_) ** 2).sum(-1).mean()

            # 后继输出
            # x = detected_model.vit.encoder.layers[-1](out_)
            # x = detected_model.vit.encoder.ln(x)
            # x = x[:, 0]
            # x = detected_model.vit.heads(out_)

            # 重构准确率
            # predict = x.max(dim=-1)[-1]
            # acc = predict.eq(target).float().mean()
            # if epoch % 20 == 19 and idx % 8 == 0:
            #     print(f"epoch:[{epoch:02d}/{epochs}]    batch:[{idx:02d}/{batch_num}]    loss:{loss.cpu().item():.4f}     distance:{distance.cpu().item():.4f}    reconstruct acc:{acc.cpu().item():.4f}")
            print(f"epoch:[{epoch:02d}/{epochs}]    batch:[{idx:02d}/{batch_num}]    loss:{loss.cpu().item():.4f}     distance:{distance.cpu().item():.4f}")
            del img, out_, loss

        if epoch % 10 == 9:
            torch.save(auto_encoder.state_dict(), param.backdoor_weights_file_name + "-auto_encoder")
            print("save VAE weights_: " + param.backdoor_weights_file_name + "-auto_encoder")
            optimizer.zero_grad()

    withdraw_hooks()


if __name__ == '__main__':
    train_auto_encoder(10)
