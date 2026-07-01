import numpy as np
import torch
from sklearn import metrics
from torch import nn

import param
from utils_ import data_loader as dl
# from utils_.models_vit import ViTBase16, ViTFeatVAE
from utils_.models_vit import ViTBase32, ViTFeatVAE

# param.attack_method = "BadNets"

# 1.定义模型，并使用后门模型参数进行初始化
# detected_model = ViTBase16(param.classes_num, False)
detected_model = ViTBase32(param.classes_num, False)
# 2. 构建模型与优化器
auto_encoder = ViTFeatVAE(seq_len=1, feat_dim=768, latent_dim=16)
# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

hook_handles = []
feature = torch.tensor(0.)
clazz = nn.Conv2d
assert clazz in [nn.Conv2d, nn.AdaptiveAvgPool2d]


def save_feature(module, inputs, outputs):
    global feature
    feature = outputs[:, 0:1]


def withdraw_hooks():
    """ 删除钩子函数 """
    for handle in hook_handles:
        handle.remove()


def backdoor_samples_detection(attack: bool):
    global feature, test_set, test_tf
    detected_model.load_state_dict(torch.load(param.backdoor_weights_file_name, map_location=param.device))
    detected_model.to(param.device)
    detected_model.vit.encoder.layers[-3].register_forward_hook(save_feature)
    detected_model.eval()

    auto_encoder.load_state_dict(torch.load(param.backdoor_weights_file_name + "-auto_encoder", map_location=param.device2))
    auto_encoder.to(param.device2)
    auto_encoder.eval()

    data = []

    if attack:
        test_loader = dl.get_backdoor_loader(param.attack_method, test_set, test_tf, False, 1,
                                             1.0, param.target_label_type, param.target_label, param.classes_num)
    else:
        test_loader = dl.get_initial_loader(test_set, test_tf, False, 1, 1.0)

    batch_num = len(test_loader)
    count = 0
    scores = []
    for idx, (img, target) in enumerate(test_loader, start=1):
        img = img.to(param.device)
        # target = target.to(param.device)
        with torch.no_grad():
            x_ = detected_model(img)

        # b, c, h, w = feature.size()[0], feature.size()[1], feature.size()[2], feature.size()[3]
        # feature_ = feature.view((b, c, -1)).unsqueeze(1).to(param.device2)

        distance = 0
        out_, _, _ = auto_encoder(feature)
        # out = out_.squeeze(1).view(b, c, h, w).to(param.device)
        distance_abs = torch.abs(feature - out_)  # b, 1, c, x
        distance_abs = distance_abs.view([1 * 1, 3 * 16, 16])
        # print(distance_abs.shape)
        # distance_abs = torch.nn.functional.avg_pool1d(distance_abs, 4)
        # distance += torch.max(distance_abs).cpu().item() - torch.min(distance_abs).cpu().item()
        # distance += torch.max(torch.mean(distance_abs, dim=-1)).cpu().item() - torch.min(torch.mean(distance_abs, dim=-1)).cpu().item()
        distance += max(
            torch.max(torch.mean(distance_abs, dim=-1)).cpu().item() - torch.min(torch.mean(distance_abs, dim=-1)).cpu().item(),
            torch.max(distance_abs).cpu().item() - torch.min(distance_abs).cpu().item()
        )

        # 后继输出
        # x = detected_model.vit.encoder.layers[-1](out_)
        # x = detected_model.vit.encoder.ln(x)
        # x = x[:, 0]
        # x = detected_model.vit.heads(out_)
        #
        # predict = x.max(dim=-1)[-1]
        #
        # predict_ = x_.max(dim=-1)[-1]
        #
        # acc = predict.eq(predict_).float().mean()
        #
        # if distance < min_:
        #     min_ = distance
        # if distance > max_:
        #     max_ = distance
        if idx % 500 == 499:
            print(f"batch:[{idx:02d}/{batch_num}]    distance：{distance:.4f}")

        # if acc < 0.5:  # 标签发生改变
        #     # else:
        #     data.append(distance)
        #     scores.append(distance)
        # else:
        #     count += 1
        #     scores.append(0)
        data.append(distance)
        scores.append(distance)

    # print(max_, min_)
    # print(count / batch_num)

    # bins = np.arange(10, 60, 0.5)
    # hist, bin_edges = np.histogram(data, bins=bins)

    # print("区间统计:")
    # for edge in range(len(bin_edges) - 1):
    #     # print(f"({bin_edges[edge]:.1f}, {bin_edges[edge + 1]:.1f})\t{hist[edge]}")
    #     print(f"{hist[edge]}")

    np.save(f"./output/scores_{param.model_name}_{param.dataset_name}_" + ("bd_" + param.attack_method if attack else "benign"), scores)
    return count / batch_num


def AUROC_Score(pred_in, pred_out):
    # count = 251
    # for i, j in zip(pred_out, pred_in):
    #     if count > 0:
    # print(f"{i:.2f}\t{j:.2f}")

    # print(pred_out)
    y_in = [1] * len(pred_in)
    y_out = [0] * len(pred_out)

    y = y_in + y_out

    pred = pred_in.tolist() + pred_out.tolist()

    return metrics.roc_auc_score(y, pred)


def process(file_data):
    score = np.load(file_data)
    return score


if __name__ == '__main__':
    # 正常样本
    V_TPR = backdoor_samples_detection(attack=False)
    V_FPR = backdoor_samples_detection(attack=True)

    print(f"V-TPR:{V_TPR:.4f}\tV-FPR:{V_FPR:.4f}")
    AU_ROC = AUROC_Score(
        process(f"./output/scores_{param.model_name}_{param.dataset_name}_bd_{param.attack_method}.npy"),
        process(f"./output/scores_{param.model_name}_{param.dataset_name}_benign.npy"),
    )
    print(f"AUROC:{AU_ROC}")
