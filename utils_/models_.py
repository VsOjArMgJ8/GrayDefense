import sys

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

# class VariationalAutoEncoder(nn.Module):
#     def __init__(self):
#         super(VariationalAutoEncoder, self).__init__()
#
#         # 编码器部分
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入1通道，输出16通道，3x3卷积，步长为2，padding为1
#             # nn.BatchNorm2d(8),
#             nn.LayerNorm([128, 32]),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
#             # nn.BatchNorm2d(16),
#             nn.LayerNorm([64, 16]),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
#             # nn.BatchNorm2d(32),
#             nn.LayerNorm([30, 6]),
#             nn.ReLU(),
#         )
#         self.fc_mean = nn.Linear(32 * 30 * 6, 128)
#         self.fc_logvar = nn.Linear(32 * 30 * 6, 128)
#         self.fc = nn.Linear(128, 32 * 30 * 6)
#
#         # 解码器部分
#         self.decoder = nn.Sequential(
#             nn.LayerNorm([30, 6]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1)),
#             nn.LayerNorm([64, 16]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
#             nn.LayerNorm([128, 32]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
#         )
#
#     @staticmethod
#     def reparametrize(mean, logvar):
#         eps = torch.randn_like(logvar)  # 从标准正态分布中采样噪声
#         std = torch.exp(logvar / 2)
#         return mean + eps * std
#
#     def forward(self, x):
#         # 编码器
#         x = self.encoder(x)
#
#         # 重组线性化
#         size = x.size()
#         x = torch.flatten(input=x, start_dim=1, end_dim=-1)
#         mean = self.fc_mean(x)
#         logvar = self.fc_logvar(x)
#         z = self.reparametrize(mean, logvar)
#         z = self.fc(z)
#         x = z.view(size)
#
#         # 解码器
#         x = self.decoder(x)
#         return x, mean, logvar


# class VariationalAutoEncoder(nn.Module):
#     def __init__(self):
#         super(VariationalAutoEncoder, self).__init__()
#
#         # 编码器部分
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入1通道，输出16通道，3x3卷积，步长为2，padding为1
#             # nn.BatchNorm2d(8),
#             nn.LayerNorm([128, 32]),
#             nn.ReLU(),
#             nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
#             # nn.BatchNorm2d(16),
#             nn.LayerNorm([64, 16]),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
#             # nn.BatchNorm2d(32),
#             nn.LayerNorm([30, 6]),
#             nn.ReLU(),
#         )
#         self.fc_mean = nn.Linear(32 * 30 * 6, 128)
#         self.fc_logvar = nn.Linear(32 * 30 * 6, 128)
#         self.fc = nn.Linear(128, 32 * 30 * 6)
#
#         # 解码器部分
#         self.decoder = nn.Sequential(
#             nn.LayerNorm([30, 6]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1)),  # 输入64通道，输出32通道，7x7卷积转置
#             nn.LayerNorm([64, 16]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # 输入32通道，输出16通道，3x3卷积转置，步长为2，padding为1，输出padding为1
#             nn.LayerNorm([128, 32]),
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # 输入16通道，输出1通道，3x3卷积转置，步长为2，padding为1，输出padding为1
#             # nn.Sigmoid()
#         )
#
#     @staticmethod
#     def reparametrize(mean, logvar):
#         eps = torch.randn_like(logvar)  # 从标准正态分布中采样噪声
#         std = torch.exp(logvar / 2)
#         return mean + eps * std
#
#     def forward(self, x):
#         # 编码器
#         x = self.encoder(x)
#
#         # 重组线性化
#         size = x.size()
#         x = torch.flatten(input=x, start_dim=1, end_dim=-1)
#         mean = self.fc_mean(x)
#         logvar = self.fc_logvar(x)
#         z = self.reparametrize(mean, logvar)
#         z = self.fc(z)
#         x = z.view(size)
#
#         # 解码器
#         x = self.decoder(x)
#         return x, mean, logvar

# 定义一个变分自编码器的类
import param


class VariationalAutoEncoder(nn.Module):
    def __init__(self, simi_v_dim):
        super(VariationalAutoEncoder, self).__init__()

        self.activate = nn.ReLU

        self.simi_v_dim = simi_v_dim

        # 编码器部分
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 输入1通道，输出16通道，3x3卷积，步长为2，padding为1
            self.activate(),

            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入1通道，输出16通道，3x3卷积，步长为2，padding为1
            nn.LayerNorm([128, 32]),
            # self.activate(),
        )

        self.down_sample1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1, 8, kernel_size=(1, 1), stride=(1, 1))
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
            self.activate(),

            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
            nn.LayerNorm([64, 16]),
            # self.activate(),
        )

        self.down_sample2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.LayerNorm([30, 6]),
        )

        self.fc_mean = nn.Linear(32 * 30 * 6, self.simi_v_dim)
        self.fc_logvar = nn.Linear(32 * 30 * 6, self.simi_v_dim)
        self.fc = nn.Linear(self.simi_v_dim, 32 * 30 * 6)

        self.decoder1 = nn.Sequential(
            nn.LayerNorm([30, 6]),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1)),
        )

        self.up_sample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1)),
        )

        self.decoder2 = nn.Sequential(
            # self.activate(),
            nn.LayerNorm([64, 16]),
            nn.ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            self.activate(),
            nn.LayerNorm([64, 16]),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        )

        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1)),
        )

        self.decoder3 = nn.Sequential(
            # self.activate(),
            nn.LayerNorm([128, 32]),
            nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            self.activate(),
            nn.LayerNorm([128, 32]),
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        )

    #     self.init_parameter()
    #
    # def init_parameter(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             # print(m.weight)
    #             nn.init.normal_(m.weight, 5, 10)
    #             # nn.init.constant_(m.weight, 0)
    #             nn.init.constant_(m.bias, 1)

    @staticmethod
    def reparametrize(mean, logvar):
        eps = torch.randn_like(logvar)  # 从标准正态分布中采样噪声
        std = torch.exp(logvar / 2)
        return mean + eps * std

    def forward(self, x):
        # 编码器

        # print(x.size())

        x = self.encoder1(x) + self.down_sample1(x)
        x = self.activate()(x)

        # print(x.size())

        x = self.encoder2(x) + self.down_sample2(x)
        x = self.activate()(x)

        # print(x.size())

        x = self.encoder3(x)

        # print(x.size())

        # 重组线性化
        size = x.size()
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)

        # print(x.size())

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparametrize(mean, logvar)

        # print(x.size())

        z = self.fc(z)
        x = z.view(size)

        # print(x.size())

        # 解码器
        x = self.decoder1(x)

        # print(x.size())

        x = self.activate()(x)
        x = self.decoder2(x) + self.up_sample1(x)

        # print(x.size())

        x = self.activate()(x)
        x = self.decoder3(x) + self.up_sample2(x)

        # print(x.size())

        return x, mean, logvar


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入1通道，输出16通道，3x3卷积，步长为2，padding为1
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),  # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.middle_layer = nn.Sequential(
            nn.Linear(32 * 30 * 6, 32),
            nn.ReLU(),
            nn.Linear(64, 32 * 30 * 6)
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1)),  # 输入64通道，输出32通道，7x7卷积转置
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # 输入32通道，输出16通道，3x3卷积转置，步长为2，padding为1，输出padding为1
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # 输入16通道，输出1通道，3x3卷积转置，步长为2，padding为1，输出padding为1
            nn.Sigmoid()
        )

        # self.init_parameter()

    def init_parameter(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m.weight)
                nn.init.normal_(m.weight, 0.5, 1)
                nn.init.constant_(m.bias, 0.5)

    def forward(self, x):
        # print(x.size())
        x = self.encoder(x)

        size = x.size()
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)
        x = self.middle_layer(x)
        x = x.view(size)

        # print(x.size())
        x = self.decoder(x)

        return x


class AlexNetWithTwoFC(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()

        if pre_train:
            self.net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.alexnet()

        self.net.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2048, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2048, class_num, bias=False)
        )
        self.dropout_p = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.features(x)
        x = self.net.avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, self.dropout_p, True, True)
        x = self.net.classifier(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()

        if pre_train:
            self.net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.alexnet()

        self.BN = nn.BatchNorm2d(256)

        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_num)
            # nn.Linear(256 * 6 * 6, class_num, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.features(x)
        x = self.BN(x)
        x = self.net.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class VGG13(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()

        if pre_train:
            self.net = torchvision.models.vgg13(weights=torchvision.models.VGG13_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.vgg13()

        self.net.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            # nn.Dropout(p=0.3),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(p=0.3),
            # nn.Linear(4096, class_num),

            nn.Linear(512 * 7 * 7, class_num)
        )

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class VGG19(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()

        if pre_train:
            self.net = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.vgg19()

        self.BN = nn.BatchNorm2d(512)

        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, class_num)
            # nn.Linear(512 * 7 * 7, class_num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.features(x)
        x = self.BN(x)
        x = self.net.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class DenseNet121(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.densenet121()

        self.net.classifier = nn.Linear(self.net.classifier.in_features, class_num)

        self.avp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        # inputs = F.dropout(inputs, 0.1, True, True)
        # inputs = F.batch_norm(inputs, torch.tensor([0.5, 0.5, 0.5]).to(param.device), torch.tensor([0.3, 0.3, 0.3]).to(param.device))
        # inputs = F.interpolate(F.interpolate(inputs, size=(24, 24), mode="area"), size=init_features_size, mode="area")  # area
        # init_features_size = inputs.size()[-2], inputs.size()[-1]
        features = self.net.features(inputs)
        out = F.relu(features, inplace=True)
        out = self.avp(out)
        out = torch.flatten(out, 1)
        out = self.net.classifier(out)
        return out


class DenseNet169(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.densenet169()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, class_num)

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class GoogleNet(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()
        if pre_train:
            self.net = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.googlenet()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class ResNet18(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.resnet18()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)
        # self.dropout_p = 0

    def forward(self, x):
        x = self.net(x)

        # x = self.net.conv1(x)
        # x = self.net.bn1(x)
        # x = self.net.relu(x)
        # x = self.net.maxpool(x)
        #
        # x = self.net.layer1(x)
        # x = self.net.layer2(x)
        # x = self.net.layer3(x)
        # # x = F.feature_alpha_dropout(x, self.dropout_p, True, True)
        # x = self.net.layer4(x)
        #
        # x = self.net.avg_pool(x)
        # x = torch.flatten(x, 1)
        # x = self.net.fc(x)

        return x


class ResNet34(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.resnet34()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)
        # self.dropout_p = 0

    def forward(self, x):

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)

        # x = F.feature_alpha_dropout(x, self.dropout_p, True, True)

        x = self.net.layer4(x)

        # x = x[0, :, :]
        # x = x.view(64, -1).unsqueeze(0)
        # print(x.size())
        # torchvision.utils.save_image(x, "./output/BadNets(Model)-BadNets(Input).jpeg")
        # sys.exit(0)

        # x = F.dropout2d(x, self.dropout_p, True, True)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet50(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.resnet50()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)

    def forward(self, x):
        outputs = self.net(x)
        return outputs


class ResNet101(nn.Module):

    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.net = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.resnet101(pretrained=pretrain)
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class WideResNet50(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()
        if pre_train:
            self.net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.wide_resnet50_2()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class WideResNet101(nn.Module):

    def __init__(self, class_num, pre_train=True):
        super().__init__()
        if pre_train:
            self.net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        else:
            self.net = torchvision.models.wide_resnet50_2()
        self.net.fc = nn.Linear(self.net.fc.in_features, class_num)

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


if __name__ == '__main__':
    vae = VariationalAutoEncoder(param.simi_v_dim)
    inputs = torch.zeros([1, 1, 256, 64])
    print(vae(inputs)[0].size())

    # from torchviz import make_dot
    #
    # graph = make_dot(vae(inputs), params=dict(vae.named_parameters()))
    # graph.render("CNN_graph")

    # import hiddenlayer as hl
    #
    # vis_graph = hl.build_graph(vae, inputs)
    # vis_graph.theme = hl.graph.THEMES["blue"].copy()  # 指定主题颜色
    # vis_graph.save("./demo1.png")  # 保存图像的路径
