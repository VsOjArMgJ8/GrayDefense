from torch import nn


class LeeNet_5_pro(nn.Module):
    def __init__(self):
        super().__init__()

        # ------------------------
        # 定义Mnist分类模型的网络-leNet
        # ------------------------
        # --------------------
        # 定义fashion Mnist分类模型的网络
        # --------------------
        self.conv_blocks_0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),  # 28 * 28  --->  28 * 28
            nn.MaxPool2d(2),  # 28 * 28  --->  14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.identical_conv_block_0 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(1, 1)),
            nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1),
            nn.Conv2d(8, 8, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3)),  # 14 * 14  --->  12 * 12
            nn.ReLU(),
        )

        self.identical_conv_block_1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1),
            nn.Conv2d(16, 16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.conv_blocks_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),  # 12 * 12  --->  10 * 10
            nn.MaxPool2d(2),  # 10 * 10  --->  5 * 5
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.identical_conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.Conv2d(32, 32, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.fc_blocks = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs):
        # 获取输入数据的Batch
        batch_size = inputs.size(0)

        # 卷积
        x_0 = self.conv_blocks_0(inputs)
        x_1 = self.identical_conv_block_0(x_0) + x_0
        x_1 = self.conv_blocks_1(x_1)
        x_2 = self.identical_conv_block_1(x_1) + x_1
        x_2 = self.conv_blocks_2(x_2)
        x = self.identical_conv_block_2(x_2) + x_2

        # 将卷积之后的二维特征空间转为一维
        x = x.view(batch_size, -1)

        # 全连接
        outputs = self.fc_blocks(x)
        return outputs
