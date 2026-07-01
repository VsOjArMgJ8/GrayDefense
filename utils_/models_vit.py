import torch.nn as nn
import torchvision.models as models


class ViTBase16(nn.Module):
    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = models.vit_b_16()

        # 分类头
        self.vit.heads = nn.Linear(self.vit.hidden_dim, class_num)

    def forward(self, x):
        x = self.vit._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        x = x[:, 0]

        x = self.vit.heads(x)
        return x


class ViTBase32(nn.Module):
    def __init__(self, class_num, pretrain=True):
        super().__init__()
        if pretrain:
            self.vit = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            self.vit = models.vit_b_32()

        # 分类头
        self.vit.heads = nn.Linear(self.vit.hidden_dim, class_num)

    def forward(self, x):
        x = self.vit._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        x = x[:, 0]

        x = self.vit.heads(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTFeatVAE(nn.Module):
    def __init__(
            self,
            seq_len=1,
            feat_dim=768,
            latent_dim=128,
            nhead=16,
            enc_layers=1,
            dec_layers=1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # ------------------- Encoder: 把序列压缩到隐变量 z -------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim * 2,
            dropout=0, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # 🔥 原来的池化删掉，替换成：把 [B, 196, 768] 展平 + 全连接
        self.flatten = nn.Flatten()  # [B, 196*768]
        self.fc_feat = nn.Linear(seq_len * feat_dim, feat_dim)  # 全连接映射回 feat_dim

        # 投影到隐空间
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # ------------------- Decoder: 从 z 恢复序列 -------------------
        self.z_proj = nn.Linear(latent_dim, feat_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, feat_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim * 2,
            dropout=0, activation="gelu", batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def encode(self, x):
        # x: [B, 196, 768]
        feat = self.transformer_encoder(x)  # [B, 196, 768]

        # 🔥 替换池化：展平 + 全连接得到全局特征
        feat_flat = self.flatten(feat)  # [B, 196*768]
        feat_pool = self.fc_feat(feat_flat)  # [B, 768]

        mu = self.fc_mu(feat_pool)
        logvar = self.fc_logvar(feat_pool)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        z_feat = self.z_proj(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        z_feat = z_feat + self.pos_emb

        memory = torch.zeros(B, 1, self.feat_dim).to(z.device)
        out = self.transformer_decoder(z_feat, memory)
        out = self.out_proj(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class FCVAE(nn.Module):
    """
    全连接层实现的VAE模型
    输入维度: (batch_size, 768)
    输出维度: (batch_size, 768)
    """

    def __init__(self, hidden_dims: list, latent_dim: int = 128):
        """
        初始化VAE模型
        Args:
            hidden_dims: 隐藏层维度列表，例如 [512, 256] 表示编码器两层隐藏层
            latent_dim: 潜变量z的维度（编码器输出的均值、方差维度）
        """
        super(FCVAE, self).__init__()
        self.latent_dim = latent_dim

        # ==================== 编码器：768 → hidden_dims → latent_dim*2（均值+对数方差）====================
        encoder_layers = []
        # 输入层：768 → 第一个隐藏层
        encoder_layers.append(nn.Linear(768, hidden_dims[0]))
        encoder_layers.append(nn.ReLU())  # 激活函数，可替换为LeakyReLU/GELU

        # 中间隐藏层：循环构建
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            encoder_layers.append(nn.ReLU())

        # 输出层：最后一个隐藏层 → 2*latent_dim（均值mu + 对数方差log_var）
        encoder_layers.append(nn.Linear(hidden_dims[-1], 2 * latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ==================== 解码器：latent_dim → reversed(hidden_dims) → 768 ====================
        decoder_layers = []
        # 输入层：latent_dim → 反转后的第一个隐藏层
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        # 中间隐藏层：反转隐藏层维度构建
        reversed_hidden = hidden_dims[::-1]  # 反转隐藏层
        for i in range(len(reversed_hidden) - 1):
            decoder_layers.append(nn.Linear(reversed_hidden[i], reversed_hidden[i + 1]))
            decoder_layers.append(nn.ReLU())

        # 输出层：最后一个隐藏层 → 768（输出与输入同维度）
        decoder_layers.append(nn.Linear(reversed_hidden[-1], 768))
        # 输出层不激活（适配连续型输入768维向量），如果是0-1归一化数据可加nn.Sigmoid()
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """编码器前向传播：输出均值和对数方差"""
        # x: (batch, 768)
        h = self.encoder(x)  # (batch, 2*latent_dim)
        mu, log_var = torch.chunk(h, 2, dim=-1)  # 拆分为均值和对数方差，均为 (batch, latent_dim)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """重参数化技巧：采样潜变量z"""
        std = torch.exp(0.5 * log_var)  # 标准差
        eps = torch.randn_like(std)  # 标准正态分布噪声
        return mu + eps * std  # z = μ + ε·σ

    def decode(self, z):
        """解码器前向传播：重构输入"""
        return self.decoder(z)  # (batch, 768)

    def forward(self, x):
        """模型整体前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss(recon_x, x, mu, logvar):
    # 重构损失（MSE 必须用）
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL 散度（一定要小！ViT 特征不适合强 KL）
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 核心：KL 权重 0.01~0.1，不能大
    return recon_loss + 0.05 * kl_div, recon_loss, kl_div


# ===================== 测试代码（128x128 完全正常）=====================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模型
    vit_model = ViTBase16(class_num=200, pretrain=True).to(device)
    vit_model.eval()

    dummy_input = torch.randn(3, 3, 224, 224).to(device)
    print("=" * 60)
    print(f"输入形状: {dummy_input.shape}")

    # 3. 前向测试
    with torch.no_grad():
        output = vit_model(dummy_input)
    print(f"ViT 输出: {output.shape}")

    # 4. 获取中间特征
    feature = torch.tensor(0)


    def hook(module, inp, out):
        global feature
        feature = out[:, 1:]


    handle = vit_model.vit.encoder.layers[-2].register_forward_hook(hook)
    with torch.no_grad():
        vit_model(dummy_input)
    handle.remove()

    print(f"ViT 中间特征 shape: {feature.size()}")  # [1, 65, 768]

    # 初始化 VAE
    vae = ViTFeatVAE(seq_len=196, feat_dim=768, latent_dim=128).to(device)

    # 前向
    recon_feat, mu, logvar = vae(feature)
    loss, recon_loss, kl_loss = vae_loss(recon_feat, feature, mu, logvar)

    print("ViTFeatVAE 输入特征:", feature.shape)
    print("ViTFeatVAE 重构特征:", recon_feat.shape)
    print(f"总损失: {loss.item():.2f}")
    print(f"重构损失: {recon_loss.item():.2f}")
    print(f"KL散度: {kl_loss.item():.2f}")
