"""
This is the code of obtaining samples from a given dataloader and save them as a tensor.
3. 对模型进行测试
"""

import argparse

import numpy as np
import torch
import torchvision

parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument(
    "--gpu-id", default="0,1", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu_id

# data_samples = "./data/WaNet/poisoned_test_samples.pth"
data_samples = "./data/WaNet/benign_test_samples.pth"


device = torch.device("cuda:0")
resnet34 = torchvision.models.resnet18(pretrained=False)
resnet34.fc = torch.nn.Linear(512, 100)
model = resnet34
model.load_state_dict(torch.load("./data/WaNet/ckpt_epoch_10.pth", map_location=device))
model.to(device)
model.eval()

poisoned_test_samples = torch.load(data_samples, map_location=device)

# adding random noise
poisoned_test_samples = poisoned_test_samples + 0.02 * torch.rand(
    size=poisoned_test_samples.shape, device=device
)

labels = torch.load("./data/WaNet/benign_labels.pth")

decisions = np.empty((20000, 11))

for i in range(90):
    img_batch = poisoned_test_samples[i * 200: (i + 1) * 200]
    img_batch.to(device)
    # print(img_batch)
    # evals = 0.1*torch.randn(100, 3, 32, 32,device=device)

    for h in range(1, 12):
        img_batch_re = torch.clamp(h * img_batch, 0, 1)
        decisions[i * 200: (i + 1) * 200, (h - 1)] = (
            torch.max(model(img_batch_re), 1)[1].detach().cpu().numpy()
        )

print(decisions)
print(decisions.shape)
print(np.mean(decisions[:, 0] == np.reshape(labels.numpy(), 20000)))
a = decisions[decisions[:, 0] == np.reshape(labels.numpy(), 20000)]
print(a.shape)
# np.save("./saved_np/WaNet/CIFAR100_bd.npy", decisions)
np.save("./saved_np/WaNet/CIFAR100_benign.npy", decisions)
