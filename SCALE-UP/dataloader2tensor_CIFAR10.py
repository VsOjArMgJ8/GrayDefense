"""
This is the code of obtaining samples from a given dataloader and save them as a tensor.
2. 构造测试样本
"""

import torch



# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

dataloader_root_dir = './data/WaNet/poisoned_test_dataset.pth'
# dataloader_root_dir = './data/WaNet/benign_test_dataset.pth'

poisoned_test_dataloader = torch.load(dataloader_root_dir)
poisoned_test_samples = torch.zeros((20000, 3, 128, 128))
labels = torch.zeros((20000, 1))

for batch_id, batch in enumerate(poisoned_test_dataloader):
    if batch_id == 20000:
        break
    batch_img = batch[0]
    batch_label = batch[1]
    poisoned_test_samples[batch_id, :, :, :] = batch_img
    labels[batch_id, :] = batch_label
    if (batch_id + 1) % 100 == 0:
        print((batch_id + 1) / 100)

torch.save(poisoned_test_samples, './data/WaNet/poisoned_test_samples.pth')

# torch.save(poisoned_test_samples, './data/WaNet/benign_test_samples.pth')
# torch.save(labels, './data/WaNet/benign_labels.pth')

print('finished')
