import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class WaNetsDataset(Dataset):
    def __init__(self, full_dataset, train, inject_portion, class_num, target_label_type="all2one", target_label=0, transform=None):
        self.transform = transform
        self.class_num = class_num
        self.dataset = self.addTrigger(full_dataset, train, inject_portion, target_label_type, target_label)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, train, inject_portion, target_label_type, target_label):
        selected_index = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]
            if len(img.split()) == 1:
                img = img.convert("RGB")
            width, height = img.size
            label = data[1]
            # all2one attack
            if target_label_type == 'all2one':
                # test set do not contain the samples of target label in all to one
                if not train and label == target_label:
                    continue
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._warpingTrigger(img, width)
                    img = Image.fromarray(img, mode="RGB")
                    # change target
                    label = target_label
                    cnt += 1
            # all2all attack
            elif target_label_type == 'all2all':
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._warpingTrigger(img, width)
                    img = Image.fromarray(img)
                    # change target
                    label = self._change_label_next(label)
                    cnt += 1
            dataset_.append((img, label))

        print("Injecting Over: " + str(cnt) + "Bad Images, " + str(len(dataset_) - cnt) + "Clean Images")
        return dataset_

    def _change_label_next(self, label):
        return (label + 1) % self.class_num

    @staticmethod
    def _warpingTrigger(img, width):
        # WaNet
        array1d = torch.linspace(-1, 1, steps=width)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((x, y), 2)[None, ...]

        ins = torch.rand(1, 2, 32, 32) * 2 - torch.tensor(1)
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=width, mode="bicubic", align_corners=True).permute(0, 2, 3, 1)

        grid_temps = (identity_grid + noise_grid / width) * 4
        grid_temps = torch.clamp(grid_temps, -4, 4)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        inputs_bd = F.grid_sample(img, grid_temps, align_corners=True).squeeze(0)
        inputs_bd = torch.clamp(inputs_bd, 0, 255).permute(2, 1, 0)

        # img_ = Image.fromarray(np.array(inputs_bd.numpy(), dtype=np.uint8))
        # img_.save("./images/WaNet.png")
        # sys.exit(0)

        return np.array(inputs_bd.numpy(), dtype=np.uint8)
