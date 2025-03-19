import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class BlendDataset(Dataset):
    def __init__(self, full_dataset, train, inject_portion, class_num, target_label_type="all2one", target_label=0, transform=None):

        self.dataset = self.addTrigger(full_dataset, train, inject_portion, target_label_type, target_label)
        self.transform = transform
        self.class_num = class_num

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
            label = data[1]
            # all2one attack
            if target_label_type == 'all2one':
                # test set do not contain the samples of target label in all to one
                if not train and label == target_label:
                    continue
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._blendTrigger(img)
                    img = Image.fromarray(img, mode="RGB")
                    # change target
                    label = target_label
                    cnt += 1
            # all2all attack
            elif target_label_type == 'all2all':
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._blendTrigger(img)
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
    def _blendTrigger(img):
        # blend
        gauss_noise = np.random.normal(0, 45, img.shape)
        img_arr = img.astype('int')
        img_arr = img_arr + gauss_noise
        img_arr = np.clip(img_arr, 0, 255)
        img_arr = img_arr.astype('uint8')

        # img_ = Image.fromarray(img_arr)
        # img_.save("./images/Blend.png")
        # sys.exit(0)

        return img_arr
