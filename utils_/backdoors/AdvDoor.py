import copy

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm


class AdvDoor:
    def __init__(self, model, train_dataset, test_dataloader, use_cuda, target_class=0, p_samples=0.05):

        assert 0 < p_samples <= 1, "The ratio can should be in range (0,1]"

        self.model = model
        self.use_cuda = use_cuda
        self.target_class = target_class
        self.train_set = train_dataset
        print("train set", len(self.train_set))
        self.test_loader = test_dataloader
        print("test set", len(self.test_loader))
        self.p_samples = p_samples

        self.num_samples = int(self.p_samples * len(self.train_set)) + 1
        print("self.num_samples", self.num_samples)

    def deep_fool_target(self, image, num_classes, overshoot, max_iter):
        f_image = self.model(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()  # [10,]
        I = f_image.argsort()[::-1]

        I = I[0:num_classes]
        clean_label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)

        r_tot = np.zeros(input_shape)

        loop_i = 0
        # wrapped = tqdm(total=max_iter)

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = self.model(x)
        k_i = clean_label
        while k_i != self.target_class and loop_i < max_iter:
            fs[0, self.target_class].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            x.grad.zero_()

            fs[0, clean_label].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            # add mask
            w_k = grad_orig - cur_grad
            f_k = (fs[0, self.target_class] - fs[0, clean_label]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            pert = pert_k
            # update description and progress bar

            # wrapped.set_description(f"perturbation: {pert:.5f}")
            # wrapped.update(1)
            print(f"perturbation: {pert:.5f}")

            w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            if self.use_cuda:
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            else:
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

            x = Variable(pert_image, requires_grad=True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        return (1 + overshoot) * r_tot, loop_i, k_i, pert_image

    def proj_lp(self, perturbation, epsilon, p_norm):

        if p_norm == 2:
            perturbation = perturbation * min(1, epsilon / np.linalg.norm(perturbation.flatten(1)))
        elif p_norm == np.inf:
            perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), epsilon)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')
        return perturbation

    def universal_perturbation(self, transformer, num_classes, delta=0.2, max_iter_uni=2, epsilon=0.9,
                               p_norm=np.inf, overshoot=0.2, max_iter_df=10):
        if self.use_cuda:
            self.model.cuda()
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.eval()

        v = torch.tensor(0)
        fooling_rate = 0.0
        # random.seed(seed)
        total_num = len(self.train_set)
        # Using #num_images data for generating UAP
        num_images = min(total_num, self.num_samples)
        tmp_list = list(range(total_num))
        np.random.shuffle(tmp_list)
        order = np.array(tmp_list[:num_images])

        itr = 0
        while fooling_rate < 1 - delta and itr < max_iter_uni:
            # Shuffle the self.trainset
            np.random.shuffle(order)
            print('Starting pass number ', itr)
            # Go through the data set and compute the perturbation increments sequentially
            for k in order:
                cur_img, _ = self.train_set[k]
                if len(cur_img.split()) == 1:
                    cur_img = torchvision.transforms.Grayscale(num_output_channels=3)(cur_img)
                cur_img = transformer(cur_img)
                perturb_img = cur_img + v

                cur_img, perturb_img = cur_img.to(device), perturb_img.to(device)
                if int(self.model(cur_img.unsqueeze(0)).max(1)[1]) == \
                        int(self.model((perturb_img.unsqueeze(0)).type(torch.cuda.FloatTensor)).max(1)[1]):

                    print('>> k = ', np.where(k == order)[0][0], ', pass #', itr)

                    # Compute adversarial perturbation
                    dr, iterr, _, _ = self.deep_fool_target(perturb_img, num_classes=num_classes,
                                                            overshoot=overshoot, max_iter=max_iter_df)

                    dr = torch.from_numpy(dr).squeeze(0).type(torch.float32)
                    # Make sure it converged...
                    if iterr < max_iter_df - 1:
                        v = v + dr
                        # Project on l_p ball
                        v = self.proj_lp(v, epsilon, p_norm)

            itr = itr + 1
            # Perturb the self.testset with computed perturbation and test the fooling rate on the testset
            with torch.no_grad():
                print("Testing")
                test_num_images = 0
                est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
                est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))
                for batch_idx, (inputs, _) in enumerate(self.test_loader):
                    test_num_images += inputs.shape[0]
                    inputs_pert = inputs + v
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    inputs_pert = inputs_pert.to(device)
                    outputs_perturb = self.model(inputs_pert)

                    _, predicted = outputs.max(1)
                    _, predicted_pert = outputs_perturb.max(1)
                    est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
                    est_labels_pert = torch.cat((est_labels_pert, predicted_pert.cpu()))
                torch.cuda.empty_cache()

                fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert)) / float(test_num_images)

                # Compute the fooling rate
                print('FOOLING RATE = ', fooling_rate)
        print('Final FOOLING RATE = ', fooling_rate)
        return v


class AdvDoorDataset(Dataset):
    def __init__(self, full_dataset, train, inject_portion, class_num, pattern, target_label_type="all2one", target_label=0, transform=None):
        self.dataset = self.addTrigger(full_dataset, train, inject_portion, pattern, target_label_type, target_label)
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

    def addTrigger(self, dataset, train, inject_portion, pattern, target_label_type, target_label):
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
                    img = self._targetUapTrigger(img, pattern)
                    img = Image.fromarray(img, mode="RGB")
                    # change target
                    label = target_label
                    cnt += 1
            # all2all attack
            elif target_label_type == 'all2all':
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._targetUapTrigger(img, pattern)
                    img = Image.fromarray(img)
                    # change target
                    label = self._change_label_next(label)
                    cnt += 1
            elif target_label_type == 'clean label':
                if i in selected_index:
                    img = np.array(img)
                    # select trigger
                    img = self._targetUapTrigger(img, pattern)
                    img = Image.fromarray(img)
                    cnt += 1
            dataset_.append((img, label))

        print("Injecting Over: " + str(cnt) + "Bad Images, " + str(len(dataset_) - cnt) + "Clean Images")
        return dataset_

    def _change_label_next(self, label):
        return (label + 1) % self.class_num

    @staticmethod
    def _targetUapTrigger(img, pattern):
        pattern = np.array(pattern.resize((img.shape[0], img.shape[1])))
        img = np.clip(img + pattern, a_min=0, a_max=255)

        # img_ = Image.fromarray(img)
        # img_.save("./images/AdvDoor.png")
        # sys.exit(0)

        return img.astype(np.uint8)
