from torch.optim import AdamW

import param
from param import Model
from utils_ import data_loader as dl
from utils_.TAT import *

# 1. 实例化第一个模型
backdoor_model = Model(param.classes_num, param.pre_train)

# 向后门模型中，植入第二个后门
backdoor_model.load_state_dict(torch.load(param.backdoor_weights_file_name))
param.attack_method = "BadNets"

# 训练集【后门】
train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
train_loader = dl.get_backdoor_loader(param.attack_method, train_set, train_tf, True, param.train_batch_size, param.inject_rate, param.target_label_type, param.target_label, param.classes_num)

# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
asr_loader1 = dl.get_backdoor_loader(param.attack_method, test_set, test_tf, False, param.test_batch_size, 1.0, param.target_label_type, param.target_label, param.classes_num)

# 2. 构建模型与优化器
backdoor_model.to(device=param.device)
optimizer = AdamW(backdoor_model.parameters(), lr=param.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 3. 训练5轮
for epoch in range(param.train_epoch):
    loss, _ = train_model(train_loader, backdoor_model, optimizer, criterion, epoch + 1, param.device, param.train_print_freq)
    _, acc = test_model(acc_loader, backdoor_model, criterion, param.device, param.test_print_freq)
    _, asr = test_model(asr_loader1, backdoor_model, criterion, param.device, param.test_print_freq)
    print(f"Test         Epoch:[{epoch + 1}/{param.train_epoch}]   ACC:{acc:.4f}  ASR:{asr:.4f}\n")
    if loss < 0.1 and epoch > 5:
        break

torch.save(backdoor_model.state_dict(), param.backdoor_weights_file_name)
print("=========> save backdoor model weights_ to: " + param.backdoor_weights_file_name)
del backdoor_model, train_loader
