import param
from param import Model
from utils_ import data_loader as dl

from utils_.TAT import *

# 1. 实例化个模型
benign_model = Model(param.classes_num, True)

# 训练集
train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
train_loader = dl.get_initial_loader(train_set, train_tf, True, param.train_batch_size, 1.0)

# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

# 2. 构建模型与优化器
benign_model.to(device=param.device)
optimizer = torch.optim.AdamW(benign_model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 25
# 3. 训练5轮
for epoch in range(epochs):
    train_model(train_loader, benign_model, optimizer, criterion, epoch + 1, param.device, param.train_print_freq)
    _, acc = test_model(acc_loader, benign_model, criterion, param.device, param.test_print_freq)
    print(f"Benign Model, Epoch:[{epoch + 1}/{epochs}]   ACC:{acc:.4f}")

torch.save(benign_model.state_dict(), param.model_weights_file_name)
print("=========> save benign model weights_ to: " + param.model_weights_file_name)
