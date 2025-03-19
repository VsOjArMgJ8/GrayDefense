import param
from param import Model  # 单层全连接
from utils_ import data_loader as dl
from utils_.TAT import *

# 1. 实例化模型
model = Model(param.classes_num, True)
model.load_state_dict(torch.load(param.backdoor_weights_file_name))  # backdoor
# model.load_state_dict(torch.load(param.benign_weights_file_name))  # benign

# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
# acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
asr_loader1 = dl.get_backdoor_loader("BadNets", test_set, test_tf, False, param.test_batch_size,
                                     1.0, param.target_label_type, param.target_label, param.classes_num)

# 2. 构建模型与优化器
criterion = torch.nn.CrossEntropyLoss()
print("=" * 100)

# test_model(acc_loader, model, criterion, param.device, param.test_print_freq)
test_model(asr_loader1, model, criterion, param.device, param.test_print_freq)
