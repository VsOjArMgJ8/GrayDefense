import numpy as np

import param
from param import Model  # 单层全连接
from utils_ import data_loader as dl
from utils_.TAT import *

# 1. 实例化模型
backdoor_model = Model(param.classes_num, True)
backdoor_model.load_state_dict(torch.load(param.backdoor_weights_file_name))

# 测试集
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)
acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
asr_loader1 = dl.get_backdoor_loader(param.attack_method, test_set, test_tf, False, param.test_batch_size,
                                     1.0, param.target_label_type, param.target_label, param.classes_num)
# 2. 构建模型与优化器
criterion = torch.nn.CrossEntropyLoss()

print("=" * 100)
for p in np.arange(0, 1, 0.05):
    backdoor_model.dropout_p = p
    _, acc = test_model(acc_loader, backdoor_model, criterion, param.device, param.test_print_freq)
    # _, asr = test_model(asr_loader, backdoor_model, criterion, param.device, param.test_print_freq)
    # print(f"dropout_p:[{p:.2f}]   ACC:{acc:.4f}  ASR:{asr:.4f}")
    # print(f"{acc * 100:.2f}")
