"""
计算结果
"""
from utils import AUROC_Score
import numpy as np


def process(file_data):
    a = np.load(file_data)[:9000]

    score = np.empty(len(a))

    for i in range(len(score)):
        score[i] = np.mean(a[i] == a[i][0])

    print(score)
    return score


if __name__ == "__main__":
    AUROC_Score(
        process("./saved_np/WaNet/CIFAR10_bd.npy"),
        process("./saved_np/WaNet/CIFAR10_benign.npy"),
        "",
    )
