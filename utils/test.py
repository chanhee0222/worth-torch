import os
from collections import defaultdict

import numpy as np
import torch


def test():
    bases_param = torch.arange(24).reshape(3, 2, 4).to(torch.float32)
    print("Bases:")
    print(bases_param)

    weight_param = torch.arange(12).reshape(4, 3).to(torch.float32)
    print("Weight_P")
    print(weight_param)

    input_vec = torch.Tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    outputs = []
    print("Weights:")
    for vec in input_vec:
        weights = torch.matmul(vec, weight_param)
        print(weights)
        reduced = weights.reshape(-1, 1, 1) * bases_param
        reduced = reduced.sum(0)
        outputs.append(reduced)

    reduced = torch.stack(outputs)
    print("Reduced:")
    print(reduced)

    input_vec = torch.arange(16).reshape(2, 2, 4).to(torch.float32)
    weights = torch.matmul(input_vec, weight_param).unsqueeze(2)
    print(weights.shape)
    print(weights)

    bases_param_r = bases_param.reshape(1, 1, 3, -1)
    reduced2 = torch.matmul(weights, bases_param_r).reshape(2, 2, 2, 4)
    print(reduced2.shape)

    # print(bases_param.permute([1, 2, 0]).shape)
    # reduced2 = torch.matmul(bases_param.permute([1, 2, 0]), weights)
    # assert np.allclose(reduced, reduced2)

    # base_weight_mm = torch.matmul(bases_param.permute([1, 2, 0]), weight_param.t())
    # print(base_weight_mm.shape)
    # reduced3 = torch.matmul(base_weight_mm.permute([0, 1, 2]), input_vec.t()).permute([0, 2, 1])
    # print(reduced3)
    #
    # assert np.allclose(reduced, reduced3)


if __name__ == "__main__":
    # compare_logs()
    test()