import numpy as np
import torch

def dis_matrix(static, s_size):
    static = static.squeeze(0)

    # [2,20]
    obj1 = static[:2, :]
    # [20]
    obj2 = static[2:, :]

    l = obj1.size()[1]
    obj1_matrix = np.zeros((l, l))
    obj2_matrix = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i != j:
                obj1_matrix[i,j] = torch.sqrt(torch.sum(torch.pow(obj1[:, i] - obj1[:, j], 2))).detach()
                if s_size == 3:
                    obj2_matrix[i, j] = torch.abs(obj2[i] - obj2[j]).detach()
                else:
                    obj2_matrix[i, j] = torch.sqrt(torch.sum(torch.pow(obj2[:, i] - obj2[:, j], 2))).detach()

    return obj1_matrix, obj2_matrix