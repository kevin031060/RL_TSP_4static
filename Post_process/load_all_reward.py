import torch
from tasks import motsp
from tasks.motsp import TSPDataset, reward
from torch.utils.data import DataLoader
from model import DRL4TSP
from trainer_motsp_transfer import StateCritic
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from Post_process.dis_matrix import dis_matrix
import time

# Load the trained model and convert the obtained Pareto Front to the .mat file.
# It is convenient to visualize it in matlab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# "../tsp_transfer_100run_500000_5epoch_20city/20"效果一般。应该再训练一遍
save_dir = "../tsp_transfer_100run_500000_5epoch_40city/40"
# save_dir = "../tsp_transfer/100"
# param
update_fn = None
STATIC_SIZE = 4  # (x, y)
DYNAMIC_SIZE = 1  # dummy for compatibility

# claim model
actor = DRL4TSP(STATIC_SIZE,
                DYNAMIC_SIZE,
                128,
                update_fn,
                motsp.update_mask,
                1,
                0.1).to(device)
critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, 128).to(device)

# data 143
from Post_process.convet_kro_dataloader import Kro_dataset
kro = 1
D = 200
if kro:
    D = 200
    Test_data = Kro_dataset(D)
    Test_loader = DataLoader(Test_data, 1, False, num_workers=0)
else:
    # 40city_train: city20 13 city40 143 city70 2523
    #
    Test_data = TSPDataset(D, 1, 2523)
    Test_loader = DataLoader(Test_data, 1, False, num_workers=0)

iter_data = iter(Test_loader)
static, dynamic, x0 = iter_data.next()
static = static.to(device)
dynamic = dynamic.to(device)
x0 = x0.to(device) if len(x0) > 0 else None

# load 50 models
N=100
w = np.arange(N+1)/N
objs = np.zeros((N+1,2))
start  = time.time()
t1_all = 0
t2_all = 0
tours=[]
for i in range(0, N+1):
    t1 = time.time()
    ac = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"actor.pt")
    cri = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"critic.pt")
    actor.load_state_dict(torch.load(ac, device))
    critic.load_state_dict(torch.load(cri, device))
    t1_all = t1_all + time.time()-t1
    # calculate

    with torch.no_grad():
        # t2 = time.time()
        tour_indices, _ = actor.forward(static, dynamic, x0)
        # t2_all = t2_all + time.time() - t2
    _, obj1, obj2 = reward(static, tour_indices, 1-w[i], w[i])
    tours.append(tour_indices.cpu().numpy())
    objs[i,:] = [obj1, obj2]

print("time_load_model:%2.4f"%t1_all)
print("time_predict_model:%2.4f"%t2_all)
print(time.time()-start)

print(tours)
plt.figure()
plt.plot(objs[:,0],objs[:,1],"ro")
plt.show()

# Convert to .mat
obj1_matrix, obj2_matrix = dis_matrix(static, STATIC_SIZE)
scio.savemat("data/obj1_%d_%d.mat"%(STATIC_SIZE, D), {'obj1':obj1_matrix})
scio.savemat("data/obj2_%d_%d.mat"%(STATIC_SIZE, D), {'obj2':obj2_matrix})
scio.savemat("data/rl%d_%d.mat"%(STATIC_SIZE, D),{'rl':objs})
scio.savemat("data/tour%d_%d.mat"%(STATIC_SIZE, D),{'tour':np.array(tours)})


# from load_test_plot import show
# show_if = 1
# if show_if:
#     i = 0
#     ac = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"actor.pt")
#     cri = os.path.join(save_dir, "w_%2.2f_%2.2f" % (1-w[i], w[i]),"critic.pt")
#     actor.load_state_dict(torch.load(ac, device))
#     critic.load_state_dict(torch.load(cri, device))
#
#     show(Test_loader, actor)

