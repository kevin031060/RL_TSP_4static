import torch
import os
from model import DRL4TSP, Encoder
import argparse
from tasks import motsp
from trainer_motsp import StateCritic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATIC_SIZE_original = 2  # (x, y)
STATIC_SIZE = 3  # (x, y)
DYNAMIC_SIZE = 1  # dummy for compatibility
update_fn = None
hidden_size = 128
num_layers = 1
dropout = 0.1
checkpoint = "tsp20"
actor = DRL4TSP(STATIC_SIZE_original,
                DYNAMIC_SIZE,
                hidden_size,
                update_fn,
                motsp.update_mask,
                num_layers,
                dropout).to(device)

critic = StateCritic(STATIC_SIZE_original, DYNAMIC_SIZE, hidden_size).to(device)
# 加载原128*2*1的原模型
path = os.path.join(checkpoint, 'actor.pt')
actor.load_state_dict(torch.load(path, device))

path = os.path.join(checkpoint, 'critic.pt')
critic.load_state_dict(torch.load(path, device))
# 其中actor的static_encoder，decoder需要更改维度，critic需要更改维度
# static_encoder
static_parameter = actor.static_encoder.state_dict()
temp = static_parameter['conv.weight']
temp = torch.cat([temp, temp[:,1,:].unsqueeze(1)], dim=1)   # 在第二维拓展一列
static_parameter['conv.weight'] = temp
actor.static_encoder = Encoder(STATIC_SIZE, hidden_size)
actor.static_encoder.load_state_dict(static_parameter)
# decoder
static_parameter = actor.decoder.state_dict()
temp = static_parameter['conv.weight']
temp = torch.cat([temp, temp[:,1,:].unsqueeze(1)], dim=1)   # 在第二维拓展一列
static_parameter['conv.weight'] = temp
actor.decoder = Encoder(STATIC_SIZE, hidden_size)
actor.decoder.load_state_dict(static_parameter)

# CRITIC
static_parameter = critic.static_encoder.state_dict()
temp = static_parameter['conv.weight']
temp = torch.cat([temp, temp[:,1,:].unsqueeze(1)], dim=1)   # 在第二维拓展一列
static_parameter['conv.weight'] = temp
critic.static_encoder = Encoder(STATIC_SIZE, hidden_size)
critic.static_encoder.load_state_dict(static_parameter)

save_path = os.path.join("modified_checkpoint_3obj", 'actor.pt')
torch.save(actor.state_dict(), save_path)
save_path = os.path.join("modified_checkpoint_3obj", 'critic.pt')
torch.save(critic.state_dict(), save_path)

print(actor,critic)
