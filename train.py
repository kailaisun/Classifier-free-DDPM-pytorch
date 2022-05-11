from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer,diffusionnet

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import  torch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# )#.cuda()

import torch
from torch.utils.data import TensorDataset,DataLoader
action=torch.tensor(torch.load('ac.pth'))   #[-1,1] [-1,1]
observation=torch.tensor(torch.load('ob.pth'))
# print(torch.min(action,0))
# print(max(action[:,0]),min(action[:,0]))
train_ids=TensorDataset(action,observation)

ac_dim=len(action[0])
ob_dim=len(observation[0])

model = diffusionnet(
    dim=8,   #time
    cdim_start=ob_dim,  #c label
    in_dim=ac_dim,
    w=3
).to(device)
diffusion = GaussianDiffusion(
    model,
    image_size = ac_dim,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).to(device)

trainer = Trainer(
    diffusion,
    train_ids,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True   ,                     # turn on mixed precision
    w=3
)

trainer.train()