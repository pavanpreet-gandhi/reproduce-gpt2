import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from gpt2 import GPT, GPTConfig
from dataloader import DataLoader


# Device and manual seed
torch.manual_seed(13)
torch.cuda.manual_seed(13)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device.type)


# Load and split dataset
print('Loading dataset...')
dataset = load_dataset(path='HuggingFaceFW/fineweb-edu', split='train', streaming=True)
dataset = dataset.shuffle(seed=13)
train_dataset = dataset.skip(1000)
val_dataset = dataset.take(1000)


# Set learning rate schedule
max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10
num_steps = 50

def lr_scheduler(step: int) -> float:
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    elif step < num_steps:
        return min_lr + (max_lr - min_lr) * (1 + math.cos((step - warmup_steps) / (num_steps - warmup_steps) * math.pi)) / 2
    else:
        return min_lr

filepath = 'logs/lr_schedule.png'
plt.figure(figsize=(10, 6))
plt.plot([lr_scheduler(step) for step in range(0, num_steps + 1)])
plt.title('Learning rate schedule')
plt.xlabel('Step')
plt.ylabel('Learning rate')
plt.savefig(filepath)
print(f'Learning rate schedule saved to {filepath}')


# Set batch sizes
B, T = 4, 1024
total_batch_size = 524288 # 2**19
assert total_batch_size % (B * T) == 0, 'Total batch size must be divisible by B * T'


# Create data loaders
train_data_loader = DataLoader(B=B, T=T, dataset=train_dataset)
val_data_loader = DataLoader(B=B, T=T, dataset=val_dataset)


# Create model
config = GPTConfig(vocab_size=50304)
model = GPT(config).to(device)


# Configure optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
weight_decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
non_weight_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': weight_decay_params, 'weight_decay': 0.01},
    {'params': non_weight_decay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=1.0, betas=(0.9, 0.95), eps=1e-8, fused=True)


# Training loop
pass