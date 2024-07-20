import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from gpt2 import GPT, GPTConfig
from dataloader import DataLoader

print('Loading dataset...')
dataset = load_dataset(path='HuggingFaceFW/fineweb-edu', split='train', streaming=True)
dataset = dataset.shuffle(seed=13)
train_dataset = dataset.skip(1000)
val_dataset = dataset.take(1000)

B, T = 4, 1024
train_data_loader = DataLoader(B=B, T=T, dataset=train_dataset)
val_data_loader = DataLoader(B=B, T=T, dataset=val_dataset)

x, y = train_data_loader.next_batch()
print(f'x: {x.shape}, y: {y.shape}')