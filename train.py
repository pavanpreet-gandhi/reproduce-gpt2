import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from gpt2 import GPT, GPTConfig
from dataloader import DataLoader


# Configure logging
log_dir = 'logs'
checkpoint_dir = f'{log_dir}/checkpoints'
logging.basicConfig(
    filename=f'{log_dir}/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Device and manual seed
torch.manual_seed(13)
torch.cuda.manual_seed(13)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using {device.type} device for training')


# Load and split dataset
dataset = load_dataset(
    path='HuggingFaceFW/fineweb-edu', 
    name='default', # the full most recent version of the dataset with over 1.3 trillion GPT-2 tokens
    split='train', # there is only this one split
    streaming=True
)
logging.info(f'Dataset loaded with {dataset.dataset_size} text samples')
dataset = dataset.shuffle(seed=13)
train_dataset = dataset.skip(1000)
val_dataset = dataset.take(1000)


# Set learning rate schedule
max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10 # 715
decay_steps = 40 # int(10e9 / 2**19) - warmup_steps # roughly over 1 billion tokens
num_steps = 50 # int(10e9 / 2**19) * 2 # roughly over 2 billion tokens
assert num_steps >= (warmup_steps + decay_steps), 'Total number of steps must be greater than warmup steps + decay steps'
logging.info(f'Learning rate schedule: max_lr={max_lr}, min_lr={min_lr}, warmup_steps={warmup_steps}, decay_steps={decay_steps}, num_steps={num_steps}')

def lr_scheduler(step: int) -> float:
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    elif step < (warmup_steps + decay_steps):
        return min_lr + (max_lr - min_lr) * (1 + math.cos(step / decay_steps * math.pi)) / 2
    else:
        return min_lr

filepath = f'{log_dir}/lr_schedule.png'
plt.figure(figsize=(10, 6))
plt.plot([lr_scheduler(step) for step in range(0, num_steps + 1)])
plt.title('Learning rate schedule')
plt.xlabel('Step')
plt.ylabel('Learning rate')
plt.savefig(filepath)
logging.info(f'Learning rate schedule graph saved to {filepath}')


# Set batch sizes
B, T = 4, 1024
total_batch_size = 2**19
assert total_batch_size % (B * T) == 0, 'Total batch size must be divisible by B * T'
grad_accum_steps = total_batch_size // (B * T)
logging.info(f'Batch sizes: B={B}, T={T}, grad_accum_steps={grad_accum_steps}')


# Create data loaders
train_data_loader = DataLoader(B=B, T=T, dataset=train_dataset)
val_data_loader = DataLoader(B=B, T=T, dataset=val_dataset)
logging.info('Data loaders created')


# Create model
config = GPTConfig(vocab_size=50304)
model = GPT(config).to(device)
logging.info(f'Model created with config: {config}')


# Configure optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
weight_decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
non_weight_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
param_groups = [
    {'params': weight_decay_params, 'weight_decay': 0.01},
    {'params': non_weight_decay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, 0.95), eps=1e-8, fused=True)
logging.info(f'Optimizer created with AdamW algorithm')

# Computational optimizations
logging.info('Compiling model...')
model = torch.compile(model)
scaler = torch.cuda.amp.GradScaler() # required when autocasting to float16
# torch.set_float32_matmul_precision('high') # requires compute capability >= 8.0


# Training loop
checkpoint_freq = 1000
validation_freq = 250
validation_steps = 20

for step in range(num_steps):
    t0 = time.time()

    loss_accum = torch.tensor(0.0, device=device, requires_grad=False)
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):

        # forward pass
        x, y = train_data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16): # casting to bfloat16 requires compute capability >= 8.0
            logits, loss = model(x, targets=y)

        # backward pass
        loss = loss / grad_accum_steps # loss is averaged over each micro batch in the batch
        loss_accum += loss.detach() # accumulate loss for logging
        scaler.scale(loss).backward() # gradients are scaled to prevent underflow and accumulated over each call of backward()
    
    # optimizer step
    lr = lr_scheduler(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    scaler.unscale_(optimizer) # unscale the gradients of the optimizer's assigned params in-place
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
    scaler.step(optimizer) # optimizer step
    scaler.update() # updates the scale for next iteration

    # logging
    torch.cuda.synchronize() # wait for everything to finish running
    t1 = time.time()
    dt = (t1 - t0) * 1000 # in milliseconds
    throughput = (B * T * grad_accum_steps) / (dt / 1000) # tokens per second
    logging.info(f"step {step:4d} | loss {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms | throughput: {throughput:.2f} tps")

    # save model checkpoint
    if step % checkpoint_freq == 0:
        filepath = f'{checkpoint_dir}/checkpoint_{step}.pt'
        torch.save(model.state_dict(), filepath)
        logging.info(f'Model checkpoint saved to {filepath}')
    
    # validation
    if step % validation_freq == 0:
        t0 = time.time()
        model.eval()
        val_loss_accum = torch.tensor(0.0, device=device, requires_grad=False)
        with torch.no_grad():
            for _ in range(validation_steps):
                x, y = val_data_loader.next_batch()
                x, y = x.to(device), y.to(device)
                logits, val_loss = model(x, targets=y)
                val_loss = val_loss / validation_steps
                val_loss_accum += val_loss.detach()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        logging.info(f"Validation loss: {val_loss_accum:.6f} | dt: {dt:.2f} ms")
        model.train()