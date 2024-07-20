import torch
from datasets import IterableDataset
from transformers import GPT2TokenizerFast
from typing import Tuple


class DataLoader:

    def __init__(self, B: int, T: int, dataset: IterableDataset) -> None:
        self.B = B
        self.T = T
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.dataset = dataset
        self.reset()
    

    def reset(self) -> None:
        self.iterator = iter(self.dataset)
        self.buffer = [self.tokenizer.eos_token_id]
    

    def _fill_buffer(self) -> None:
        text = ''
        while len(self.buffer) < (self.B * self.T + 1):
            try:
                text += next(self.iterator)['text']
                self.buffer += self.tokenizer.encode(text, max_length=None)
                self.buffer.append(self.tokenizer.eos_token_id)
            except StopIteration:
                self.iterator = iter(self.dataset)
    
    
    def next_batch(self) -> Tuple[torch.Tensor]:
        self._fill_buffer()
        tokens = torch.tensor(self.buffer[:(self.B * self.T + 1)])
        self.buffer = self.buffer[self.B * self.T:]
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        return x, y