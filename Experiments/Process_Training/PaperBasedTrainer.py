import os
os.environ["MKL_WARNING_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # Add this too for MacOS
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
from AutoregressiveArithmeticTransformer import create_arithmetic_transformer

"""
Model Configurations and Parameters
---------------------------------

MODEL SIZES:
-----------
Tiny (~26K parameters):
    embed_size: 32
    num_heads: 2
    ff_dim: 128
    num_layers: 2
    dropout: 0.1
    batch_size: 32
    learning_rate: 2e-3

Small (~105K parameters):
    embed_size: 64
    num_heads: 2
    ff_dim: 256
    num_layers: 2
    dropout: 0.1
    batch_size: 32
    learning_rate: 1e-3

Medium (~1.8M parameters):
    embed_size: 256
    num_heads: 4
    ff_dim: 1024
    num_layers: 4
    dropout: 0.1
    batch_size: 64
    learning_rate: 3e-4

Large (~11M parameters):  # Your current config
    embed_size: 512
    num_heads: 8
    ff_dim: 2048
    num_layers: 8
    dropout: 0.1
    batch_size: 128
    learning_rate: 1e-4

XLarge (~45M parameters):
    embed_size: 1024
    num_heads: 16
    ff_dim: 4096
    num_layers: 12
    dropout: 0.1
    batch_size: 256
    learning_rate: 5e-5

XXLarge (~175M parameters):
    embed_size: 2048
    num_heads: 32
    ff_dim: 8192
    num_layers: 16
    dropout: 0.1
    batch_size: 512
    learning_rate: 1e-5

HARDWARE REQUIREMENTS:
--------------------
Tiny/Small: Single GPU (4GB VRAM)
Medium: Single GPU (8GB VRAM)
Large: Single GPU (16GB VRAM)
XLarge: Multi-GPU (32GB+ VRAM)
XXLarge: Multi-GPU (64GB+ VRAM)

TRAINING PARAMETERS:
------------------
Common Settings:
    vocab_size: 20       # 0-9 plus special tokens
    max_length: 512      # Maximum sequence length
    warmup_steps: 1000
    max_epochs: 5-20 (smaller models need fewer epochs)
    optimizer: AdamW
    weight_decay: 0.1
    betas: (0.9, 0.95)
    grad_clip: 1.0

Note: For arithmetic learning tasks, the Medium size (~1.8M params) 
often provides the best balance of performance and computational efficiency.
"""

class Config:
    def __init__(self):
        # Model Architecture
        self.vocab_size = 32  # Updated to match actual vocab size (0-31)
        self.embed_size = 256
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_layers = 4
        self.max_length = 512
        self.dropout = 0.1
        
        # Training Parameters
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.max_epochs = 5
        self.warmup_steps = 1000
        self.grad_clip = 1.0
        
        # Dataset Parameters
        self.max_digit_length = 3
        self.train_samples = 100000
        self.val_samples = 1000
        self.train_seed = 42
        self.val_seed = 43
        
        # Save Parameters
        self.save_dir = "checkpoints"
        self.model_name = "arithmetic_transformer"
        self.save_every = 5000
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
class ProcessAdditionDataset(Dataset):
    def __init__(self, config, num_samples, seed):
        self.config = config
        self.num_samples = num_samples
        self.max_length = config.max_length
        
        # Setup vocabulary with new tokens
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({
            '+': 12, '>': 13, 
            'A': 14, 'C': 15,
            '-': 16, ',': 17,
            ' ': 18, '\n': 19,
            ':': 20,  # Added for "Input:" and "Target:"
            'I': 21, 'n': 22, 'p': 23, 'u': 24, 't': 25,
            'T': 26, 'a': 27, 'r': 28, 'g': 29, 'e': 30,
            '.': 31
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        random.seed(seed)
        self.data = self.generate_dataset()
    
    def generate_sequence(self, num1, num2):
        # Generate properly formatted input line
        input_str = f"Input: {num1}+{num2}\n"
        target_str = "Target:\n"
        
        # Calculate single digit addition
        total = num1 + num2
        digit, carry = total % 10, total // 10
        
        # Format carry step with period
        step_str = f"A->{digit}, C->{carry}.\n"
        
        # Add final result
        result_str = str(total)
        
        # Combine all parts
        complete_str = input_str + target_str + step_str + result_str
        
        # Convert to tokens
        tokens = []
        for c in complete_str:
            if c in self.vocab:
                tokens.append(self.vocab[c])
        
        return torch.tensor(tokens, dtype=torch.long)

    def generate_dataset(self):
        data = []
        pbar = tqdm(total=self.num_samples, desc="Generating samples")
        
        while len(data) < self.num_samples:
            # Generate simple single digit numbers for testing
            num1 = random.randint(0, 9)
            num2 = random.randint(0, 9)
            
            sequence = self.generate_sequence(num1, num2)
            if sequence is not None and len(sequence) <= self.max_length:
                data.append(sequence)
                pbar.update(1)
        
        pbar.close()
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = sequence[:-1]
        y = sequence[1:]
        return x, y

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(input_seq)] = input_seq
        padded_targets[i, :len(target_seq)] = target_seq
    
    return padded_inputs, padded_targets

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = create_arithmetic_transformer(config).to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95)
        )
        
        self.train_dataset = ProcessAdditionDataset(
            config, config.train_samples, config.train_seed
        )
        self.val_dataset = ProcessAdditionDataset(
            config, config.val_samples, config.val_seed
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        self.global_step = 0
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            logits, loss = self.model(inputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            
            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, Acc: {total_correct/total_tokens:.4f}"
            )
            
            self.global_step += 1
            
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
        
        return total_loss / len(self.train_loader), total_correct / total_tokens
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            logits, loss = self.model(inputs, targets)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            
        return total_loss / len(self.val_loader), total_correct / total_tokens

    def save_checkpoint(self, is_best=False, is_final=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'vocab': self.train_dataset.vocab,
            'inv_vocab': self.train_dataset.inv_vocab
        }
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Regular checkpoint
        path = os.path.join(
            self.config.save_dir,
            f"{self.config.model_name}_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, path)
        
        # Save best model if this is the best validation accuracy
        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
        # Save final model
        if is_final:
            final_path = os.path.join(self.config.save_dir, "final_model.pt")
            torch.save(checkpoint, final_path)
    
    def train(self):
        print("Starting training...")
        try:
            for epoch in range(self.config.max_epochs):
                print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
                
                train_loss, train_acc = self.train_epoch()
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
                val_loss, val_acc = self.validate()
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save regular checkpoint
                self.save_checkpoint()
                
                # Save best model if this is the best validation accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint(is_best=True)
            
            # Save final model
            self.save_checkpoint(is_final=True)
            print("Training completed!")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint()
            
            
def test_dataset_generation():
    config = Config()
    dataset = ProcessAdditionDataset(config, num_samples=5, seed=21)
    
    print("Sample Addition Sequences:")
    print("-" * 50)
    
    for i, seq in enumerate(dataset.data[:5]):
        sequence = ''.join([dataset.inv_vocab[token.item()] for token in seq])
        print(f"\nExample {i+1}:")
        print(sequence)
        print("-" * 30)

def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
    #print("========== Testing Dataset Generation =========")
    #test_dataset_generation()