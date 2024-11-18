
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
from collections import defaultdict
import numpy as np
import locale
import signal
from AutoregressiveArithmeticTransformer import create_arithmetic_transformer

# Set locale to handle encoding issues
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, '')

# Signal handler for clean process termination
def signal_handler(signum, frame):
    print("\nSignal received, cleaning up...")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

"""
Model & Training Configurations
-----------------------------

MODEL SIZES:
-----------
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

Large (~11M parameters):
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

TRAINING PARAMETERS:
------------------
Common Settings:
    vocab_size: 16        # 0-9 digits plus special tokens
    max_length: 256       # Maximum sequence length
    warmup_steps: 1000    # Learning rate warmup
    max_epochs: 20        # Number of training epochs
    
Dataset Parameters:
    train_samples: 50000  # Number of training examples
    eval_samples: 1000    # Number of validation examples
    max_digit_length: 10  # Maximum length of input numbers

Optimization Parameters:
    optimizer: AdamW
    weight_decay: 0.1
    beta1: 0.9
    beta2: 0.95
    grad_clip: 1.0       # Gradient clipping value

Training Schedule:
    Small:
        - First 5 epochs: learning_rate = 1e-3
        - Next 10 epochs: learning_rate = 5e-4
        - Final epochs: learning_rate = 1e-4
    
    Medium/Large:
        - Use cosine learning rate schedule with warmup
        - Warmup over first 1000 steps
        - Minimum learning rate = initial_lr / 10

Checkpointing:
    save_every: 1000     # Save checkpoint every N steps
    keep_best: 5         # Keep top 5 models by validation loss

Hardware Recommendations:
    Small: 
        - Single GPU (4GB+ VRAM)
        - Batch size: 32-64
    Medium: 
        - Single GPU (8GB+ VRAM)
        - Batch size: 64-128
    Large: 
        - Single/Multi GPU (16GB+ VRAM)
        - Batch size: 128-256 (per GPU)
    XLarge:
        - Multi GPU (32GB+ VRAM)
        - Batch size: 256-512 (total)

Example Configurations for Different Scenarios:
-------------------------------------------
Quick Experimentation:
    embed_size: 64
    num_heads: 2
    ff_dim: 256
    num_layers: 2
    batch_size: 32
    learning_rate: 1e-3
    train_samples: 10000
    max_epochs: 5

Production Training:
    embed_size: 512
    num_heads: 8
    ff_dim: 2048
    num_layers: 8
    batch_size: 128
    learning_rate: 1e-4
    train_samples: 100000
    max_epochs: 20

Memory-Limited:
    embed_size: 256
    num_heads: 4
    ff_dim: 1024
    num_layers: 4
    batch_size: 32
    gradient_accumulation_steps: 4
    learning_rate: 3e-4
    train_samples: 50000
    max_epochs: 15
"""


class Config:
    def __init__(self):
        
        '''
        Medium (~1.8M parameters):
            embed_size: 256
            num_heads: 4
            ff_dim: 1024
            num_layers: 4
            dropout: 0.1
            batch_size: 64
            learning_rate: 3e-4
        '''
        # Model Architecture
        self.vocab_size = 17    # 0-9 plus special tokens
        self.embed_size = 256   
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_layers = 4
        self.max_length = 512   # Increased to handle longer sequences
        self.dropout = 0.1
        
        # Training Parameters
        self.batch_size = 64    # Reduced to handle longer sequences
        self.learning_rate = 3e-4
        self.max_epochs = 5 
        self.warmup_steps = 1000
        self.grad_clip = 1.0
        
        # Dataset Parameters
        self.max_digit_length = 20  # This affects sequence length
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
        print(f"Initializing dataset with {num_samples} samples...")
        self.config = config
        self.num_samples = num_samples
        self.max_length = config.max_length
        
        # Setup vocabulary
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({
            ',': 10, '>': 11, ';': 12, '|': 13,
            '+': 14, '=': 15, ':': 16
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Generate data
        random.seed(seed)
        print("Generating dataset...")
        self.data = self.generate_dataset()
        print(f"Generated {len(self.data)} samples")
        
    def estimate_sequence_length(self, len1, len2):
        """Estimate sequence length before generation"""
        # Formula: input numbers + operators + steps + result
        # Each step is: d1,d2,c>r,c format (7 chars per step)
        max_len = max(len1, len2)
        step_chars = max_len * 7  # 7 chars per step
        input_chars = len1 + len2 + 3  # numbers + '+' + ':' + '='
        separator_chars = max_len - 1  # semicolons between steps
        result_chars = max(len1, len2) + 1  # possibly one more digit
        
        return input_chars + step_chars + separator_chars + result_chars
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = sequence[:-1]  # Input is all tokens except last
        y = sequence[1:]   # Target is all tokens except first
        return x, y
    
    def generate_sequence(self, num1, num2):
        """Generate complete addition sequence with steps"""
        # First estimate sequence length
        len1, len2 = len(str(num1)), len(str(num2))
        est_length = self.estimate_sequence_length(len1, len2)
        
        # Skip if estimated length exceeds max_length
        if est_length > self.max_length:
            return None
            
        # Create core problem
        input_str = f"{num1}+{num2}:"
        
        # Generate step-by-step process
        steps = []
        n1 = str(num1).zfill(len(str(max(num1, num2))))
        n2 = str(num2).zfill(len(str(max(num1, num2))))
        carry = 0
        
        for i in range(len(n1)-1, -1, -1):
            d1, d2 = int(n1[i]), int(n2[i])
            total = d1 + d2 + carry
            digit, carry = total % 10, total // 10
            
            steps.append(f"{d1},{d2},{carry}>{digit},{carry}")
            #steps.insert(0, f"{d1},{d2},{carry}>{digit},{carry}")
        
        if carry:
            #steps.append(f"{d1},{d2},{carry}>{digit},{carry}")
            steps.append(f"0,0,{carry}>{carry},0")  # Changed from insert(0)
        
        # Combine all parts
        process = ";".join(steps)
        result = str(num1 + num2)
        complete_str = f"{input_str}{process}|={result}"
        
        # Verify final length
        if len(complete_str) > self.max_length:
            return None
            
        # Convert to token indices
        return torch.tensor([self.vocab[c] for c in complete_str], dtype=torch.long)
    
    def generate_dataset(self):
        data = []
        attempts = 0
        max_attempts = self.num_samples * 2
        
        pbar = tqdm(total=self.num_samples, desc="Generating samples")
        while len(data) < self.num_samples and attempts < max_attempts:
            # Generate numbers with gradually increasing length
            max_len = min(2 + (len(data) // 10000), self.config.max_digit_length)
            len1 = random.randint(1, max_len)
            len2 = random.randint(1, max_len)
            
            num1 = random.randint(10**(len1-1), 10**len1-1) if len1 > 1 else random.randint(0, 9)
            num2 = random.randint(10**(len2-1), 10**len2-1) if len2 > 1 else random.randint(0, 9)
            
            sequence = self.generate_sequence(num1, num2)
            if sequence is not None:
                data.append(sequence)
                pbar.update(1)
            attempts += 1
            
        pbar.close()
        if len(data) < self.num_samples:
            print(f"Warning: Only generated {len(data)} valid samples out of {self.num_samples} requested")
            
        # Print some statistics
        lengths = [len(seq) for seq in data]
        print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        return data

def collate_fn(batch):
    """Collate and pad sequences"""
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
        
        # Create datasets
        self.train_dataset = ProcessAdditionDataset(
            config, config.train_samples, config.train_seed
        )
        self.val_dataset = ProcessAdditionDataset(
            config, config.val_samples, config.val_seed
        )
        
        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Start with 0 to debug
            pin_memory=False,  # Disable for now
            persistent_workers=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Start with 0 to debug
            pin_memory=False,  # Disable for now
            persistent_workers=False
        )
        
        self.global_step = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            logits, loss = self.model(inputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, Acc: {total_correct/total_tokens:.4f}"
            )
            
            self.global_step += 1
            
            # Save checkpoint
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
    
    def save_checkpoint(self, is_best=False, metrics=None):
        """Save a checkpoint of the model
        
        Args:
            is_best (bool): If True, this is our best model so far
            metrics (dict): Dictionary of metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics or {},
            'vocab': self.train_dataset.vocab,
            'inv_vocab': self.train_dataset.inv_vocab
        }
        
        # Create checkpoint directory
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.save_dir,
            f"{self.config.model_name}_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # If this is our best model, save a copy
        if is_best:
            best_path = os.path.join(
                self.config.save_dir,
                f"{self.config.model_name}_best.pt"
            )
            torch.save(checkpoint, best_path)
            print(f"\nSaved best model to {best_path}")
            print(f"Metrics: {metrics}")
        
        
    @classmethod
    def load_checkpoint(cls, path, device=None):
        """Load a checkpoint and create a trainer instance
        
        Args:
            path (str): Path to checkpoint file
            device (torch.device): Device to load model to
        """
        print(f"Loading checkpoint from {path}")
        
        # Load checkpoint to CPU first
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create config and trainer
        config = checkpoint['config']
        if device is not None:
            config.device = device
            
        trainer = cls(config)
        
        # Load state
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.current_epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from step {trainer.global_step}")
        if 'metrics' in checkpoint:
            print(f"Metrics: {checkpoint['metrics']}")
        
        return trainer
    
    def save_final_model(self):
        """Save the final trained model in trained_models directory"""
        final_model_dir = "trained_models"
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Save final model with training info
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab': self.train_dataset.vocab,
            'inv_vocab': self.train_dataset.inv_vocab,
            'final_metrics': {
                'total_steps': self.global_step,
                'total_epochs': self.current_epoch + 1
            }
        }
        
        # Include timestamp in filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_name}_final_{timestamp}.pt"
        final_path = os.path.join(final_model_dir, model_name)
        
        # Save the model
        torch.save(final_checkpoint, final_path)
        print(f"\nSaved final model to {final_path}")
        
        # Save a config summary
        config_summary = {
            'model_name': self.config.model_name,
            'embed_size': self.config.embed_size,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'vocab_size': self.config.vocab_size,
            'max_length': self.config.max_length,
            'timestamp': timestamp
        }
        
        import json
        config_path = os.path.join(final_model_dir, f"model_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config_summary, f, indent=4)
    
    def train(self):
        best_val_loss = float('inf')
        self.current_epoch = 0
        
        print("Starting training...")
        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch()
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                
                # Validate
                val_loss, val_acc = self.validate()
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Regular checkpointing
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(metrics=metrics)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(is_best=True, metrics=metrics)
            
            # Save final model after training is complete
            self.save_final_model()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(metrics=metrics)
            # Also save final model if interrupted
            self.save_final_model()
            


def load_final_model(model_path, device=None):
    """Load a final trained model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_arithmetic_transformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['vocab'], checkpoint['inv_vocab']


def test_dataset_generation():
    config = Config()
    
    # Create small test dataset
    dataset = ProcessAdditionDataset(config, num_samples=100, seed=42)
    
    print("Sample training sequences:")
    print("-" * 50)
    
    for i, seq in enumerate(dataset.data[:100]):
        # Convert tokens back to string
        sequence = ''.join([dataset.inv_vocab[token.item()] for token in seq])
        
        # Split into parts for readability
        problem, steps = sequence.split(':')
        steps, result = steps.split('|')
        
        print(f"\nExample {i+1}:")
        print(f"Problem: {problem}")
        print(f"Steps: {steps}")
        print(f"Result: {result}")

def main():
    try:
        print("Creating config...")
        config = Config()
        print(f"Using device: {config.device}")
        
        print("Initializing trainer...")
        trainer = Trainer(config)
        
        print("Starting training...")
        trainer.train()
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()
    #test_dataset_generation()