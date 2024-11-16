import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import random
import os
from ArithmeticTransformer import create_arithmetic_transformer

'''
# Small (~105K parameters)
- embed_size: 64
- num_heads: 2
- ff_dim: 256
- num_layers: 2

# Medium (~1.8M parameters)
- embed_size: 256
- num_heads: 4
- ff_dim: 1024
- num_layers: 4

# Large (~11M parameters)
- embed_size: 512
- num_heads: 8
- ff_dim: 2048
- num_layers: 8

# XLarge (~45M parameters)
- embed_size: 1024
- num_heads: 16
- ff_dim: 4096
- num_layers: 12
'''

# Hyperparameters
HYPERPARAMETERS = {
    # Model Architecture
    'vocab_size': 16,  # 0-9 plus ',', '>', ';', '|', '+', '='
    'embed_size': 64,
    'num_heads': 2,
    'ff_dim': 256,
    'num_layers': 2,
    'max_length': 128,  # Increased for process steps
    'dropout': 0.1,
    
    # Training Parameters
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 1e-3,
    'train_samples': 200_000,
    'test_samples': 1_000,
    'max_digit_length': 20,
    'train_seed': 42,
    'test_seed': 43,
    
    # Device
    'use_gpu': True,
    
    # Save Settings
    'model_save_path': 'trained_models',
    'model_name': 'process_addition_model'
}

def collate_batch(batch):
    # Sort by input length for potential future optimizations
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    # Get lengths
    input_lengths = [len(x) for x in inputs]
    target_lengths = [len(x) for x in targets]
    
    # Find max lengths
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    # Pad sequences
    padded_inputs = torch.full((len(batch), max_input_len), 0, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_target_len), 0, dtype=torch.long)
    
    # Fill in the actual sequences
    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(input_seq)] = input_seq
        padded_targets[i, :len(target_seq)] = target_seq
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return padded_inputs.to(device), padded_targets.to(device)

class ProcessAdditionDataset(Dataset):
    def __init__(self, max_length, num_samples, random_seed):
        self.max_length = max_length
        self.num_samples = num_samples
        self.max_input_length = max_length * 2 + 1  # num1 + '+' + num2
        self.max_target_length = max_length * 10  # Generous estimate for process+result
        # Define vocabulary
        self.vocab = {str(i): i for i in range(10)}  # 0-9
        self.vocab.update({
            ',': 10,  # step element separator
            '>': 11,  # operation result
            ';': 12,  # step separator
            '|': 13,  # process-result separator
            '+': 14,  # addition operator
            '=': 15   # equals sign
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        random.seed(random_seed)
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def pad_sequence(self, seq, max_len):
        return torch.nn.functional.pad(
            seq, 
            (0, max_len - len(seq)), 
            mode='constant', 
            value=0
        )

    def generate_number(self, length):
        return random.randint(10**(length-1), 10**length - 1)

    def generate_addition_steps(self, num1, num2):
        steps = []
        carry = 0
        num1_digits = list(map(int, str(num1)))[::-1]
        num2_digits = list(map(int, str(num2)))[::-1]
        max_len = max(len(num1_digits), len(num2_digits))
        
        for pos in range(max_len):
            d1 = num1_digits[pos] if pos < len(num1_digits) else 0
            d2 = num2_digits[pos] if pos < len(num2_digits) else 0
            
            current_sum = d1 + d2 + carry
            result_digit = current_sum % 10
            next_carry = current_sum // 10
            
            steps.append(f"{d1},{d2},{carry}>{result_digit},{next_carry}")
            carry = next_carry
            
        if carry > 0:
            steps.append(f"0,0,{carry}>{carry},0")
            
        return steps

    def tokenize(self, s):
        return [self.vocab[c] for c in s]

    def generate_data(self):
        data = []
        samples_per_combination = max(1, self.num_samples // (self.max_length ** 2))
        
        for i in range(1, self.max_length + 1):
            for j in range(1, self.max_length + 1):
                for _ in range(samples_per_combination):
                    num1 = self.generate_number(i)
                    num2 = self.generate_number(j)
                    result = num1 + num2
                    
                    # Generate input: "num1+num2"
                    input_str = f"{num1}+{num2}"
                    
                    # Generate process steps
                    steps = self.generate_addition_steps(num1, num2)
                    process_str = ";".join(steps)
                    
                    # Combine process and result
                    target_str = f"{process_str}|={result}"
                    
                    # Convert to tokens and pad
                    input_tokens = self.tokenize(input_str)
                    target_tokens = self.tokenize(target_str)
                    
                    # Pad sequences to fixed lengths
                    input_tensor = torch.tensor(input_tokens, dtype=torch.long)
                    target_tensor = torch.tensor(target_tokens, dtype=torch.long)
                    
                    input_tensor = self.pad_sequence(input_tensor, self.max_input_length)
                    target_tensor = self.pad_sequence(target_tensor, self.max_target_length)
                    
                    data.append((input_tensor, target_tensor))
        
        random.shuffle(data)
        return data

def train_model(model, train_loader, test_loader, criterion, optimizer, params):
    device = torch.device('cuda' if params['use_gpu'] and torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = model.to(device)
    criterion = criterion.to(device)
    best_accuracy = 0
    
    # Debug first batch
    print("\nDEBUG: Examining first batch from train_loader...")
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    print(f"Sample batch input shape: {inputs.shape}")
    print(f"Sample batch target shape: {targets.shape}")
    
    for epoch in range(params['num_epochs']):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{params['num_epochs']}")
        
        for batch_idx, (inputs, targets) in progress_bar:
            try:
                print("\nDEBUG: Batch Information")
                print(f"Batch idx: {batch_idx}")
                print(f"Input shape: {inputs.shape}")
                print(f"Target shape: {targets.shape}")
                
                optimizer.zero_grad()
                outputs = model(inputs)
                print(f"Model output shape: {outputs.shape}")
                
                # Get shapes
                B, S, V = outputs.shape
                print(f"Batch size (B): {B}")
                print(f"Sequence length (S): {S}")
                print(f"Vocab size (V): {V}")
                
                # Reshape outputs and targets for loss calculation
                outputs_flat = outputs.view(-1, V)
                targets_flat = targets.reshape(-1)
                print(f"Flattened outputs shape: {outputs_flat.shape}")
                print(f"Flattened targets shape: {targets_flat.shape}")
                
                # Debug unique values
                print(f"Unique values in targets: {torch.unique(targets).tolist()}")
                print(f"Target min/max: {targets.min().item()}, {targets.max().item()}")
                
                # Calculate loss (ignoring padding)
                loss = criterion(outputs_flat, targets_flat)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(dim=-1)
                correct_predictions += (predicted == targets).all(dim=1).sum().item()
                total_predictions += B
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct_predictions/total_predictions:.4f}"
                })
                
                # Break after first batch for debugging
                if batch_idx == 0:
                    print("\nDEBUG: Completed first batch successfully")
                    print("-" * 60)
                    break
                    
            except Exception as e:
                print("\nERROR DETAILS:")
                print(f"Error occurred in batch {batch_idx}")
                print(f"Input tensor shape: {inputs.shape}")
                print(f"Target tensor shape: {targets.shape}")
                if 'outputs' in locals():
                    print(f"Output tensor shape: {outputs.shape}")
                if 'outputs_flat' in locals():
                    print(f"Flattened output shape: {outputs_flat.shape}")
                if 'targets_flat' in locals():
                    print(f"Flattened target shape: {targets_flat.shape}")
                raise e
                
        print("\nEpoch Summary:")
        print(f"Total batches processed: {batch_idx + 1}")
        print(f"Final input shape: {inputs.shape}")
        print(f"Final target shape: {targets.shape}")
        print(f"Final output shape: {outputs.shape}")
        break  # Stop after first epoch for debugging
    
    return model

def main():
    # Create datasets with different seeds
    train_dataset = ProcessAdditionDataset(
        HYPERPARAMETERS['max_digit_length'],
        HYPERPARAMETERS['train_samples'],
        HYPERPARAMETERS['train_seed']
    )
    
    test_dataset = ProcessAdditionDataset(
        HYPERPARAMETERS['max_digit_length'],
        HYPERPARAMETERS['test_samples'],
        HYPERPARAMETERS['test_seed']
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=HYPERPARAMETERS['batch_size'],
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=HYPERPARAMETERS['batch_size'],
        collate_fn=collate_batch
    )
    
    # Create model
    model = create_arithmetic_transformer(
        vocab_size=HYPERPARAMETERS['vocab_size'],
        embed_size=HYPERPARAMETERS['embed_size'],
        num_heads=HYPERPARAMETERS['num_heads'],
        head_size=HYPERPARAMETERS['embed_size'] // HYPERPARAMETERS['num_heads'],
        ff_dim=HYPERPARAMETERS['ff_dim'],
        num_layers=HYPERPARAMETERS['num_layers'],
        max_length=HYPERPARAMETERS['max_length'],
        dropout=HYPERPARAMETERS['dropout']
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
    
    # Train model
    model = train_model(model, train_loader, test_loader, criterion, optimizer, HYPERPARAMETERS)
    print("Training completed!")

if __name__ == "__main__":
    main()