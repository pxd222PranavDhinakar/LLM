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

class ProcessAdditionDataset(Dataset):
    def __init__(self, max_length, num_samples, random_seed):
        self.max_length = max_length
        self.num_samples = num_samples
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, target_tensor = self.data[idx]
        return input_tensor.to(self.device), target_tensor.to(self.device)

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
                    
                    # Convert to tokens
                    input_tokens = self.tokenize(input_str)
                    target_tokens = self.tokenize(target_str)
                    
                    # Convert to tensors
                    input_tensor = torch.tensor(input_tokens, dtype=torch.long)
                    target_tensor = torch.tensor(target_tokens, dtype=torch.long)
                    
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
    
    # Create save directory if it doesn't exist
    os.makedirs(params['model_save_path'], exist_ok=True)
    
    for epoch in range(params['num_epochs']):
        # Training phase
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        start_time = time.time()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{params['num_epochs']}")
        
        for batch_idx, (inputs, targets) in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(dim=-1)
            # Count exact matches (entire process+result must be correct)
            correct_predictions += (predicted == targets).all(dim=1).sum().item()
            total_predictions += targets.size(0)
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        epoch_time = time.time() - start_time
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                test_loss += loss.item()
                
                _, predicted = outputs.max(dim=-1)
                test_correct += (predicted == targets).all(dim=1).sum().item()
                test_total += targets.size(0)
        
        test_accuracy = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{params["num_epochs"]} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_path = os.path.join(params['model_save_path'], f"{params['model_name']}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'model_config': {
                    'vocab_size': params['vocab_size'],
                    'embed_size': params['embed_size'],
                    'num_heads': params['num_heads'],
                    'ff_dim': params['ff_dim'],
                    'num_layers': params['num_layers'],
                    'max_length': params['max_length'],
                    'dropout': params['dropout']
                },
                'vocab': train_loader.dataset.vocab,
                'inv_vocab': train_loader.dataset.inv_vocab
            }, save_path)
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
        
        print('-' * 60)
    
    print(f'Training completed. Best test accuracy: {best_accuracy:.4f}')
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=HYPERPARAMETERS['batch_size'])
    
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