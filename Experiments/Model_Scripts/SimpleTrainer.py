# SimpleTrainer.py
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
# Current Model (~105K parameters)
SMALL_MODEL = {
    'vocab_size': 14,
    'embed_size': 64,
    'num_heads': 2,
    'ff_dim': 256,
    'num_layers': 2,
    'max_length': 42,
    'dropout': 0.1,
}

# Medium Model (~1.8M parameters)
MEDIUM_MODEL = {
    'vocab_size': 14,
    'embed_size': 256,
    'num_heads': 4,
    'ff_dim': 1024,
    'num_layers': 4,
    'max_length': 42,
    'dropout': 0.1,
}

# Large Model (~11M parameters)
LARGE_MODEL = {
    'vocab_size': 14,
    'embed_size': 512,
    'num_heads': 8,
    'ff_dim': 2048,
    'num_layers': 8,
    'max_length': 42,
    'dropout': 0.15,
}

# XLarge Model (~45M parameters)
XLARGE_MODEL = {
    'vocab_size': 14,
    'embed_size': 1024,
    'num_heads': 16,
    'ff_dim': 4096,
    'num_layers': 12,
    'dropout': 0.2,
    'max_length': 42,
}

# Training parameters should also be adjusted per model size
TRAINING_CONFIGS = {
    'SMALL': {
        'batch_size': 32,
        'learning_rate': 1e-3,
    },
    'MEDIUM': {
        'batch_size': 64,
        'learning_rate': 5e-4,
    },
    'LARGE': {
        'batch_size': 128,
        'learning_rate': 3e-4,
    },
    'XLARGE': {
        'batch_size': 256,
        'learning_rate': 1e-4,
    }
}
'''

# Hyperparameters
HYPERPARAMETERS = {
    # Model Architecture
    'vocab_size': 14,
    'embed_size': 256,
    'num_heads': 4,
    'ff_dim': 1024,
    'num_layers': 4,
    'max_length': 42,
    'dropout': 0.1,
    
    # Training Parameters
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'train_samples': 200_000,
    'test_samples': 1_000,
    'max_digit_length': 20,
    
    # Device
    'use_gpu': True,
    
    # Save Settings
    'model_save_path': 'trained_models'
}

class AdditionDataset(Dataset):
    def __init__(self, max_length, num_samples):
        self.max_length = max_length
        self.num_samples = num_samples
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({'+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.device = torch.device('cuda' if HYPERPARAMETERS['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.max_seq_length = (max_length * 2) + 3  # 2 numbers + '+' + '=' + possible carry
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, target_tensor = self.data[idx]
        return input_tensor.to(self.device), target_tensor.to(self.device)

    def generate_number(self, length):
        return random.randint(10**(length-1), 10**length - 1)

    def tokenize(self, s):
        return [self.vocab[c] for c in s if c in self.vocab]

    def pad_sequence(self, seq):
        seq = seq[:self.max_seq_length]  # Truncate if too long
        return seq + [self.vocab['<PAD>']] * (self.max_seq_length - len(seq))

    def generate_data(self):
        data = []
        samples_per_combination = max(1, self.num_samples // (self.max_length ** 2))
        
        for i in range(1, self.max_length + 1):
            for j in range(1, self.max_length + 1):
                for _ in range(samples_per_combination):
                    num1 = self.generate_number(i)
                    num2 = self.generate_number(j)
                    result = num1 + num2
                    
                    input_str = f"{num1:0{i}}+{num2:0{j}}="
                    input_str = input_str[::-1]
                    target_str = f"{result}"[::-1]
                    
                    input_tokens = self.tokenize(input_str)
                    target_tokens = self.tokenize(target_str) + [self.vocab['<EOS>']]
                    
                    input_padded = self.pad_sequence(input_tokens)
                    target_padded = self.pad_sequence(target_tokens)
                    
                    input_tensor = torch.tensor(input_padded, dtype=torch.long)
                    target_tensor = torch.tensor(target_padded, dtype=torch.long)
                    
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
            non_pad_mask = targets.ne(train_loader.dataset.vocab['<PAD>'])
            correct_predictions += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
            total_predictions += non_pad_mask.sum().item()
            
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                test_loss += loss.item()
                
                _, predicted = outputs.max(dim=-1)
                non_pad_mask = targets.ne(test_loader.dataset.vocab['<PAD>'])
                correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                total += non_pad_mask.sum().item()
        
        test_accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{params["num_epochs"]} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, os.path.join(params['model_save_path'], 'best_model.pth'))
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
        
        print('-' * 60)
    
    print(f'Training completed. Best test accuracy: {best_accuracy:.4f}')
    return model

def main():
    # Create datasets
    train_dataset = AdditionDataset(HYPERPARAMETERS['max_digit_length'],
                                  HYPERPARAMETERS['train_samples'])
    test_dataset = AdditionDataset(HYPERPARAMETERS['max_digit_length'],
                                 HYPERPARAMETERS['test_samples'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS['batch_size'])
    
    # Create model
    model = create_arithmetic_transformer(
        vocab_size=HYPERPARAMETERS['vocab_size'],
        embed_size=HYPERPARAMETERS['embed_size'],
        num_heads=HYPERPARAMETERS['num_heads'],
        ff_dim=HYPERPARAMETERS['ff_dim'],
        num_layers=HYPERPARAMETERS['num_layers'],
        max_length=HYPERPARAMETERS['max_length'],
        dropout=HYPERPARAMETERS['dropout']
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=HYPERPARAMETERS['vocab_size']-2)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
    
    # Train model
    model = train_model(model, train_loader, test_loader, criterion, optimizer, HYPERPARAMETERS)
    print("Training completed!")

if __name__ == "__main__":
    main()