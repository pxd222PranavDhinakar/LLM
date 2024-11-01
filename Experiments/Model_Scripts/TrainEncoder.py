import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
import random
import os
from ArithmeticEncoder import create_arithmetic_transformer

# Hyperparameters - Edit these as needed
HYPERPARAMETERS = {
    # Model Architecture
    'vocab_size': 14,          # Size of vocabulary (digits + special tokens)
    'embed_size': 64,          # Dimension of embeddings
    'num_heads': 2,            # Number of attention heads
    'ff_dim': 256,            # Feed-forward dimension (4x embed_size)
    'num_layers': 2,          # Number of transformer layers
    'max_length': 42,         # Maximum sequence length
    'dropout': 0.1,           # Dropout rate
    
    # Training Parameters
    'batch_size': 32,         # Batch size
    'num_epochs': 10,         # Number of epochs
    'learning_rate': 0.001,   # Learning rate
    'train_samples': 200_000, # Number of training samples
    'test_samples': 1_000,    # Number of test samples
    'max_digit_length': 20,   # Maximum length of operands
    
    # Device Settings
    'use_gpu': True,  # Set to False to force CPU usage
    
    # Save and Log Settings
    'model_save_path': 'trained_models',  # Directory to save models
    'log_interval': 100,      # Steps between logging
    'save_interval': 1000,    # Steps between saving model checkpoints
}

class AdditionDataset(Dataset):
    def __init__(self, max_length, num_samples):
        self.max_length = max_length
        self.num_samples = num_samples
        # Define vocabulary: 0-9 for digits, 10 for '+', 11 for '=', 12 for padding, 13 for EOS
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({'+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Modify the return to specify device
        input_tensor, target_tensor = self.data[idx]
        device = torch.device('cuda' if HYPERPARAMETERS['use_gpu'] and torch.cuda.is_available() else 'cpu')
        return input_tensor.to(device), target_tensor.to(device)

    def generate_number(self, length):
        return random.randint(10**(length-1), 10**length - 1)

    def tokenize(self, s):
        return [self.vocab[c] for c in s if c in self.vocab]

    def pad_sequence(self, seq, max_length):
        return seq + [self.vocab['<PAD>']] * (max_length - len(seq))

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
                    input_str = input_str[::-1]  # Reverse string
                    target_str = f"{result}"[::-1]
                    
                    input_tokens = self.tokenize(input_str)
                    target_tokens = self.tokenize(target_str) + [self.vocab['<EOS>']]
                    
                    max_seq_length = self.max_length * 2 + 2
                    input_padded = self.pad_sequence(input_tokens, max_seq_length)
                    target_padded = self.pad_sequence(target_tokens, max_seq_length)
                    
                    input_tensor = torch.tensor(input_padded, dtype=torch.long)
                    target_tensor = torch.tensor(target_padded, dtype=torch.long)
                    
                    data.append((input_tensor, target_tensor))
        
        random.shuffle(data)
        return data

class TrainingTracker:
    def __init__(self, model, chars_to_track):
        self.embedding_history = []
        self.qkv_history = []
        self.loss_history = []
        self.global_steps = []
        self.chars_to_track = chars_to_track
        self.char_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Indices for digits + operators
        self.model = model
        self.device = torch.device('cuda' if HYPERPARAMETERS['use_gpu'] and torch.cuda.is_available() else 'cpu')

        
    def store_state(self, step, loss):
        # Ensure we detach and move to CPU before converting to numpy
        current_embeddings = self.model.embedding.embed.weight[self.char_indices].detach().cpu().numpy()
        self.embedding_history.append(current_embeddings)
        
        # Store Q, K, V weights from first attention head of first layer
        first_block = self.model.blocks[0]
        first_head = first_block.attention.heads[0]
        qkv_weights = {
            'query': first_head.query.weight.detach().cpu().numpy(),
            'key': first_head.key.weight.detach().cpu().numpy(),
            'value': first_head.value.weight.detach().cpu().numpy()
        }
        self.qkv_history.append(qkv_weights)
        
        self.loss_history.append(loss)
        self.global_steps.append(step)
    
    def save_histories(self, save_path):
        save_dict = {
            'embedding_history': np.array(self.embedding_history),
            'qkv_history': self.qkv_history,
            'loss_history': np.array(self.loss_history),
            'global_steps': np.array(self.global_steps),
            'chars_to_track': self.chars_to_track
        }
        torch.save(save_dict, save_path)

def train_model(model, train_loader, test_loader, criterion, optimizer, params, save_dir='trained_models'):
    # Set device
    device = torch.device('cuda' if params['use_gpu'] and torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Move model and criterion to device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize tracking
    chars_to_track = [str(i) for i in range(10)] + ['+', '=']
    tracker = TrainingTracker(model, chars_to_track)
    best_accuracy = 0
    total_steps = 0
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(dim=-1)
            non_pad_mask = targets.ne(train_loader.dataset.vocab['<PAD>'])
            correct_predictions += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
            total_predictions += non_pad_mask.sum().item()
            
            # Store embeddings and loss
            if total_steps % params['log_interval'] == 0:
                tracker.store_state(total_steps, loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
            
            # Save checkpoint
            if total_steps % params['save_interval'] == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{total_steps}.pth')
                save_checkpoint(model, optimizer, total_steps, checkpoint_path)
            
            total_steps += 1
        
        # Calculate epoch statistics
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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Calculate test loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                test_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(dim=-1)
                non_pad_mask = targets.ne(test_loader.dataset.vocab['<PAD>'])
                correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
                total += non_pad_mask.sum().item()
        
        test_accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f'\nEpoch {epoch+1}/{params["num_epochs"]} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(model, optimizer, total_steps, 
                          os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
        
        print('-' * 60)
    
    # Save final model and history
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    history_path = os.path.join(save_dir, 'training_history.pth')
    save_checkpoint(model, optimizer, total_steps, final_model_path)
    tracker.save_histories(history_path)
    
    return model, tracker

def save_checkpoint(model, optimizer, step, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'model_config': {
            'vocab_size': model.embedding.embed.num_embeddings,
            'embed_size': model.embedding.embed.embedding_dim,
            'num_heads': len(model.blocks[0].attention.heads),
            'ff_dim': model.blocks[0].feed_forward.net[0].out_features,
            'num_layers': len(model.blocks),
            'max_length': model.embedding.max_length,
            'dropout': model.blocks[0].attention.dropout.p,
        }
    }, path)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # No need to move to device here as it's handled in the Dataset class
            outputs = model(inputs)
            _, predicted = outputs.max(dim=-1)
            non_pad_mask = targets.ne(dataloader.dataset.vocab['<PAD>'])
            correct += (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
            total += non_pad_mask.sum().item()
    
    return correct / total

def main():
    # Set device
    device = torch.device('cuda' if HYPERPARAMETERS['use_gpu'] and torch.cuda.is_available() else 'cpu')
    
    # Print device information at start of training
    print(f"\nInitializing training with device: {device}")
    if device.type == 'cuda':
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
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
    model, tracker = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        HYPERPARAMETERS
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()