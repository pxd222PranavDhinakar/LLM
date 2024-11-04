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

# Hyperparameters
HYPERPARAMETERS = {
    # Model Architecture
    'vocab_size': 14,
    'embed_size': 64,
    'num_heads': 2,
    'ff_dim': 256,
    'num_layers': 2,
    'max_length': 42,
    'dropout': 0.1,
    
    # Training Parameters
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'train_samples': 200_000,
    'test_samples': 1_000,
    'max_digit_length': 20,
    
    # Device Settings
    'use_gpu': True,
    
    # Save and Log Settings
    'model_save_path': 'trained_models',
    'log_interval': 500,  # Increased from 100 to reduce overhead
    'save_interval': 1000,
}

class AdditionDataset(Dataset):
    def __init__(self, max_length, num_samples):
        super().__init__()
        self.max_length = max_length
        self.num_samples = num_samples
        
        # Define vocabulary
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({'+': 10, '=': 11, '<PAD>': 12, '<EOS>': 13})
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Set device and sequence length once
        self.device = torch.device('cuda' if HYPERPARAMETERS['use_gpu'] and 
                                 torch.cuda.is_available() else 'cpu')
        self.max_seq_length = (max_length * 2) + 3  # Account for operators and carry
        
        # Generate data
        self.data = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Data is already on correct device

    def generate_number(self, length):
        return random.randint(10**(length-1), 10**length - 1)

    def tokenize(self, s):
        return [self.vocab[c] for c in s if c in self.vocab]

    def pad_sequence(self, seq):
        seq = seq[:self.max_seq_length]
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
                    
                    # Move tensors to device immediately
                    input_tensor = torch.tensor(input_padded, dtype=torch.long, 
                                             device=self.device)
                    target_tensor = torch.tensor(target_padded, dtype=torch.long, 
                                              device=self.device)
                    
                    data.append((input_tensor, target_tensor))
        
        random.shuffle(data)
        return data

class TrainingTracker:
    def __init__(self, model, chars_to_track):
        self.model = model
        self.chars_to_track = chars_to_track
        self.char_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.device = next(model.parameters()).device
        
        # Initialize storage
        self.embedding_history = []
        self.qkv_history = []
        self.loss_history = []
        self.global_steps = []
        
        # Pre-allocate CPU buffers
        self.embedding_buffer = torch.zeros(
            len(self.char_indices), 
            model.embedding.embed.embedding_dim,
            pin_memory=True
        )
        
        # Pre-allocate QKV buffers
        head_size = model.blocks[0].attention.heads[0].key.weight.size(0)
        embed_size = model.blocks[0].attention.heads[0].key.weight.size(1)
        self.qkv_buffers = {
            'query': torch.zeros(head_size, embed_size, pin_memory=True),
            'key': torch.zeros(head_size, embed_size, pin_memory=True),
            'value': torch.zeros(head_size, embed_size, pin_memory=True)
        }
        
        # Create events for synchronization
        self.copy_event = torch.cuda.Event()
        self.compute_event = torch.cuda.Event()

    @torch.no_grad()
    def store_state(self, step, loss):
        # Record completion of compute
        self.compute_event.record()
        
        # Copy embeddings
        self.embedding_buffer.copy_(
            self.model.embedding.embed.weight[self.char_indices].detach(),
            non_blocking=True
        )
        
        # Copy QKV weights
        first_head = self.model.blocks[0].attention.heads[0]
        self.qkv_buffers['query'].copy_(first_head.query.weight.detach(), 
                                      non_blocking=True)
        self.qkv_buffers['key'].copy_(first_head.key.weight.detach(), 
                                    non_blocking=True)
        self.qkv_buffers['value'].copy_(first_head.value.weight.detach(), 
                                      non_blocking=True)
        
        # Record completion of copy
        self.copy_event.record()
        
        # Store values (will be processed after sync)
        self.embedding_history.append(self.embedding_buffer.clone())
        self.qkv_history.append({
            'query': self.qkv_buffers['query'].clone(),
            'key': self.qkv_buffers['key'].clone(),
            'value': self.qkv_buffers['value'].clone()
        })
        self.loss_history.append(loss)
        self.global_steps.append(step)

    def synchronize(self):
        # Wait for copies to complete
        self.copy_event.synchronize()

    def save_histories(self, save_path):
        # Ensure all operations are complete
        self.synchronize()
        
        # Convert to numpy arrays efficiently
        embedding_history_np = torch.stack(self.embedding_history).cpu().numpy()
        qkv_history_np = [{
            'query': qkv['query'].cpu().numpy(),
            'key': qkv['key'].cpu().numpy(),
            'value': qkv['value'].cpu().numpy()
        } for qkv in self.qkv_history]
        
        save_dict = {
            'embedding_history': embedding_history_np,
            'qkv_history': qkv_history_np,
            'loss_history': np.array(self.loss_history),
            'global_steps': np.array(self.global_steps),
            'chars_to_track': self.chars_to_track
        }
        torch.save(save_dict, save_path)

def train_model(model, train_loader, test_loader, criterion, optimizer, params, 
                save_dir='trained_models'):
    # Setup device
    device = torch.device('cuda' if params['use_gpu'] and torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Initialize tracking
    chars_to_track = [str(i) for i in range(10)] + ['+', '=']
    tracker = TrainingTracker(model, chars_to_track)
    best_accuracy = 0
    total_steps = 0
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

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
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(dim=-1)
            non_pad_mask = targets.ne(train_loader.dataset.vocab['<PAD>'])
            correct = (predicted[non_pad_mask] == targets[non_pad_mask]).sum().item()
            correct_predictions += correct
            total_predictions += non_pad_mask.sum().item()
            
            # Store state if needed
            if total_steps % params['log_interval'] == 0:
                tracker.store_state(total_steps, loss.item())
            
            # Periodic synchronization
            if total_steps % (params['log_interval'] * 10) == 0:
                tracker.synchronize()
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
            
            # Save checkpoint
            if total_steps % params['save_interval'] == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': total_steps,
                }, os.path.join(save_dir, f'checkpoint_step_{total_steps}.pth'))
            
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
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               targets.view(-1))
                test_loss += loss.item()
                
                _, predicted = outputs.max(dim=-1)
                non_pad_mask = targets.ne(test_loader.dataset.vocab['<PAD>'])
                correct += (predicted[non_pad_mask] == 
                          targets[non_pad_mask]).sum().item()
                total += non_pad_mask.sum().item()
        
        # Calculate test metrics
        test_accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{params["num_epochs"]} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'step': total_steps
            }, save_path)
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
        
        print('-' * 60)
    
    # Save final state
    tracker.save_histories(os.path.join(save_dir, 'training_history.pth'))
    return model, tracker

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create datasets
    train_dataset = AdditionDataset(HYPERPARAMETERS['max_digit_length'], 
                                  HYPERPARAMETERS['train_samples'])
    test_dataset = AdditionDataset(HYPERPARAMETERS['max_digit_length'], 
                                 HYPERPARAMETERS['test_samples'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS['batch_size'],
                            shuffle=True, pin_memory=False)  # Already on device
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS['batch_size'],
                           pin_memory=False)  # Already on device
    
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