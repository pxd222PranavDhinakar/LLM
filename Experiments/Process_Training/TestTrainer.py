import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
from AutoregressiveArithmeticTransformer import create_arithmetic_transformer

class Config:
    def __init__(self):
        # Model Architecture
        self.vocab_size = 33  # Updated to include END token
        self.embed_size = 256
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_layers = 4
        self.max_length = 512
        self.dropout = 0.1
        
        # Training Parameters
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.max_epochs = 20
        self.warmup_steps = 2000
        self.grad_clip = 1.0
        
        # Dataset Parameters
        self.train_samples = 250000
        self.val_samples = 2500
        self.train_seed = 42
        self.val_seed = 43
        self.curriculum_epochs = [0, 5, 10]  # When to increase difficulty
        
        # Save Parameters
        self.save_dir = "checkpoints"
        self.model_name = "arithmetic_transformer"
        self.save_every = 5000
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessAdditionDataset(Dataset):
    def __init__(self, config, num_samples, seed, curriculum_step=0):
        self.config = config
        self.num_samples = num_samples
        self.max_length = config.max_length
        self.curriculum_step = curriculum_step
        
        # Setup vocabulary with new tokens
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({
            '+': 12, '>': 13, 
            'A': 14, 'C': 15,
            '-': 16, ',': 17,
            ' ': 18, '\n': 19,
            ':': 20,
            'I': 21, 'n': 22, 'p': 23, 'u': 24, 't': 25,
            'T': 26, 'a': 27, 'r': 28, 'g': 29, 'e': 30,
            '.': 31,
            '<END>': 32  # Added END token
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = 0
        
        random.seed(seed)
        self.data = self.generate_dataset()
    
    def get_number_range(self):
        if self.curriculum_step == 0:
            return 0, 4  # Start with small numbers
        elif self.curriculum_step == 1:
            return 0, 7  # Medium difficulty
        else:
            return 0, 9  # Full range
    
    def generate_sequence(self, num1, num2):
        # Format validation
        if not (0 <= num1 <= 9 and 0 <= num2 <= 9):
            return None
            
        # Generate sequence with consistent format
        input_str = f"Input: {num1}+{num2}\n"
        target_str = "Target:\n"
        
        # Calculate addition
        total = num1 + num2
        digit, carry = total % 10, total // 10
        
        # Format carry step and result
        step_str = f"A->{digit}, C->{carry}.\n"
        result_str = f"{total}<END>"
        
        # Combine all parts
        complete_str = input_str + target_str + step_str + result_str
        
        # Convert to tokens
        tokens = []
        for c in complete_str:
            if c in self.vocab:
                tokens.append(self.vocab[c])
            else:
                print(f"Warning: Character '{c}' not in vocabulary!")
                return None
                
        return torch.tensor(tokens, dtype=torch.long)

    def generate_dataset(self):
        data = []
        pbar = tqdm(total=self.num_samples, desc="Generating samples")
        
        min_num, max_num = self.get_number_range()
        
        while len(data) < self.num_samples:
            num1 = random.randint(min_num, max_num)
            num2 = random.randint(min_num, max_num)
            
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

def create_padding_mask(tensor, pad_token=0):
    """Create mask for padding tokens"""
    return (tensor != pad_token).float()

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
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        self.current_curriculum = 0
        self.train_dataset = None
        self.val_dataset = None
        self.setup_datasets()
        
        self.global_step = 0
        self.best_val_acc = 0.0
        
    def setup_datasets(self):
        """Initialize or update datasets based on curriculum step"""
        self.train_dataset = ProcessAdditionDataset(
            self.config, 
            self.config.train_samples, 
            self.config.train_seed,
            self.current_curriculum
        )
        
        self.val_dataset = ProcessAdditionDataset(
            self.config, 
            self.config.val_samples, 
            self.config.val_seed,
            self.current_curriculum
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        sequence_correct = 0
        total_sequences = 0
        
        progress_bar = tqdm(self.train_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Create padding mask
            padding_mask = create_padding_mask(targets)
            padding_mask = padding_mask.to(self.config.device)
            
            # Forward pass
            logits, loss = self.model(inputs, targets)
            
            # Apply padding mask to loss
            loss = loss * padding_mask.view(-1)
            loss = loss.sum() / padding_mask.sum()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Calculate metrics
            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == targets) * padding_mask).sum().item()
            total_tokens += padding_mask.sum().item()
            
            # Sequence-level accuracy
            sequence_match = ((predictions == targets) * padding_mask).all(dim=1)
            sequence_correct += sequence_match.sum().item()
            total_sequences += inputs.size(0)
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, "
                f"Token Acc: {total_correct/total_tokens:.4f}, "
                f"Seq Acc: {sequence_correct/total_sequences:.4f}"
            )
            
            self.global_step += 1
            
            # Log sample predictions periodically
            if self.global_step % 1000 == 0:
                self.log_sample_predictions(inputs[0], targets[0], predictions[0])
            
            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
        
        return (total_loss / len(self.train_loader), 
                total_correct / total_tokens,
                sequence_correct / total_sequences)
    
    def log_sample_predictions(self, input_seq, target_seq, pred_seq):
        """Log a sample prediction for debugging"""
        print("\nSample Prediction:")
        print("-" * 50)
        pred_str = ''.join([self.train_dataset.inv_vocab[t.item()] 
                          for t in pred_seq])
        target_str = ''.join([self.train_dataset.inv_vocab[t.item()] 
                            for t in target_seq])
        print(f"Prediction: {pred_str}")
        print(f"Target: {target_str}")
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        sequence_correct = 0
        total_sequences = 0
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            padding_mask = create_padding_mask(targets).to(self.config.device)
            
            logits, loss = self.model(inputs, targets)
            loss = (loss * padding_mask.view(-1)).sum() / padding_mask.sum()
            
            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == targets) * padding_mask).sum().item()
            total_tokens += padding_mask.sum().item()
            
            sequence_match = ((predictions == targets) * padding_mask).all(dim=1)
            sequence_correct += sequence_match.sum().item()
            total_sequences += inputs.size(0)
            
        return (total_loss / len(self.val_loader),
                total_correct / total_tokens,
                sequence_correct / total_sequences)

    def save_checkpoint(self, is_best=False, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'vocab': self.train_dataset.vocab,
            'inv_vocab': self.train_dataset.inv_vocab,
            'curriculum_step': self.current_curriculum
        }
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save regular checkpoint
        path = os.path.join(
            self.config.save_dir,
            f"{self.config.model_name}_step_{self.global_step}.pt"
        )
        torch.save(checkpoint, path)
        
        # Save best model
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
                
                # Update curriculum if needed
                if epoch in self.config.curriculum_epochs:
                    self.current_curriculum += 1
                    print(f"Updating curriculum to step {self.current_curriculum}")
                    self.setup_datasets()
                
                # Train epoch
                train_loss, train_token_acc, train_seq_acc = self.train_epoch()
                print(f"Train - Loss: {train_loss:.4f}, "
                      f"Token Acc: {train_token_acc:.4f}, "
                      f"Sequence Acc: {train_seq_acc:.4f}")
                
                # Validate
                val_loss, val_token_acc, val_seq_acc = self.validate()
                print(f"Val - Loss: {val_loss:.4f}, "
                      f"Token Acc: {val_token_acc:.4f}, "
                      f"Sequence Acc: {val_seq_acc:.4f}")
                
                # Save if best model
                if val_seq_acc > self.best_val_acc:
                    self.best_val_acc = val_seq_acc
                    print(f"New best validation accuracy: {val_seq_acc:.4f}")
                    self.save_checkpoint(is_best=True)
            
            # Save final model
            self.save_checkpoint(is_final=True)
            print("Training completed!")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint()
            
            
            

def test_model_inference(trainer):
    """Test the trained model on a set of examples"""
    print("\n" + "="*50)
    print("Running inference tests...")
    print("="*50)
    
    model = trainer.model
    vocab = trainer.train_dataset.vocab
    inv_vocab = trainer.train_dataset.inv_vocab
    device = trainer.config.device
    
    # Test cases covering different scenarios
    test_cases = [
        (2, 3),  # Simple addition
        (5, 4),  # Another simple case
        (9, 8),  # Addition with carry
        (0, 7),  # Addition with zero
        (9, 9),  # Maximum single digit addition
    ]
    
    model.eval()
    with torch.no_grad():
        for num1, num2 in test_cases:
            print(f"\nTesting: {num1} + {num2}")
            print("-" * 30)
            
            # Create input sequence
            input_str = f"Input: {num1}+{num2}\nTarget:\n"
            tokens = []
            for c in input_str:
                if c in vocab:
                    tokens.append(vocab[c])
            
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            generated = x.clone()
            
            # Generate output token by token
            for _ in range(50):  # Maximum sequence length
                if generated.size(1) > trainer.config.max_length:
                    generated = generated[:, -trainer.config.max_length:]
                
                logits = model(generated)
                next_token_logits = logits[:, -1, :] / 0.6  # Temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Break if we generate END token
                if next_token.item() == vocab['<END>']:
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                    
                generated = torch.cat([generated, next_token], dim=1)
            
            # Convert output tokens to string
            output = ''.join([inv_vocab.get(t.item(), '?') for t in generated[0]])
            
            # Print formatted output
            print("Model output:")
            print(output)
            
            # Verify result
            try:
                lines = output.split('\n')
                # Find carry steps
                carry_steps = [l for l in lines if l.startswith('A->')]
                # Find final result
                result_line = next((l for l in reversed(lines) 
                                  if l.strip() and l[0].isdigit()), None)
                
                if result_line:
                    predicted = int(result_line.strip().replace('<END>', ''))
                    actual = num1 + num2
                    print(f"\nCarry steps found:")
                    for step in carry_steps:
                        print(step)
                    print(f"Predicted result: {predicted}")
                    print(f"Actual result: {actual}")
                    print(f"Correct: {predicted == actual}")
            except Exception as e:
                print(f"Error parsing result: {str(e)}")

def main():
    config = Config()
    trainer = Trainer(config)
    
    # Train the model
    trainer.train()
    
    # Run inference tests
    test_model_inference(trainer)

if __name__ == "__main__":
    main()
    
    