# TestTrainer.py: Contains the Trainer class for training the model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np
from NewArchitecture import create_arithmetic_transformer

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

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
        self.vocab_size = 33  # Includes END token
        self.embed_size = 256
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_layers = 4
        self.max_length = 512
        self.dropout = 0.1
        
        # Training Parameters (updated)
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.max_epochs = 10  # Increased from 1 to 10
        self.warmup_steps = 2000
        self.grad_clip = 1.0
        
        # Dataset Parameters (updated)
        self.train_samples = 10000  # Decreased from 250000
        self.val_samples = 1000     # Decreased from 2500
        self.train_seed = 42
        self.val_seed = 43
        
        self.curriculum_epochs = [0, 2, 4]  # When to increase difficulty
        self.curriculum_samples = [5000, 7500, 10000]  # Samples per curriculum stage
        self.curriculum_digit_ranges = [(0, 4), (0, 7), (0, 9)]  # Number ranges       
         
        # Save Parameters
        self.save_dir = "checkpoints"
        self.model_name = "arithmetic_transformer"
        self.save_every = 1000  # Decreased from 5000
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessAdditionDataset(Dataset):
    def __init__(self, config, num_samples, seed, curriculum_step=0):
        self.config = config
        self.num_samples = num_samples
        self.max_length = config.max_length
        self.curriculum_step = curriculum_step
        
        # Setup vocabulary (unchanged)
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
            '$': 32  # END token
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = 0
        
        # Set random seed and generate dataset
        random.seed(seed)
        self.data = []
        self.generate_dataset()
    
    def get_number_range(self):
        """Get number range based on curriculum step"""
        ranges = self.config.curriculum_digit_ranges
        min_num, max_num = ranges[min(self.curriculum_step, len(ranges)-1)]
        return max(0, min_num), max(max_num, min_num + 1)  # Ensure valid range
    
    def compute_carries(self, num1, num2):
        """Compute carries and digits for addition in LSB to MSB order"""
        total = num1 + num2
        num1_digits = [int(d) for d in str(num1).zfill(3)]  # Pad to 3 digits
        num2_digits = [int(d) for d in str(num2).zfill(3)]  # Pad to 3 digits
        
        # Store carries and digits
        result = []
        carry = 0
        
        # Process from right to left (LSB to MSB)
        for i in range(2, -1, -1):  # 2,1,0 for 3 digits
            d1 = num1_digits[i]
            d2 = num2_digits[i]
            current_sum = d1 + d2 + carry
            digit = current_sum % 10
            carry = current_sum // 10
            result.append((digit, carry))
        
        return result  # Returns in LSB to MSB order
    
    def generate_detailed_sequence(self, num1, num2):
        """Generate sequence with explicit LSB-to-MSB progression"""
        # Format validation
        if not (0 <= num1 <= 999 and 0 <= num2 <= 999):
            return None
            
        # Calculate addition
        total = num1 + num2
        
        # Get carries and digits from LSB to MSB
        carries = self.compute_carries(num1, num2)
        
        # Create the sequence with LSB-to-MSB progression
        lines = [
            f"Input: {num1}+{num2}\n",
            "Target:\n"
        ]
        
        # Add carry steps from LSB to MSB
        for i, (digit, prev_carry) in enumerate(carries):
            if i < len(carries) - 1:
                lines.append(f"A->{digit}, C->{prev_carry}\n")
            else:
                lines.append(f"A->{digit}, C->{prev_carry}.\n")  # Last step has period
        
        # Add final result
        lines.append(f"{total}$")
        
        complete_str = ''.join(lines)
        
        # Convert to tokens
        tokens = []
        for c in complete_str:
            if c not in self.vocab:
                print(f"Warning: Character '{c}' not in vocabulary!")
                return None
            tokens.append(self.vocab[c])
                
        return torch.tensor(tokens, dtype=torch.long)

    def generate_balanced_dataset(self):
        """Generate dataset with balanced digit lengths and carries"""
        min_num, max_num = self.get_number_range()
        
        samples_per_step = self.config.curriculum_samples[self.curriculum_step]
        target_per_length = max(1, samples_per_step // 3)
        target_per_carry = max(1, samples_per_step // 4)
        
        pbar = tqdm(total=samples_per_step, desc="Generating samples")
        attempts = 0
        max_attempts = samples_per_step * 100  # Increased threshold
        
        while len(self.data) < samples_per_step and attempts < max_attempts:
            attempts += 1
            try:
                num1 = random.randint(min_num, max_num)
                num2 = random.randint(min_num, max_num)
                
                if len(str(num1)) > 3 or len(str(num2)) > 3:
                    continue
                    
                carries = self.compute_carries(num1, num2)
                carry_count = sum(1 for _, c in carries if c > 0)
                
                sequence = self.generate_detailed_sequence(num1, num2)
                if sequence is not None and len(sequence) <= self.max_length:
                    self.data.append(sequence)
                    pbar.update(1)
            
            except Exception as e:
                print(f"Error generating sample: {e}")
                continue
        
        pbar.close()
        
        if len(self.data) < samples_per_step:
            print(f"Warning: Generated {len(self.data)}/{samples_per_step} samples after {attempts} attempts")
    
    def generate_dataset(self):
        """Generate the dataset"""
        self.generate_balanced_dataset()  # Already has its own progress bar

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Find the position of "Target:\n"
        target_str = "Target:\n"
        target_tokens = [self.vocab[c] for c in target_str]
        target_pos = 0
        
        # Find the target_str in the sequence
        for i in range(len(sequence) - len(target_tokens)):
            if all(sequence[i + j] == target_tokens[j] for j in range(len(target_tokens))):
                target_pos = i
                break
                
        # Split into input and target parts properly
        input_seq = sequence[:target_pos + len(target_tokens)]  # Include "Target:\n"
        target_seq = sequence[target_pos + len(target_tokens):]  # Start after "Target:\n"
        
        return input_seq, target_seq

def create_padding_mask(tensor, pad_token=0):
    """Create mask for padding tokens"""
    return (tensor != pad_token).float()

def collate_fn(batch):
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    # Find max lengths
    max_input_len = max(len(x) for x in inputs)
    max_target_len = max(len(y) for y in targets)
    
    # Create padded tensors
    padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    
    # Fill padded tensors
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
        samples = self.config.curriculum_samples[self.current_curriculum]
        
        self.train_dataset = ProcessAdditionDataset(
            self.config, 
            samples,
            self.config.train_seed,
            self.current_curriculum
        )
        
        self.val_dataset = ProcessAdditionDataset(
            self.config, 
            self.config.val_samples,
            self.config.val_seed,
            self.current_curriculum
        )
        
        # Add DataLoader creation
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
            padding_mask = create_padding_mask(targets).to(self.config.device)
            
            # Forward pass
            logits, loss = self.model(inputs, targets)
            
            # Ensure logits and targets have same sequence length
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]
            padding_mask = padding_mask[:, :min_len]
            
            # Calculate metrics
            predictions = logits.argmax(dim=-1)
            active_elements = padding_mask.sum()
            
            if active_elements > 0:
                total_correct += ((predictions == targets) * padding_mask).sum().item()
                total_tokens += active_elements.item()
                
                sequence_match = ((predictions == targets) * padding_mask).all(dim=1)
                sequence_correct += sequence_match.sum().item()
                total_sequences += inputs.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_description(
                f"Loss: {loss.item():.4f}, "
                f"Token Acc: {total_correct/total_tokens:.4f}, "
                f"Seq Acc: {sequence_correct/total_sequences:.4f}"
            )
            
            total_loss += loss.item()
            
            if self.global_step % 1000 == 0:
                self.log_sample_predictions(inputs[0], targets[0], predictions[0])
            
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()
                
            self.global_step += 1
        
        return (total_loss / len(self.train_loader), 
                total_correct / total_tokens,
                sequence_correct / total_sequences)
    
    def log_sample_predictions(self, input_seq, target_seq, pred_seq):
        """Enhanced logging of predictions with error handling"""
        print("\nSample Prediction Debug:")
        print("-" * 50)
        
        # Convert sequences to strings with error handling
        def safe_convert_to_string(seq, vocab):
            return ''.join([vocab.get(t.item(), '?') for t in seq])
        
        try:
            input_str = safe_convert_to_string(input_seq, self.train_dataset.inv_vocab)
            target_str = safe_convert_to_string(target_seq, self.train_dataset.inv_vocab)
            pred_str = safe_convert_to_string(pred_seq, self.train_dataset.inv_vocab)
            
            # Print full sequences
            print("Input sequence:")
            print(input_str)
            print("\nExpected target:")
            print(target_str)
            print("\nModel prediction:")
            print(pred_str)
            
            # Try to parse and verify arithmetic
            input_parts = input_str.split('+')
            num1 = int(input_parts[0].split(':')[1].strip())
            num2 = int(input_parts[1].split('\n')[0].strip())
            total = num1 + num2
            
            # Parse carries from prediction
            pred_lines = pred_str.split('\n')
            carry_steps = [l for l in pred_lines if l.startswith('A->')]
            result_line = next((l for l in reversed(pred_lines) if l.strip() and l[0].isdigit()), None)
            
            print("\nArithmetic Verification:")
            print(f"Operation: {num1} + {num2} = {total}")
            print("Carry steps found:")
            for step in carry_steps:
                print(f"  {step}")
            if result_line:
                pred_result = int(result_line.strip().replace('$', ''))
                print(f"Predicted result: {pred_result}")
                print(f"Correct: {pred_result == total}")
        except Exception as e:
            print(f"\nError in prediction analysis: {str(e)}")
    
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

            # Truncate predictions and targets to the same length
            predictions = logits.argmax(dim=-1)
            seq_len = min(predictions.size(1), targets.size(1))
            predictions = predictions[:, :seq_len]
            targets = targets[:, :seq_len]
            padding_mask = padding_mask[:, :seq_len]

            # Calculate metrics
            active_elements = padding_mask.sum()
            if active_elements > 0:
                total_correct += ((predictions == targets) * padding_mask).sum().item()
                total_tokens += active_elements.item()
                
                sequence_match = ((predictions == targets) * padding_mask).all(dim=1)
                sequence_correct += sequence_match.sum().item()
                total_sequences += inputs.size(0)
                
            total_loss += loss.item()
            
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
                    self.current_curriculum = min(
                        self.current_curriculum + 1,
                        len(self.config.curriculum_samples) - 1
                    )
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
            
            # Create input sequence matching training format
            carries = trainer.train_dataset.compute_carries(num1, num2)
            input_str = f"Input: {num1}+{num2}\nTarget:\n"
            
            # Convert to tokens
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
                if next_token.item() == vocab['$']:
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                    
                generated = torch.cat([generated, next_token], dim=1)
            
            # Convert output tokens to string
            output = ''.join([inv_vocab.get(t.item(), '?') for t in generated[0]])
            
            # Enhanced output verification
            print("\nDetailed Output Analysis:")
            print("Generated sequence:")
            print(output)
            
            # Verify carry steps and result
            try:
                lines = output.split('\n')
                carry_steps = [l for l in lines if l.startswith('A->')]
                result_line = next((l for l in reversed(lines) 
                                  if l.strip() and l[0].isdigit()), None)
                
                print("\nStep-by-step verification:")
                print("1. Carry operations (should be LSB to MSB):")
                for step in carry_steps:
                    print(f"  {step}")
                    
                if result_line:
                    predicted = int(result_line.strip().replace('$', ''))
                    actual = num1 + num2
                    print("\n2. Final result verification:")
                    print(f"  Predicted: {predicted}")
                    print(f"  Actual: {actual}")
                    print(f"  Correct: {predicted == actual}")
                    
                    # Verify LSB-to-MSB ordering
                    expected_carries = carries  # From compute_carries function
                    print("\n3. LSB-to-MSB ordering check:")
                    for i, ((pred_step, _), (exp_digit, exp_carry)) in enumerate(zip(zip(carry_steps, [0]*len(carry_steps)), expected_carries)):
                        print(f"  Position {i}: Expected A->{exp_digit}, C->{exp_carry}")
                        print(f"           Got: {pred_step}")
            except Exception as e:
                print(f"\nError in verification: {str(e)}")

def main():
    config = Config()
    trainer = Trainer(config)
    
    # Train the model
    trainer.train()
    
    # Run inference tests
    test_model_inference(trainer)

if __name__ == "__main__":
    main()