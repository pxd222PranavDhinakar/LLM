import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import os
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
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

class Config:
    """Configuration for the entire training process"""
    def __init__(self):
        # Model Architecture
        self.vocab_size = 16    # 0-9 plus special tokens
        self.embed_size = 256   # Medium size model
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_layers = 4
        self.max_length = 256   # Maximum sequence length
        self.dropout = 0.1
        
        # Training Parameters
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.max_epochs = 20
        self.warmup_steps = 1000
        self.train_samples = 50000
        self.eval_samples = 1000
        self.max_digit_length = 10  # Maximum length of input numbers
        
        # Dataset Parameters
        self.train_seed = 42
        self.eval_seed = 43
        
        # Save/Load Parameters
        self.save_dir = "checkpoints"
        self.model_name = "arithmetic_transformer"
        self.save_every = 1000  # Save every N steps
        
        # Debug Parameters
        self.debug_mode = True
        self.debug_every = 100   # Show debug info every N batches
        self.num_debug_examples = 3
        self.debug_problems = ["45+34", "999+1", "1000+234"]  # Example problems to test
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AutoregressiveAdditionDataset(Dataset):
    def __init__(self, config, num_samples, seed):
        self.config = config
        self.num_samples = num_samples
        
        # Add start-of-steps delimiter to vocabulary
        self.vocab = {str(i): i for i in range(10)}  # 0-9
        self.vocab.update({
            ',': 10,   # step element separator
            '>': 11,   # operation result
            ';': 12,   # step separator
            '|': 13,   # process-result separator
            '+': 14,   # addition operator
            '=': 15,   # equals sign
            ':': 16    # start-of-steps delimiter
        })
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Generate data
        random.seed(seed)
        self.data = self.generate_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = sequence[:-1]  # all but last token
        y = sequence[1:]   # all but first token
        return x, y
    
    def process_addition(self, num1, num2):
        """Generate step-by-step addition process"""
        steps = []
        
        # Convert to lists of digits and pad to same length
        num1_str = str(num1)
        num2_str = str(num2)
        max_len = max(len(num1_str), len(num2_str))
        num1_digits = list(map(int, num1_str.zfill(max_len)))
        num2_digits = list(map(int, num2_str.zfill(max_len)))
        
        carry = 0
        # Process from right to left but store steps from left to right
        for i in range(max_len-1, -1, -1):
            d1 = num1_digits[i]
            d2 = num2_digits[i]
            
            # Calculate step
            total = d1 + d2 + carry
            result_digit = total % 10
            carry = total // 10
            
            # Format: "d1,d2,carry_in>result,carry_out"
            step = f"{d1},{d2},{carry}>{result_digit},{carry}"
            steps.insert(0, step)  # Insert at front to maintain left-to-right order
        
        # Handle final carry if present
        if carry > 0:
            steps.insert(0, f"0,0,{carry}>{carry},0")
            
        # Remove leading zeros unless they're meaningful
        while steps and steps[0] == "0,0,0>0,0" and len(steps) > 1:
            steps.pop(0)
            
        return steps

    def generate_sequence(self, num1, num2):
        """Generate complete sequence for addition problem"""
        # Create input string with delimiter
        input_str = f"{num1}+{num2}"
        
        # Generate process steps
        steps = self.process_addition(num1, num2)
        process_str = ";".join(steps)
        
        # Calculate result
        result = str(num1 + num2)
        
        # Combine everything with the start-of-steps delimiter
        complete_str = f"{input_str}:{process_str}|={result}"
        
        # Convert to tokens
        return torch.tensor([self.vocab[c] for c in complete_str], dtype=torch.long)
    
    def generate_number(self, length):
        """Generate a random number of specified length"""
        if length == 1:
            return random.randint(0, 9)
        return random.randint(10**(length-1), (10**length)-1)
    
    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            len1 = random.randint(1, self.config.max_digit_length)
            len2 = random.randint(1, self.config.max_digit_length)
            num1 = self.generate_number(len1)
            num2 = self.generate_number(len2)
            sequence = self.generate_sequence(num1, num2)
            data.append(sequence)
        return data

def collate_batch(batch):
    """Collate batch for DataLoader with debugging info"""
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    # Get lengths
    lengths = [len(x) for x in inputs]
    max_len = max(lengths)
    
    # Pad sequences
    padded_inputs = torch.full((len(batch), max_len), 0, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), 0, dtype=torch.long)
    
    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(input_seq)] = input_seq
        padded_targets[i, :len(target_seq)] = target_seq
    
    return padded_inputs, padded_targets

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.global_step = 0
        
        self.setup()
    
    def setup(self):
        """Setup model, optimizer, datasets"""
        self.model = create_arithmetic_transformer(self.config)
        self.model.to(self.config.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95)
        )
        
        self.train_dataset = AutoregressiveAdditionDataset(
            self.config,
            self.config.train_samples,
            self.config.train_seed
        )
        
        self.eval_dataset = AutoregressiveAdditionDataset(
            self.config,
            self.config.eval_samples,
            self.config.eval_seed
        )
    
    def debug_step(self, inputs, targets, logits):
        """Print debug information for current step"""
        # Take first sequence from batch
        input_seq = inputs[0]
        target_seq = targets[0]
        pred_seq = logits[0].argmax(dim=-1)
        
        # Convert to strings
        input_str = ''.join([self.train_dataset.inv_vocab[t.item()] 
                        for t in input_seq if t.item() != 0])
        target_str = ''.join([self.train_dataset.inv_vocab[t.item()] 
                            for t in target_seq if t.item() != 0])
        pred_str = ''.join([self.train_dataset.inv_vocab[t.item()] 
                        for t in pred_seq if t.item() != 0])
        
        print("\nDebug Information:")
        print(f"Input:      {input_str}")
        print(f"Target:     {target_str}")
        print(f"Prediction: {pred_str}")
        
        # Show attention maps if available
        attention_maps = self.model.get_attention_maps()
        if attention_maps and len(attention_maps) > 0:
            print("\nAttention pattern (first layer, first head):")
            # Detach the attention tensor before converting to numpy
            attn = attention_maps[0][0][0].detach()  # [layer][head][batch]
            print(attn[:10, :10].cpu().numpy())  # Show first 10x10 positions
            
    # In Trainer class, add this method:
    def debug_generation(self, input_str="45+34"):
        """Debug the generation process"""
        self.model.eval()
        print("\nTesting autoregressive generation:")
        
        # Convert input to tokens
        input_tokens = torch.tensor(
            [self.train_dataset.vocab[c] for c in input_str + ":"],
            dtype=torch.long
        ).unsqueeze(0).to(self.config.device)
        
        # Generate step by step
        with torch.no_grad():
            for i in range(50):  # Maximum steps
                # Get prediction for next token
                logits = self.model(input_tokens)
                next_token_logits = logits[0, -1]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = next_token_probs.argmax()
                
                # Convert current sequence to string
                current_str = ''.join([
                    self.train_dataset.inv_vocab[t.item()] 
                    for t in input_tokens[0]
                ])
                next_char = self.train_dataset.inv_vocab[next_token.item()]
                
                print(f"Input: {current_str}")
                print(f"Predicted next: {next_char}\n")
                
                # Add prediction to sequence
                input_tokens = torch.cat([
                    input_tokens, 
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # Stop if we hit equals sign (end of process)
                if next_char == '=':
                    # Generate one more token for the final answer
                    final_logits = self.model(input_tokens)
                    final_pred = final_logits[0, -1].argmax()
                    final_char = self.train_dataset.inv_vocab[final_pred.item()]
                    print(f"Final answer: {final_char}")
                    break
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            logits, loss = self.model(inputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = logits.argmax(-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.numel()
            
            # Debug output
            if self.config.debug_mode and batch_idx % self.config.debug_every == 0:
                self.debug_step(inputs, targets, logits)
                # Add this line:
                self.debug_generation("45+34")  # You can change the example
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}",
                'step': self.global_step
            })
            
            self.global_step += 1
        
        return total_loss / len(train_loader), correct_predictions / total_predictions
    
    def generate_examples(self):
        """Generate some example predictions"""
        self.model.eval()
        print("\nGenerated Examples:")
        
        for _ in range(self.config.num_debug_examples):
            # Generate random numbers
            len1 = random.randint(1, 3)
            len2 = random.randint(1, 3)
            num1 = self.train_dataset.generate_number(len1)
            num2 = self.train_dataset.generate_number(len2)
            
            # Create input
            input_str = f"{num1}+{num2}"
            input_tokens = torch.tensor(
                [self.train_dataset.vocab[c] for c in input_str],
                dtype=torch.long
            ).unsqueeze(0).to(self.config.device)
            
            # Generate output
            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_tokens,
                    max_new_tokens=100,
                    temperature=0.7
                )
            
            # Convert to string
            output_str = ''.join([
                self.train_dataset.inv_vocab[t.item()]
                for t in output_tokens[0]
            ])
            
            print(f"Input: {input_str}")
            print(f"Generated: {output_str}\n")

def main():
    # Create config
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create dataloaders
    train_loader = DataLoader(
        trainer.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        persistent_workers=False
    )
    
    eval_loader = DataLoader(
        trainer.eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        persistent_workers=False
    )
    
    # Training loop
    best_loss = float('inf')
    
    try:
        for epoch in range(config.max_epochs):
            print(f"\nEpoch {epoch+1}/{config.max_epochs}")
            
            # Train
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Generate examples
            if config.debug_mode:
                trainer.generate_examples()
            
            # Save if best
            if train_loss < best_loss:
                best_loss = train_loss
                if config.save_dir:
                    os.makedirs(config.save_dir, exist_ok=True)
                    path = os.path.join(config.save_dir, f"{config.model_name}_best.pt")
                    torch.save(trainer.model.state_dict(), path)
                    print(f"Saved best model to {path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("Training completed!")

def test_dataset_samples():
    config = Config()
    dataset = AutoregressiveAdditionDataset(config, num_samples=5, seed=42)
    
    test_cases = [
        (45, 34),     # Basic case
        (999, 1),     # Lots of carries
        (1000, 234),  # Zeros in middle
        (5, 5),       # Same digits
        (0, 456)      # Leading zero
    ]
    
    print("\n=== Testing Addition Cases ===")
    for num1, num2 in test_cases:
        print(f"\n{num1} + {num2}:")
        sequence = dataset.generate_sequence(num1, num2)
        decoded = ''.join([dataset.inv_vocab[t.item()] for t in sequence])
        print(f"Full sequence: {decoded}")
        
def test_addition():
    config = Config()
    dataset = AutoregressiveAdditionDataset(config, num_samples=1, seed=42)
    
    # Test simple addition
    print("\n45 + 34:")
    result = dataset.process_addition(45, 34)
    for step in result:
        print(f"Step: {step}")
    print(f"Expected Result: 79")
    
    # Test addition with carries
    print("\n187 + 945:")
    result = dataset.process_addition(187, 945)
    for step in result:
        print(f"Step: {step}")
    print(f"Expected Result: 1132")


def test_model_architecture():
    """Test the AutoregressiveArithmeticTransformer architecture"""
    print("\nTesting Model Architecture")
    print("=" * 50)
    
    # Create a minimal config for testing
    class TestConfig:
        def __init__(self):
            self.vocab_size = 17     # 0-9 plus 7 special tokens
            self.embed_size = 64   
            self.num_heads = 2
            self.ff_dim = 256
            self.num_layers = 2
            self.max_length = 128   
            self.dropout = 0.1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.max_digit_length = 2  # Small numbers for testing
            self.train_samples = 2
            self.train_seed = 42

    config = TestConfig()
    
    # 1. Create test dataset and sample
    print("\n1. Testing Dataset Generation...")
    try:
        dataset = AutoregressiveAdditionDataset(config, num_samples=2, seed=42)
        sample = dataset[0]  # Get first sample
        print("✓ Dataset creation successful")
        print(f"Sample input shape: {sample[0].shape}")
        print(f"Sample target shape: {sample[1].shape}")
        
        # Decode a sample to verify format
        decoded_input = ''.join([dataset.inv_vocab[t.item()] for t in sample[0]])
        print(f"Sample decoded input: {decoded_input}")
    except Exception as e:
        print(f"✗ Dataset creation failed: {str(e)}")
        return
    
    # 2. Create and test model
    print("\n2. Testing Model Creation...")
    try:
        model = create_arithmetic_transformer(config).to(config.device)
        print("✓ Model creation successful")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model creation failed: {str(e)}")
        return
    
    # 3. Test forward pass
    print("\n3. Testing Forward Pass...")
    try:
        # Convert sample to batch
        x = sample[0].unsqueeze(0).to(config.device)
        y = sample[1].unsqueeze(0).to(config.device)
        
        # Test training mode (with targets)
        logits, loss = model(x, y)
        print("✓ Forward pass (training) successful")
        print(f"Loss value: {loss.item():.4f}")
        
        # Test inference mode
        logits = model(x)
        print("✓ Forward pass (inference) successful")
        print(f"Output logits shape: {logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
        return
    
    # 4. Test autoregressive generation
    print("\n4. Testing Autoregressive Generation...")
    try:
        # Create a simple test input
        test_input = "12+34:"
        input_tokens = torch.tensor(
            [[dataset.vocab[c] for c in test_input]], 
            dtype=torch.long
        ).to(config.device)
        
        print(f"Test input: {test_input}")
        print("Generating tokens:")
        
        # Generate step by step
        with torch.no_grad():
            current_tokens = input_tokens
            for i in range(20):  # Maximum steps
                logits = model(current_tokens)
                next_token = logits[0, -1].argmax()
                next_char = dataset.inv_vocab[next_token.item()]
                print(f"Step {i}: Predicted next token -> {next_char}")
                
                # Add to sequence
                current_tokens = torch.cat([
                    current_tokens,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # Stop if we hit equals sign
                if next_char == '=':
                    break
                    
        print("✓ Generation test successful")
    except Exception as e:
        print(f"✗ Generation test failed: {str(e)}")
        return
    
    print("\nArchitecture test completed successfully!")
    return model

if __name__ == "__main__":
    #main()  # Comment out during testing
    #test_dataset_samples()
    #test_addition()
     
    model = test_model_architecture()
