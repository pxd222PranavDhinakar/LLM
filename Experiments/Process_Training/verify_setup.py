# verify_setup.py
import os
import torch
from tqdm import tqdm
import random
from collections import defaultdict
from TestTrainer import Config, ProcessAdditionDataset, create_arithmetic_transformer

def verify_config():
    """Verify configuration settings"""
    print("\n=== Configuration Verification ===")
    config = Config()
    
    print("\nModel Parameters:")
    print(f"Vocabulary Size: {config.vocab_size} (Expected: 32)")
    print(f"Embedding Size: {config.embed_size} (Expected: 64)")
    print(f"Number of Heads: {config.num_heads} (Expected: 2)")
    print(f"FF Dimension: {config.ff_dim} (Expected: 256)")
    print(f"Number of Layers: {config.num_layers} (Expected: 2)")
    
    print("\nTraining Parameters:")
    print(f"Batch Size: {config.batch_size} (Expected: 32)")
    print(f"Learning Rate: {config.learning_rate} (Expected: 1e-3)")
    print(f"Max Epochs: {config.max_epochs} (Expected: 10)")
    
    print("\nDataset Parameters:")
    print(f"Training Samples: {config.train_samples}")
    print(f"Validation Samples: {config.val_samples}")
    print(f"Max Digit Range: {config.curriculum_digit_ranges[0][1]}")

def verify_sequence_examples():
    """Verify sequence generation for various cases"""
    print("\n=== Sequence Generation Verification ===")
    config = Config()
    dataset = ProcessAdditionDataset(config, num_samples=1, seed=42, curriculum_step=0)
    
    # Test cases covering different scenarios
    test_cases = [
        (7, 8),      # Basic carry
        (99, 1),     # Two digit with carry
        (5, 5),      # Same number
        (0, 9),      # Zero handling
        (123, 456),  # Three digit, no carry
        (999, 1),    # Maximum carry propagation
    ]
    
    for num1, num2 in test_cases:
        print(f"\nTesting: {num1} + {num2} = {num1 + num2}")
        sequence = dataset.generate_detailed_sequence(num1, num2)
        if sequence is not None:
            # Convert to string and print
            seq_str = ''.join([dataset.inv_vocab[t.item()] for t in sequence])
            print("Generated sequence:")
            print(seq_str)
            
            # Verify carries
            carries = dataset.compute_carries(num1, num2)
            num_carries = sum(1 for _, c in carries if c > 0)
            print(f"Number of carries: {num_carries}")

def verify_dataset_generation():
    """Verify dataset generation and distribution"""
    print("\n=== Dataset Generation Verification ===")
    config = Config()
    
    for curriculum_step in range(len(config.curriculum_digit_ranges)):
        print(f"\nTesting Curriculum Step {curriculum_step}")
        max_digits = len(str(config.curriculum_digit_ranges[curriculum_step][1]))
        target_samples = config.curriculum_samples[curriculum_step]
        
        dataset = ProcessAdditionDataset(
            config, 
            target_samples,
            seed=42, 
            curriculum_step=curriculum_step
        )
        
        # Analyze length distribution
        length_dist = defaultdict(int)
        carry_dist = defaultdict(int)
        total_samples = len(dataset.data)
        
        for sequence in dataset.data:
            # Convert sequence to string for analysis
            seq_str = ''.join([dataset.inv_vocab[t.item()] for t in sequence])
            if 'Input:' in seq_str:
                # Extract numbers
                parts = seq_str.split('+')
                num1 = int(parts[0].split(':')[1].strip())
                num2 = int(parts[1].split('\n')[0].strip())
                
                # Track length distribution
                key = f"{len(str(num1))},{len(str(num2))}"
                length_dist[key] += 1
                
                # Track carry distribution
                carries = dataset.compute_carries(num1, num2)
                num_carries = sum(1 for _, c in carries if c > 0)
                carry_dist[num_carries] += 1
        
        print(f"\nTotal samples generated: {total_samples}")
        print("\nLength Distribution:")
        for key, count in sorted(length_dist.items()):
            print(f"Length pair {key}: {count} samples ({count/total_samples*100:.1f}%)")
        
        print("\nCarry Distribution:")
        for carries, count in sorted(carry_dist.items()):
            print(f"{carries} carries: {count} samples ({count/total_samples*100:.1f}%)")

def run_verification():
    """Run all verification checks"""
    try:
        verify_config()
        verify_sequence_examples()
        verify_dataset_generation()
        print("\n=== All Verification Checks Complete ===")
    except Exception as e:
        print(f"\nVerification failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_verification()