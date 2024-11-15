import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Scripts.ArithmeticTransformer import create_arithmetic_transformer

def test_model(model_path="Emergence_Models/model_01_38k.pth"):
    
    # Print the model path
    print(f"Model path: {model_path}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inv_vocab']
    
    model = create_arithmetic_transformer(
        vocab_size=config['vocab_size'],
        embed_size=config['embed_size'],
        num_heads=config['num_heads'],
        head_size=config['embed_size'] // config['num_heads'],
        ff_dim=config['ff_dim'],
        num_layers=config['num_layers'],
        max_length=config['max_length']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup results storage
    os.makedirs("Emergence_Analysis_Results", exist_ok=True)
    results_file = "Emergence_Analysis_Results/results.csv"
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['length1', 'length2', 'accuracy'])
        
        # Test different lengths
        for length1 in tqdm(range(1, 21)):
            for length2 in range(1, 21):
                correct = 0
                total = 100  # 100 samples per length pair
                
                for _ in range(total):
                    # Generate problem
                    num1 = random.randint(10**(length1-1), 10**length1 - 1)
                    num2 = random.randint(10**(length2-1), 10**length2 - 1)
                    expected = num1 + num2
                    
                    # Preprocess
                    input_str = f"{num1}+{num2}="[::-1]
                    tokens = [vocab[c] for c in input_str if c in vocab]
                    padded = tokens + [vocab['<PAD>']] * (config['max_length'] - len(tokens))
                    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = output.max(2)
                        
                    # Decode output
                    result = []
                    for token in predicted[0]:
                        if token.item() == vocab.get('<EOS>', 13):
                            break
                        if token.item() not in [vocab.get('<PAD>', 12), vocab.get('<EOS>', 13)]:
                            result.append(inv_vocab[token.item()])
                    
                    predicted_num = int(''.join(result)[::-1])
                    correct += (predicted_num == expected)
                
                accuracy = correct/total
                writer.writerow([length1, length2, accuracy])
                print(f"Length {length1}-{length2}: {accuracy:.2%}")

if __name__ == "__main__":
    test_model()