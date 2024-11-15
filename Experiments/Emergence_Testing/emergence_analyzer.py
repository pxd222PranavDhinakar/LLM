import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Scripts.ArithmeticTransformer import create_arithmetic_transformer
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

class EmergenceAnalyzer:
    def __init__(self, models_dir="Emergence_Models", seed=42, samples_per_length=100):
        self.models_dir = Path(models_dir)
        self.seed = seed
        self.samples_per_length = samples_per_length
        self.max_test_length = 20
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        self.results = {}

    def generate_test_problems(self, length1, length2):
        num1 = random.randint(10**(length1-1), 10**length1 - 1)
        num2 = random.randint(10**(length2-1), 10**length2 - 1)
        return num1, num2, num1 + num2

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']
        
        model = create_arithmetic_transformer(
            vocab_size=config['vocab_size'],
            embed_size=config['embed_size'], 
            num_heads=config['num_heads'],
            head_size=config['embed_size'] // config['num_heads'],
            ff_dim=config['ff_dim'],
            num_layers=config['num_layers'],
            max_length=config['max_length']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint['vocab'], checkpoint['inv_vocab'], config['max_length']

    def evaluate_model(self, model_path):
        print(f"\nEvaluating {model_path.name}")
        model, vocab, inv_vocab, max_length = self.load_model(model_path)
        
        total_correct = 0
        total_problems = 0
        
        for length1 in tqdm(range(1, self.max_test_length + 1)):
            for length2 in range(1, self.max_test_length + 1):
                correct = 0
                
                for _ in range(self.samples_per_length):
                    num1, num2, expected = self.generate_test_problems(length1, length2)
                    
                    input_str = f"{num1}+{num2}="[::-1]
                    tokens = [vocab[c] for c in input_str if c in vocab]
                    padded = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
                    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = output.max(2)
                        
                    result = []
                    for token in predicted[0]:
                        if token.item() == vocab.get('<EOS>', 13):
                            break
                        if token.item() not in [vocab.get('<PAD>', 12), vocab.get('<EOS>', 13)]:
                            result.append(inv_vocab[token.item()])
                    
                    predicted_num = int(''.join(result)[::-1])
                    correct += (predicted_num == expected)
                    total_problems += 1
                
                total_correct += correct
                
                print(f"Length {length1}-{length2}: {correct/self.samples_per_length:.2%} accurate")
        
        print(f"\nOverall accuracy: {total_correct/total_problems:.2%}")
        return total_correct/total_problems

    def analyze_all_models(self):
        for model_path in sorted(self.models_dir.glob('*.pth')):
            try:
                self.results[model_path.name] = self.evaluate_model(model_path)
            except Exception as e:
                print(f"Error processing {model_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    analyzer = EmergenceAnalyzer(samples_per_length=100)
    analyzer.analyze_all_models()