import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Scripts.ArithmeticTransformer import create_arithmetic_transformer

class EmergenceAnalyzer:
    def __init__(self, models_dir="Emergence_Models", results_dir="Results", seed=42, samples_per_length=100):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.seed = seed
        self.samples_per_length = samples_per_length
        self.max_test_length = 20
        
        # Set random seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
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
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, vocab, inv_vocab, config

    def calculate_metrics(self, predicted_str, expected_str):
        metrics = {
            'exact_match': predicted_str == expected_str,
            'token_edit_distance': self.levenshtein_distance(predicted_str, expected_str),
            'normalized_edit_distance': self.levenshtein_distance(predicted_str, expected_str) / max(len(predicted_str), len(expected_str)),
            'digit_accuracy': sum(p == e for p, e in zip(predicted_str.zfill(len(expected_str)), expected_str)) / len(expected_str)
        }
        return metrics

    @staticmethod
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return EmergenceAnalyzer.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def evaluate_model(self, model_path):
        print(f"\nEvaluating {model_path.name}")
        model, vocab, inv_vocab, config = self.load_model(model_path)
        
        results = []
        for length1 in tqdm(range(1, self.max_test_length + 1)):
            for length2 in range(1, self.max_test_length + 1):
                length_metrics = {
                    'exact_match': [],
                    'token_edit_distance': [],
                    'normalized_edit_distance': [],
                    'digit_accuracy': []
                }
                
                for _ in range(self.samples_per_length):
                    num1 = random.randint(10**(length1-1), 10**length1 - 1)
                    num2 = random.randint(10**(length2-1), 10**length2 - 1)
                    expected = str(num1 + num2)
                    
                    input_str = f"{num1}+{num2}="[::-1]
                    tokens = [vocab[c] for c in input_str if c in vocab]
                    padded = tokens + [vocab['<PAD>']] * (config['max_length'] - len(tokens))
                    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(self.device)
                    
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
                    predicted_str = ''.join(result)[::-1]
                    
                    metrics = self.calculate_metrics(predicted_str, expected)
                    for key, value in metrics.items():
                        length_metrics[key].append(value)
                
                result_row = {
                    'length1': length1,
                    'length2': length2,
                    'exact_match': np.mean(length_metrics['exact_match']),
                    'token_edit_distance': np.mean(length_metrics['token_edit_distance']),
                    'normalized_edit_distance': np.mean(length_metrics['normalized_edit_distance']),
                    'digit_accuracy': np.mean(length_metrics['digit_accuracy'])
                }
                results.append(result_row)
        
        return pd.DataFrame(results)

    def analyze_all_models(self):
        model_paths = sorted(self.models_dir.glob('*.pth'),
                           key=lambda x: int(x.stem.split('_')[1].replace('k', '000').replace('M', '000000')))
        
        aggregate_results = []
        for model_path in model_paths:
            try:
                # Evaluate individual model
                results_df = self.evaluate_model(model_path)
                
                # Save individual model results
                model_results_path = self.results_dir / f"{model_path.stem}_results.csv"
                results_df.to_csv(model_results_path, index=False)
                
                # Calculate aggregate metrics for this model
                model_summary = {
                    'model_name': model_path.name,
                    'parameters': int(model_path.stem.split('_')[1].replace('k', '000').replace('M', '000000')),
                    'avg_exact_match': results_df['exact_match'].mean(),
                    'avg_token_edit_distance': results_df['token_edit_distance'].mean(),
                    'avg_normalized_edit_distance': results_df['normalized_edit_distance'].mean(),
                    'avg_digit_accuracy': results_df['digit_accuracy'].mean()
                }
                aggregate_results.append(model_summary)
                
            except Exception as e:
                print(f"Error processing {model_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save aggregate results
        if aggregate_results:
            aggregate_df = pd.DataFrame(aggregate_results)
            aggregate_df.to_csv(self.results_dir / 'aggregate_results.csv', index=False)
            print(f"\nResults saved to {self.results_dir}")

if __name__ == "__main__":
    analyzer = EmergenceAnalyzer(seed=42, samples_per_length=100)
    analyzer.analyze_all_models()