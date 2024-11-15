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
    def __init__(self, models_dir="Emergence_Models", results_dir="Results", seed=42, samples_per_length=100, batch_size=32):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.seed = seed
        self.samples_per_length = samples_per_length
        self.max_test_length = 20
        self.batch_size = batch_size
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        torch.backends.cudnn.benchmark = True

    def prepare_batch(self, problems, vocab, max_length):
        batch_inputs = []
        for num1, num2 in problems:
            input_str = f"{num1}+{num2}="[::-1]
            tokens = [vocab[c] for c in input_str if c in vocab]
            padded = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
            batch_inputs.append(padded)
        return torch.tensor(batch_inputs, dtype=torch.long, device=self.device)

    def evaluate_model(self, model_path):
        print(f"\nEvaluating {model_path.name}")
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

        results = []
        for length1 in tqdm(range(1, self.max_test_length + 1)):
            for length2 in range(1, self.max_test_length + 1):
                batch_metrics = {
                    'exact_match': [],
                    'token_edit_distance': [],
                    'normalized_edit_distance': [],
                    'digit_accuracy': []
                }

                # Process in batches
                for batch_start in range(0, self.samples_per_length, self.batch_size):
                    batch_size = min(self.batch_size, self.samples_per_length - batch_start)
                    
                    # Generate problems for the batch
                    problems = [
                        (random.randint(10**(length1-1), 10**length1 - 1),
                         random.randint(10**(length2-1), 10**length2 - 1))
                        for _ in range(batch_size)
                    ]
                    
                    expected = [str(n1 + n2) for n1, n2 in problems]
                    batch_input = self.prepare_batch(problems, vocab, config['max_length'])

                    with torch.no_grad():
                        output = model(batch_input)
                        _, predicted = output.max(2)

                    # Process batch predictions
                    for i in range(batch_size):
                        result = []
                        for token in predicted[i]:
                            if token.item() == vocab.get('<EOS>', 13):
                                break
                            if token.item() not in [vocab.get('<PAD>', 12), vocab.get('<EOS>', 13)]:
                                result.append(inv_vocab[token.item()])
                        predicted_str = ''.join(result)[::-1]
                        
                        # Calculate metrics
                        exact_match = predicted_str == expected[i]
                        edit_dist = self.levenshtein_distance(predicted_str, expected[i])
                        norm_edit_dist = edit_dist / max(len(predicted_str), len(expected[i]))
                        digit_acc = sum(p == e for p, e in zip(
                            predicted_str.zfill(len(expected[i])), 
                            expected[i]
                        )) / len(expected[i])

                        batch_metrics['exact_match'].append(exact_match)
                        batch_metrics['token_edit_distance'].append(edit_dist)
                        batch_metrics['normalized_edit_distance'].append(norm_edit_dist)
                        batch_metrics['digit_accuracy'].append(digit_acc)

                results.append({
                    'length1': length1,
                    'length2': length2,
                    'exact_match': np.mean(batch_metrics['exact_match']),
                    'token_edit_distance': np.mean(batch_metrics['token_edit_distance']),
                    'normalized_edit_distance': np.mean(batch_metrics['normalized_edit_distance']),
                    'digit_accuracy': np.mean(batch_metrics['digit_accuracy'])
                })

        return pd.DataFrame(results)

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

    def analyze_all_models(self):
        model_paths = sorted(self.models_dir.glob('*.pth'),
                           key=lambda x: int(x.stem.split('_')[1].replace('k', '000').replace('M', '000000')))
        
        aggregate_results = []
        for model_path in model_paths:
            try:
                results_df = self.evaluate_model(model_path)
                
                model_results_path = self.results_dir / f"{model_path.stem}_results.csv"
                results_df.to_csv(model_results_path, index=False)
                
                aggregate_results.append({
                    'model_name': model_path.name,
                    'parameters': int(model_path.stem.split('_')[1].replace('k', '000').replace('M', '000000')),
                    'avg_exact_match': results_df['exact_match'].mean(),
                    'avg_token_edit_distance': results_df['token_edit_distance'].mean(),
                    'avg_normalized_edit_distance': results_df['normalized_edit_distance'].mean(),
                    'avg_digit_accuracy': results_df['digit_accuracy'].mean()
                })
                
            except Exception as e:
                print(f"Error processing {model_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if aggregate_results:
            aggregate_df = pd.DataFrame(aggregate_results)
            aggregate_df.to_csv(self.results_dir / 'aggregate_results.csv', index=False)
            print(f"\nResults saved to {self.results_dir}")

if __name__ == "__main__":
    analyzer = EmergenceAnalyzer(seed=42, samples_per_length=100, batch_size=32)
    analyzer.analyze_all_models()