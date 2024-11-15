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

import locale

# Try to set the locale to en_US.UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, '')  # Try system default
    except:
        pass  # If both fail, continue anyway

# Also set PYTHONIOENCODING
os.environ['PYTHONIOENCODING'] = 'utf-8'


class EmergenceAnalyzer:
    def __init__(self, models_dir="Emergence_Models", seed=42, samples_per_length=100):
        self.models_dir = Path(models_dir)
        self.seed = seed
        self.samples_per_length = samples_per_length
        self.max_test_length = 20
        
        # Set random seeds first
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            if torch.cuda.is_available():
                # Try with default settings first
                torch.cuda.init()
                dummy_tensor = torch.zeros(1).cuda()
                _ = dummy_tensor + 1  # Test basic operation
                self.device = torch.device('cuda')
                print(f"CUDA initialized successfully")
            else:
                self.device = torch.device('cpu')
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU")
            self.device = torch.device('cpu')
            torch.cuda.is_available = lambda: False  # Prevent further CUDA attempts
        
        print(f"Using device: {self.device}")
        
        # Results storage
        self.model_results = {}
        
    def generate_n_digit_number(self, n):
        """Generate an n-digit number using string manipulation"""
        first_digit = random.randint(1, 9)
        rest_digits = [random.randint(0, 9) for _ in range(n-1)]
        return int(str(first_digit) + ''.join(map(str, rest_digits)))

    def generate_test_problems(self, length1, length2):
        """Generate test addition problems of specified lengths"""
        num1 = self.generate_n_digit_number(length1)
        num2 = self.generate_n_digit_number(length2)
        result = num1 + num2
        return num1, num2, result

    def load_model(self, model_path):
        """Load a model and its configuration with standardized vocabulary"""
        # Load with map_location to handle CUDA architecture differences
        checkpoint = torch.load(model_path, map_location='cpu')  # First load to CPU
        print("Loaded vocabulary:", checkpoint.get('vocab'))
        print("Inverse vocabulary:", checkpoint.get('inv_vocab'))
        config = checkpoint['model_config']
        
        # Calculate head_size from embed_size and num_heads
        head_size = config['embed_size'] // config['num_heads']
        
        model = create_arithmetic_transformer(
            vocab_size=config['vocab_size'],
            embed_size=config['embed_size'],
            num_heads=config['num_heads'],
            head_size=head_size,
            ff_dim=config['ff_dim'],
            num_layers=config['num_layers'],
            max_length=config['max_length']
        )
        
        # Load state dict to CPU first, then transfer to GPU
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        # Explicitly put model in eval mode
        model.eval()
        
        # Get vocabulary from checkpoint
        vocab = checkpoint.get('vocab', {})
        inv_vocab = checkpoint.get('inv_vocab', {})
        
        # Print model device location for debugging
        print(f"Model device: {next(model.parameters()).device}")
        
        # Try to run a small forward pass to verify GPU compatibility
        try:
            with torch.no_grad():
                dummy_input = torch.zeros((1, config['max_length']), dtype=torch.long).to(self.device)
                _ = model(dummy_input)
                print("Model successfully verified on device")
        except Exception as e:
            print(f"GPU verification failed, falling back to CPU: {str(e)}")
            model = model.cpu()
            self.device = torch.device('cpu')
        
        return model, vocab, inv_vocab

    def preprocess_input(self, input_str, max_length, vocab):
        """Preprocess input string for model inference"""
        input_str = input_str[::-1]  # Reverse string
        tokens = [vocab[c] for c in input_str if c in vocab]
        padded = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
        return torch.tensor(padded).unsqueeze(0).to(self.device)

    def decode_output(self, output_tensor, inv_vocab):
        """Decode model output with more robust token handling"""
        _, predicted = output_tensor.max(2)
        decoded = []
        for token in predicted[0]:
            token_val = token.item()
            # Check for both EOS and PAD tokens, accounting for possible vocabulary differences
            if token_val in [inv_vocab.get('<EOS>', 13), inv_vocab.get('EOS', 13)]:  # 13 is the typical EOS token value
                break
            if token_val not in [inv_vocab.get('<PAD>', 12), inv_vocab.get('PAD', 12)]:  # 12 is the typical PAD token value
                if token_val in inv_vocab:
                    decoded.append(inv_vocab[token_val])
        return ''.join(decoded)[::-1]

    def evaluate_single_problem(self, model, vocab, inv_vocab, num1, num2, max_length):  # Add max_length parameter
        """Evaluate model on a single addition problem"""
        input_str = f"{num1}+{num2}="
        expected = str(num1 + num2)
        
        input_tensor = self.preprocess_input(input_str, max_length, vocab)  # Use passed max_length
        
        with torch.no_grad():
            output = model(input_tensor)
            predicted = self.decode_output(output, inv_vocab)
        
        # Calculate various metrics
        metrics = {
            'exact_match': predicted == expected,
            'length_match': len(predicted) == len(expected),
            'carries': len(expected) > max(len(str(num1)), len(str(num2))),
            'correct_with_carry': predicted == expected and len(expected) > max(len(str(num1)), len(str(num2)))
        }
        
        # Calculate digit accuracy only if lengths match
        if metrics['length_match']:
            metrics['digit_accuracy'] = sum(p == e for p, e in zip(predicted, expected)) / len(expected)
        else:
            metrics['digit_accuracy'] = 0
            
        return metrics

    def evaluate_model(self, model_path):
        """Evaluate a single model across all test cases"""
        print(f"\nEvaluating {model_path.name}")
        model, vocab, inv_vocab = self.load_model(model_path)
        
        # Get max_length from the saved config
        checkpoint = torch.load(model_path, map_location=self.device)
        max_length = checkpoint['model_config']['max_length']
        
        results = {
            'per_length': {},
            'aggregate': {
                'total_exact_match': 0,
                'total_length_match': 0,
                'avg_digit_accuracy': 0.0,
                'carry_accuracy': 0.0,
                'total_problems_with_carry': 0
            }
        }
        
        total_problems = 0
        
        # Test problems for each length combination
        for length1 in tqdm(range(1, self.max_test_length + 1)):
            for length2 in range(1, self.max_test_length + 1):
                key = f"{length1}_{length2}"
                results['per_length'][key] = {
                    'exact_match': 0,
                    'length_match': 0,
                    'digit_accuracy': 0.0,
                    'carry_accuracy': 0.0,
                    'problems_with_carry': 0,
                    'samples': self.samples_per_length
                }
                
                for _ in range(self.samples_per_length):
                    num1, num2, _ = self.generate_test_problems(length1, length2)
                    metrics = self.evaluate_single_problem(model, vocab, inv_vocab, num1, num2, max_length)  # Pass max_length
                    
                    # Update per-length metrics
                    results['per_length'][key]['exact_match'] += metrics['exact_match']
                    results['per_length'][key]['length_match'] += metrics['length_match']
                    results['per_length'][key]['digit_accuracy'] += metrics['digit_accuracy']
                    
                    if metrics['carries']:
                        results['per_length'][key]['problems_with_carry'] += 1
                        results['per_length'][key]['carry_accuracy'] += metrics['correct_with_carry']
                        results['aggregate']['total_problems_with_carry'] += 1
                        results['aggregate']['carry_accuracy'] += metrics['correct_with_carry']
                    
                    # Update aggregate metrics
                    results['aggregate']['total_exact_match'] += metrics['exact_match']
                    results['aggregate']['total_length_match'] += metrics['length_match']
                    results['aggregate']['avg_digit_accuracy'] += metrics['digit_accuracy']
                    
                    total_problems += 1
                
                # Compute averages for this length combination
                results['per_length'][key]['exact_match'] /= self.samples_per_length
                results['per_length'][key]['length_match'] /= self.samples_per_length
                results['per_length'][key]['digit_accuracy'] /= self.samples_per_length
                if results['per_length'][key]['problems_with_carry'] > 0:
                    results['per_length'][key]['carry_accuracy'] /= results['per_length'][key]['problems_with_carry']
        
        # Compute overall averages
        results['aggregate']['total_exact_match'] /= total_problems
        results['aggregate']['total_length_match'] /= total_problems
        results['aggregate']['avg_digit_accuracy'] /= total_problems
        if results['aggregate']['total_problems_with_carry'] > 0:
            results['aggregate']['carry_accuracy'] /= results['aggregate']['total_problems_with_carry']
        
        return results

    def analyze_all_models(self):
        """Analyze all models in the directory"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Evaluate all models
        model_paths = sorted(self.models_dir.glob('*.pth'), 
                        key=lambda x: int(x.stem.split('_')[1].replace('k', '000').replace('M', '000000')))
        
        for model_path in model_paths:
            try:
                print(f"\nProcessing {model_path.name}")
                self.model_results[model_path.name] = self.evaluate_model(model_path)
            except Exception as e:
                print(f"Error processing {model_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if self.model_results:
            try:
                self.generate_report(timestamp)
                self.generate_visualizations(timestamp)
            except Exception as e:
                print(f"Error generating results: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("No models were successfully processed.")

    def generate_report(self, timestamp):
        """Generate detailed report of results"""
        report_path = f"emergence_analysis_report_{timestamp}.csv"
        
        # Prepare data for report
        report_data = []
        for model_name, results in self.model_results.items():
            param_count = self._extract_param_count(model_name)
            aggregate = results['aggregate']
            
            row = {
                'model_name': model_name,
                'parameters': param_count,
                'exact_match_accuracy': aggregate['total_exact_match'],
                'length_match_accuracy': aggregate['total_length_match'],
                'digit_accuracy': aggregate['avg_digit_accuracy'],
                'carry_accuracy': aggregate['carry_accuracy'],
                'total_problems_with_carry': aggregate['total_problems_with_carry']
            }
            
            # Add per-length results
            for length_key, length_results in results['per_length'].items():
                len1, len2 = length_key.split('_')
                prefix = f"len_{len1}_{len2}"
                row.update({
                    f"{prefix}_exact_match": length_results['exact_match'],
                    f"{prefix}_digit_accuracy": length_results['digit_accuracy'],
                    f"{prefix}_carry_accuracy": length_results['carry_accuracy'],
                    f"{prefix}_problems_with_carry": length_results['problems_with_carry']
                })
            
            report_data.append(row)
        
        # Save to CSV
        pd.DataFrame(report_data).to_csv(report_path, index=False)
        print(f"\nDetailed report saved to {report_path}")

    def generate_visualizations(self, timestamp):
        """Generate visualizations of the results"""
        fig = plt.figure(figsize=(20, 15))
        
        # Extract metrics for plotting
        param_counts = []
        exact_match = []
        digit_accuracy = []
        carry_accuracy = []
        
        for model_name, results in self.model_results.items():
            param_count = self._extract_param_count(model_name)
            param_counts.append(param_count)
            exact_match.append(results['aggregate']['total_exact_match'])
            digit_accuracy.append(results['aggregate']['avg_digit_accuracy'])
            carry_accuracy.append(results['aggregate']['carry_accuracy'])
        
        # Plot 1: Overall metrics vs parameter count
        plt.subplot(2, 2, 1)
        plt.semilogx(param_counts, exact_match, 'b.-', label='Exact Match')
        plt.semilogx(param_counts, digit_accuracy, 'r.-', label='Digit Accuracy')
        plt.semilogx(param_counts, carry_accuracy, 'g.-', label='Carry Accuracy')
        plt.grid(True)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Accuracy')
        plt.title('Model Performance vs Parameter Count')
        plt.legend()
        
        # Plot 2: Performance heatmap for largest model
        plt.subplot(2, 2, 2)
        latest_model = max(self.model_results.keys())
        exact_match_data = self._prepare_heatmap_data(self.model_results[latest_model]['per_length'], 
                                                     'exact_match')
        
        sns.heatmap(exact_match_data, cmap='viridis')
        plt.title(f'Exact Match Accuracy by Input Lengths\n({latest_model})')
        plt.xlabel('Length of Second Number')
        plt.ylabel('Length of First Number')
        
        # Plot 3: Carry accuracy heatmap for largest model
        plt.subplot(2, 2, 3)
        carry_accuracy_data = self._prepare_heatmap_data(self.model_results[latest_model]['per_length'], 
                                                        'carry_accuracy')
        
        sns.heatmap(carry_accuracy_data, cmap='viridis')
        plt.title(f'Carry Accuracy by Input Lengths\n({latest_model})')
        plt.xlabel('Length of Second Number')
        plt.ylabel('Length of First Number')
        
        # Plot 4: Problems with carries heatmap
        plt.subplot(2, 2, 4)
        carries_data = self._prepare_heatmap_data(self.model_results[latest_model]['per_length'], 
                                                 'problems_with_carry')
        
        sns.heatmap(carries_data, cmap='viridis')
        plt.title(f'Problems with Carries by Input Lengths\n({latest_model})')
        plt.xlabel('Length of Second Number')
        plt.ylabel('Length of First Number')
        
        plt.tight_layout()
        plt.savefig(f'emergence_analysis_plots_{timestamp}.png')
        print(f"Visualizations saved to emergence_analysis_plots_{timestamp}.png")

    def _extract_param_count(self, model_name):
        """Extract parameter count from model name"""
        param_str = model_name.split('_')[1]
        if 'k' in param_str:
            return float(param_str.replace('k', '')) * 1000
        elif 'M' in param_str:
            return float(param_str.replace('M', '')) * 1_000_000
        return float(param_str)

    def _prepare_heatmap_data(self, per_length_results, metric):
        """Prepare data for heatmap visualization"""
        data = np.zeros((self.max_test_length, self.max_test_length))
        for key, results in per_length_results.items():
            i, j = map(int, key.split('_'))
            data[i-1, j-1] = results[metric]
        return data

if __name__ == "__main__":
    try:
        # Create analyzer and run analysis
        analyzer = EmergenceAnalyzer(seed=42, samples_per_length=100)
        analyzer.analyze_all_models()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()