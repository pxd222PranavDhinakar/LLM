# evaluate.py
import argparse
from math import log2
from model_builder import NGram
import argparse

def calculate_perplexity(model, test_corpus):
    """
    Calculate the perplexity of the language model on the given test corpus.
    
    Args:
    model (NGram): The trained language model.
    test_corpus (list of str): The test corpus as a list of words.
    
    Returns:
    float: The perplexity of the model on the test corpus.
    """
    N = len(test_corpus)
    log_probability = 0.0

    for i in range(N):
        context = test_corpus[max(0, i-model.n+1):i]
        word = test_corpus[i]
        prob = model.prob(context, word)
        
        log_probability += log2(prob)

    perplexity = 2 ** (-1/N * log_probability)

    return perplexity

def evaluate_model(model, test_file):
    """
    Evaluate the model on a test file and print the perplexity.
    
    Args:
    model (NGram): The trained language model.
    test_file (str): Path to the test file.
    """
    with open(test_file, 'r') as f:
        test_text = f.read().lower() 
    
    test_corpus = test_text.split()
    
    perplexity = calculate_perplexity(model, test_corpus)
    print(f"Language Model: {model.n}-gram model")
    print(f"Perplexity: {perplexity:.2f}")


# Unscramble function that builds the sentence
def unscramble(ngram: NGram, scrambled_words: list[str]) -> str:
    sentence = []
    remaining_words = scrambled_words[:]
    current_word = remaining_words.pop(0)
    sentence.append(current_word)

    while remaining_words:
        best_next_word = None
        best_prob = -1
        for next_word in remaining_words:
            prob = ngram.prob([current_word], next_word)
            if prob > best_prob:
                best_next_word = next_word
                best_prob = prob
        current_word = best_next_word
        sentence.append(current_word)
        remaining_words.remove(current_word)

    return " ".join(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models and unscramble sentences.")
    parser.add_argument('--model', required=True, help='Path to the language model file')
    parser.add_argument('--evaluate', help='Path to the file for evaluating perplexity')
    parser.add_argument('--unscramble', help='Path to the file containing a scrambled sentence')
    
    args = parser.parse_args()

    # Load the model
    model = NGram.load_model(args.model)

    if args.evaluate:
        evaluate_model(model, args.evaluate)
    
    if args.unscramble:
        unscramble(model, args.unscramble)
