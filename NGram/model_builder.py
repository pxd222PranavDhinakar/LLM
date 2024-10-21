# model_builder.py
import nltk
import numpy as np
import re
nltk.download('brown')
from nltk.corpus import brown

def ensure_nltk_resources():
  required_resources = ['punkt', 'brown', 'punkt_tab']
  for resource in required_resources:
    try:
      if resource == 'punkt_tab':
        nltk.data.find(f'tokenizers/{resource}/english/')
      else:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
      nltk.download(resource, download_dir='C://Users//caden/nltk_data')

# ensure_nltk_resources()

class NGram():
    def __init__(self, n: int):
        self.n = n
        self.counts = None
        self.vocab = None
    
    def train(self, corpus) -> None:
        corpus: list[str] = [token.lower() for token in corpus if not re.match(r'[^\w\s]', token)]
        self.vocab: list[str] = list(set(corpus))
        self.vocab.append("<UNK>")  # Add unknown token to vocabulary
        self.corpus_ind: list[int] = [self.vocab.index(word) if word in self.vocab else self.vocab.index("<UNK>") for word in corpus]
        self.counts = np.zeros((len(self.vocab),) * self.n, dtype=np.uint16)
        for i in range(len(self.corpus_ind) - self.n + 1):
            self.counts[*self.corpus_ind[i:i+self.n]] += 1
    
    def prob(self, context: list[str], word: str) -> float:
        word_index: int = self.vocab.index(word) if word in self.vocab else self.vocab.index("<UNK>")
        context_indices: list[int] = [self.vocab.index(token) if token in self.vocab else self.vocab.index("<UNK>") for token in context]
        
        counts = np.sum(self.counts, axis=tuple(range(self.n - len(context) - 1))).astype(np.float64) + 1
        context_counts: np.ndarray = np.sum(counts, axis=-1, keepdims=True) + len(self.vocab)
        probs: np.ndarray = np.divide(counts, context_counts, where=context_counts != 0)
    
        return probs[*context_indices, word_index]
    
    def save_model(self, path: str) -> None:
        np.save(path, self.counts)
        np.save(path + '_lexicon', self.vocab)
        print(f"Model saved to {path}" + '.npy')
    
    @staticmethod
    def load_model(path: str):
        counts: np.ndarray = np.load(path + '.npy')
        lexicon: np.ndarray = np.load(path + '_lexicon.npy')
        ngram = NGram(len(counts.shape))
        ngram.counts = counts
        ngram.vocab = lexicon
        print(f"Model loaded from {path}" + '.npy')
        return ngram

'''
class NGram():
  def __init__(self, n: int):
    self.n = n
    self.counts = None
  
  def train(self, corpus) -> None:
    # Take out punctuation, make all lowercase
    corpus: list[str] = [token.lower() for token in corpus if not re.match(r'[^\w\s]', token)]
    
    self.vocab: list[str] = list(set(corpus))
    
    # convert the corpus into a list of token indices
    self.corpus_ind: list[int] = [self.vocab.index(word) for word in corpus]
    
    # Split in to substrings of length n, and count occurances of each word after each context
    self.counts = np.zeros((len(self.vocab),) * self.n, dtype=np.uint16)
    for i in range(len(self.corpus_ind) - self.n + 1):
      self.counts[*self.corpus_ind[i:i+self.n]] += 1
  
  def prob(self, context: list[str], word: str) -> float:
    # Indicize the words
    word_index: int = self.vocab.index(word)
    context_indices: list[int] = [self.vocab.index(token) for token in context]
    
    # Sum over c_k prefix exluded in the context
    counts = np.sum(self.counts, axis=tuple(range(self.n - len(context) - 1)), dtype=np.uint16) + 1
    
    # Normalize the counts into probabilities
    context_counts: np.ndarray = np.sum(counts, axis=-1, keepdims=True) + len(self.vocab)
    probs: float = np.divide(counts, context_counts, out=np.zeros_like(counts), where= context_counts != 0)
  
    return probs[*context_indices, word_index]
  
  def save_model(self, path: str) -> None:
    np.save(path, self.counts)
    np.save(path + '_lexicon', self.vocab)
    print(f"Model saved to {path}" + '.npy')
    
  def load_model(path: str):
    counts: np.ndarray = np.load(path + '.npy')
    lexicon: np.ndarray = np.load(path + '_lexicon.npy')
    ngram = NGram(len(counts.shape))
    ngram.counts = counts
    ngram.vocab = lexicon
    print(f"Model loaded from {path}" + '.npy')
    return ngram
'''

if __name__ == "__main__":
  path = 'model/bigram'
  ngram = NGram(2)
  brown.ensure_loaded()
  print(' '.join(brown.words()[5055:5100]))
  ngram.train(brown.words()[:10000])
  ngram.save_model(path)
  
  
  
