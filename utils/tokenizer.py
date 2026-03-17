from collections import Counter
import pickle
import re

class SimpleTokenizer:
    """Simple word-level tokenizer for captions"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        
        # Special tokens required by AI
        self.pad_token = '<PAD>'   # Used to fill empty space
        self.start_token = '<START>' # Tells model to start generating
        self.end_token = '<END>'     # Tells model it's done
        self.unk_token = '<UNK>'     # For words we don't know
        
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3
        
    def build_vocab(self, captions):
        """ učen Words from the Training Data"""
        print(f"\n📚 Building vocabulary (max size: {self.vocab_size})...")
        
        # Count how often every word appears
        word_freq = Counter()
        for caption in captions:
            words = self._tokenize(caption)
            word_freq.update(words)
        
        print(f"  • Total unique words found: {len(word_freq)}")
        
        # Assign IDs to special tokens
        self.word2idx = {
            self.pad_token: self.pad_idx,
            self.start_token: self.start_idx,
            self.end_token: self.end_idx,
            self.unk_token: self.unk_idx
        }
        
        # Add the most common words to our dictionary
        most_common = word_freq.most_common(self.vocab_size - 4)
        for idx, (word, freq) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
        
        # Create the reverse map (ID -> Word)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"  ✓ Vocabulary built with {len(self.word2idx)} words")
        
    def _tokenize(self, text):
        """Clean and split text"""
        text = str(text).lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def encode(self, caption, max_length=30):
        """Text -> Numbers"""
        words = self._tokenize(caption)
        
        # Start with <START>
        tokens = [self.start_idx]
        # Convert words to IDs
        tokens.extend([self.word2idx.get(w, self.unk_idx) for w in words])
        # End with <END>
        tokens.append(self.end_idx)
        
        # Pad if too short, Truncate if too long
        if len(tokens) < max_length:
            tokens.extend([self.pad_idx] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length-1] + [self.end_idx]
        
        return tokens
    
    def decode(self, token_ids):
        """Numbers -> Text"""
        words = []
        for idx in token_ids:
            if idx == self.end_idx:
                break
            if idx in [self.pad_idx, self.start_idx]:
                continue
            words.append(self.idx2word.get(idx, self.unk_token))
        
        return ' '.join(words)
    
    def save(self, filepath):
        """Save the dictionary to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
        print(f"✓ Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """Load the dictionary from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        