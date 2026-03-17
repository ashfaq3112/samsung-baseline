import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Adds info so the model knows the order of words (1st, 2nd, 3rd...)"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Use sine/cosine waves to mark positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # Add the position info to the word embeddings
        return x + self.pe[:, :x.size(1)]

class CaptionDecoder(nn.Module):
    """BRAIN: The central processor"""
    
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, num_layers=6, 
                 dropout=0.1, max_seq_len=30):
        super().__init__()
        
        print("🔧 Initializing BRAIN (Caption Decoder)...")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 1. Word Embedding: Converts word IDs (like 452) into vectors
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Position Encoding: Adds order info
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # 3. Transformer Decoder: The smart part that learns language
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads, # 8 parallel "attention heads" thinking at once
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 4. Output Layer: Converts vectors back to word probabilities
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✓ Brain built with {num_layers} layers and {num_heads} heads")
    
    def forward(self, visual_features, captions, caption_mask=None):
        """
        Input: 
            visual_features = What the EYE saw
            captions = What the words are so far
        Output:
            prediction = What the next word should be
        """
        # Embed the text
        caption_embed = self.word_embed(captions)
        caption_embed = self.pos_encoding(caption_embed)
        caption_embed = self.dropout(caption_embed)
        
        # Create a "Causal Mask" 
        # (Prevents the model from cheating by reading the end of the sentence)
        seq_len = captions.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(captions.device)
        
        # The Brain thinks!
        # It looks at the text (tgt) AND the image (memory) at the same time
        decoder_out = self.decoder(
            tgt=caption_embed,
            memory=visual_features, # <--- THIS IS WHERE THE EYE CONNECTS
            tgt_mask=causal_mask,
            tgt_key_padding_mask=caption_mask
        )
        
        # Predict the next word
        logits = self.output_proj(decoder_out)
        
        return logits
    
    def generate(self, visual_features, max_length=30, start_token=1, end_token=2):
        """Used during Testing: Generates a caption word by word"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Start with just the <START> token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Ask the brain for the next word
            logits = self.forward(visual_features, generated)
            
            # Pick the word with the highest score
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Add it to the sentence
            generated = torch.cat([generated, next_token], dim=1)
            
            # If we hit <END>, stop (in a real loop we handle batch stopping, here we simplify)
            if (next_token == end_token).all():
                break
        
        return generated