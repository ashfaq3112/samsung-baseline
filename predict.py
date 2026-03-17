import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import timm
import math
import os
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "best_model.pth" 
CSV_PATH = "data/heart_annotations/train_emotions.csv"
DEVICE = torch.device('cpu') 
print(f"🖥️  Running on: {DEVICE}")

# ==========================================
# 2. DEFINE THE ARCHITECTURE (EXACT COPY)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=384, num_patches=196):
        super().__init__()
        self.backbone = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
        self.projection = nn.Linear(640, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = F.interpolate(features, size=(14, 14), mode='bilinear', align_corners=False)
        features = features.flatten(2).transpose(1, 2)
        features = self.projection(features)
        features = features + self.pos_embed
        return features

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, num_layers=6):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    # Note: We don't use this generate() for Beam Search, we do it manually below
    def generate(self, visual_features, max_length=30):
        pass 

class BaselineModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.eye = VisionEncoder()
        self.brain = CaptionDecoder(vocab_size)

# ==========================================
# 3. LOAD SYSTEM
# ==========================================

def load_system():
    print("📖 Rebuilding Dictionary...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find CSV at {CSV_PATH}. Check your path!")
        
    df = pd.read_csv(CSV_PATH)
    all_words = " ".join(df['caption'].astype(str)).lower().split()
    from collections import Counter
    word_counts = Counter(all_words)
    
    # STRICT LIMIT 5000
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    idx = 4
    for word, count in word_counts.most_common(5000 - 4):
        vocab[word] = idx
        idx += 1
    
    idx_to_word = {v: k for k, v in vocab.items()}
    print(f"✓ Dictionary Ready: {len(vocab)} words.")

    print(f"⚖️ Loading Model from {MODEL_PATH}...")
    model = BaselineModel(vocab_size=len(vocab)).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("🎉 Model Loaded Successfully!")
        return model, idx_to_word
    else:
        raise FileNotFoundError(f"Cannot find model at {MODEL_PATH}")

# ==========================================
# 4. PREDICT FUNCTION (UPGRADED WITH BEAM SEARCH)
# ==========================================

def predict_caption(model, idx_to_word, image_path, beam_size=3):
    """
    Uses Beam Search to find the best sentence instead of just the next word.
    This fixes repetitive grammar issues.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # 1. Prepare Image
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 2. Get Visual Features (The Eye)
        with torch.no_grad():
            visual_features = model.eye(img_tensor)
        
        # 3. Start Beam Search
        # We start with one candidate: [START]
        # Format: (log_score, list_of_token_ids)
        k = beam_size
        sequences = [(0.0, [1])] 
        
        # 4. Loop word by word
        for _ in range(30): # Max sentence length
            all_candidates = []
            
            # Expand each current candidate sequence
            for score, seq in sequences:
                # If this sequence already ended with <END> (2), keep it as is
                if seq[-1] == 2:
                    all_candidates.append((score, seq))
                    continue
                
                # Prepare input for the model
                input_seq = torch.tensor([seq], dtype=torch.long).to(DEVICE)
                
                # Run the Brain manually
                with torch.no_grad():
                    caption_embed = model.brain.word_embed(input_seq)
                    caption_embed = model.brain.pos_encoding(caption_embed)
                    
                    # Create mask to hide future words
                    seq_len = input_seq.size(1)
                    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(DEVICE)
                    
                    # Decoder Pass
                    out = model.brain.decoder(tgt=caption_embed, memory=visual_features, tgt_mask=mask)
                    logits = model.brain.output_proj(out)
                    
                    # Get probabilities for the NEXT word (last position)
                    probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Select top K next words
                topk_probs, topk_ids = probs.topk(k, dim=-1)
                
                # Create new candidates
                for i in range(k):
                    word_id = topk_ids[0][i].item()
                    word_prob = topk_probs[0][i].item()
                    
                    new_seq = seq + [word_id]
                    new_score = score + word_prob # Add log probability
                    
                    all_candidates.append((new_score, new_seq))
            
            # 5. Select the best K sequences from all candidates
            ordered = sorted(all_candidates, key=lambda tup: tup[0], reverse=True)
            sequences = ordered[:k]
            
            # Stop if the best sequence is finished
            if sequences[0][1][-1] == 2:
                break
        
        # 6. Convert the winner (sequences[0]) to words
        best_seq = sequences[0][1]
        caption_words = []
        for idx in best_seq:
            if idx == 2: break # Stop at END
            if idx not in [0, 1, 2]: # Skip special tokens
                caption_words.append(idx_to_word.get(idx, '<UNK>'))
        
        return " ".join(caption_words)

    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        model, vocab = load_system()
        
        while True:
            print("\n" + "="*30)
            img_path = input("🖼️  Enter image path (or 'q' to quit): ").strip().strip('"')
            
            if img_path.lower() == 'q':
                break
                
            if os.path.exists(img_path):
                print("Thinking (checking 3 possibilities)...")
                caption = predict_caption(model, vocab, img_path, beam_size=3)
                print(f"🤖 Caption: {caption}")
            else:
                print("❌ File not found. Try again.")
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")