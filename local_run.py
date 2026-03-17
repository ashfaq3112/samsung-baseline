import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import pickle
import os
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. MODEL ARCHITECTURE (Matches your dir)
# ==========================================
# These classes are reconstructed based on your 'eye' and 'brain' modules
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

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
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=1536)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, features, captions):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        embeddings = self.word_embed(captions)
        embeddings = self.pos_encoding(embeddings)
        output = self.decoder(tgt=embeddings, memory=features, tgt_mask=tgt_mask)
        return self.output_proj(output)

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim=384, vocab_size=5000, num_layers=6, num_heads=8):
        super().__init__()
        self.eye = VisionEncoder(embed_dim)
        self.brain = CaptionDecoder(vocab_size, embed_dim, num_layers, num_heads)

    def forward(self, images, captions):
        features = self.eye(images)
        outputs = self.brain(features, captions)
        return outputs

# ==========================================
# 2. LOAD LOCAL FILES
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# Load Tokenizer dictionary
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    # Handle the dict format found in your Kaggle run
    stoi = tokenizer['stoi'] if isinstance(tokenizer, dict) else tokenizer.stoi
    itos = tokenizer['itos'] if isinstance(tokenizer, dict) else tokenizer.itos

# Initialize and Load Model Weights
model = ImageCaptioningModel(vocab_size=5000).to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
model.eval()
print("✅ Model and Tokenizer loaded successfully!")

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.eye(img)
        # Use Beam Search for better local results (k=3)
        k = 3
        sequences = [(0.0, [stoi['<start>']])]
        
        for _ in range(25):
            all_candidates = []
            for score, seq in sequences:
                if seq[-1] == stoi['<end>']:
                    all_candidates.append((score, seq))
                    continue
                
                trg = torch.tensor([seq]).to(device)
                emb = model.brain.word_embed(trg)
                emb = model.brain.pos_encoding(emb)
                mask = torch.triu(torch.ones(trg.size(1), trg.size(1)), 1).bool().to(device)
                
                out = model.brain.decoder(tgt=emb, memory=features, tgt_mask=mask)
                probs = F.log_softmax(model.brain.output_proj(out)[:, -1, :], dim=-1)
                
                topk_probs, topk_ids = probs.topk(k, dim=-1)
                for i in range(k):
                    all_candidates.append((score + topk_probs[0][i].item(), seq + [topk_ids[0][i].item()]))
            
            sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:k]
            if all(s[1][-1] == stoi['<end>'] for s in sequences): break
            
    best_seq = sequences[0][1]
    caption = [itos[idx] for idx in best_seq if idx not in [stoi['<start>'], stoi['<end>']]]
    return " ".join(caption)

# ==========================================
# 4. EXECUTION
# ==========================================
# Change 'girl.jpg' to any image in your folder (like 'aad.jpg' or 'image.png')
test_image = "girl.jpg" 

if os.path.exists(test_image):
    print(f"\n🖼️  Image: {test_image}")
    print(f"🤖 Caption: {predict(test_image)}")
else:
    print(f"❌ Could not find {test_image} in your folder.")