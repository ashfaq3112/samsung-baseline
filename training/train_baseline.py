import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path so we can import our other files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eye.vision_encoder import VisionEncoder
from brain.caption_decoder import CaptionDecoder
from utils.tokenizer import SimpleTokenizer
from utils.dataset import create_dataloaders

class BaselineModel(nn.Module):
    """The Complete Robot: Connects Eye -> Brain"""
    
    def __init__(self, vocab_size, embed_dim=384):
        super().__init__()
        self.eye = VisionEncoder(embed_dim=embed_dim, pretrained=True)
        self.brain = CaptionDecoder(vocab_size=vocab_size, embed_dim=embed_dim)
    
    def forward(self, images, captions, caption_mask=None):
        # 1. Eye looks at image -> Gets Features
        visual_features = self.eye(images)
        # 2. Brain takes Features + Captions -> Predicts Next Word
        logits = self.brain(visual_features, captions, caption_mask)
        return logits
    
    def generate_caption(self, images, max_length=30):
        """For testing later"""
        visual_features = self.eye(images)
        return self.brain.generate(visual_features, max_length=max_length)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Run one full pass of training"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # 1. Move data to GPU/CPU
        images = batch['image'].to(device)
        captions = batch['caption'].to(device)
        
        # 2. Teacher Forcing Setup
        # Input: "<START> A dog is"
        # Target: "A dog is running"
        input_captions = captions[:, :-1]
        target_captions = captions[:, 1:]
        
        # 3. Forward Pass (The Model Guesses)
        optimizer.zero_grad()
        logits = model(images, input_captions)
        
        # 4. Calculate Error (Loss)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_captions.reshape(-1)
        )
        
        # 5. Backward Pass (The Model Learns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def main():
    print("\n" + "="*60)
    print("STEP 7: Start Training")
    print("="*60)
    
    # 1. Setup Device (Use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using Device: {device}")
    
    # 2. Build/Load Tokenizer
    print("📚 Building tokenizer...")
    # We read the captions from the training file we made in Step 3
    train_df = pd.read_csv('data/heart_annotations/train_emotions.csv')
    tokenizer = SimpleTokenizer(vocab_size=5000)
    tokenizer.build_vocab(train_df['caption'].tolist())
    
    # Save it so we can use it for testing later
    Path('checkpoints').mkdir(exist_ok=True)
    tokenizer.save('checkpoints/tokenizer.pkl')
    
    # 3. Create Data Loaders (The Dataset Script you just made)
    print("📦 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_csv='data/heart_annotations/train_emotions.csv',
        val_csv='data/heart_annotations/val_emotions.csv',
        tokenizer=tokenizer,
        batch_size=16, # Smaller batch size to be safe on laptop
        num_workers=0
    )
    
    # 4. Initialize Model
    print("🏗️  Initializing model...")
    model = BaselineModel(vocab_size=len(tokenizer.word2idx), embed_dim=384)
    model = model.to(device)
    
    # 5. Optimizer (The Teacher)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD> tokens
    
    # 6. The Training Loop
    num_epochs = 5  # We'll start with 5 epochs to test
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, f'checkpoints/best_model.pth')
        
        print(f"✅ Epoch {epoch} complete! Loss: {loss:.4f}")

if __name__ == "__main__":
    main()