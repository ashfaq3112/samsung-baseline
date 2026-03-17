import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import os

class EmotionCaptionDataset(Dataset):
    """The Bridge: Connects hard drive files to the AI model"""
    
    def __init__(self, csv_file, tokenizer, max_caption_len=30, transform=None):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        
        # Standard Image Transforms (Resize to 224x224 and Normalize)
        # This makes every image look "standard" to the AI
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], # Standard stats for pre-trained models
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Tells the model how many items we have"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """The Handshake: Get one specific data pair (Image + Text)"""
        row = self.data.iloc[idx]
        
        # 1. Load the Image
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # If image fails, return a blank black image (prevent crash)
            image = torch.zeros(3, 224, 224)
        
        # 2. Encode the Caption (Text -> Numbers)
        caption_tokens = self.tokenizer.encode(row['caption'], self.max_caption_len)
        
        # 3. Bundle Emotion Info (Pass it along just in case)
        emotions = {
            'has_face': row['has_face'],
            'face_emotion': row['face_emotion'],
            'scene_emotion': row['scene_emotion']
        }
        
        # Return the package
        return {
            'image': image,
            'caption': torch.tensor(caption_tokens, dtype=torch.long),
            'emotions': emotions,
            'image_id': row['image']
        }

def create_dataloaders(train_csv, val_csv, tokenizer, batch_size=32, num_workers=0):
    """Factory function to build the loaders"""
    
    # Training Augmentation (Randomly flip/crop images to make the AI smarter)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Create the Datasets
    train_dataset = EmotionCaptionDataset(train_csv, tokenizer, transform=train_transform)
    # Validation gets no special augmentation, just standard resize
    val_dataset = EmotionCaptionDataset(val_csv, tokenizer)
    
    # 2. Create the Loaders (The Batch Makers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data so it doesn't memorize order
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Data Loaders ready: {len(train_dataset)} training items, {len(val_dataset)} validation items")
    
    return train_loader, val_loader