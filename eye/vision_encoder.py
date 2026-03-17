import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisionEncoder(nn.Module):
    """THE EYE: Looks at the image and extracts features"""
    
    def __init__(self, embed_dim=384, num_patches=196, pretrained=True):
        # We are keeping num_patches = 196 (14x14 grid)
        super().__init__()
        
        print("🔧 Initializing EYE (Vision Encoder)...")
        
        # 1. Load Pretrained MobileViT
        self.backbone = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=0)
        
        # 2. Adapter Layer
        self.projection = nn.Linear(640, embed_dim)
        
        # 3. Position Embeddings (For 196 patches)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        print(f"✓ MobileViT-S loaded (pretrained={pretrained})")
        print(f"✓ Backbone dim: 640 -> Embed dim: {embed_dim} (Patches: {num_patches})")
        
    def forward(self, x):
        """
        Input: Image batch [Batch, 3, 224, 224]
        Output: Visual Features [Batch, 196, 384]
        """
        # 1. Extract raw features from MobileViT
        # Shape comes out as [Batch, 640, 7, 7]
        features = self.backbone.forward_features(x)
        
        # 2. Force Upscale from 7x7 (49) to 14x14 (196)
        # This fixes your error while keeping 196 patches!
        features = F.interpolate(features, size=(14, 14), mode='bilinear', align_corners=False)
        
        # 3. Reshape: [Batch, 640, 14, 14] -> [Batch, 196, 640]
        features = features.flatten(2).transpose(1, 2)
        
        # 4. Project: [Batch, 196, 640] -> [Batch, 196, 384]
        features = self.projection(features)
        
        # 5. Add Position Info
        # Now 196 matches 196!
        features = features + self.pos_embed
        
        return features