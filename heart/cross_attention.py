import torch.nn as nn

class HeartCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        num_heads=8
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(
        self,
        visual_features,
        emotion_features
    ):

        attended_features, attn_weights = self.cross_attn(
            query=emotion_features,
            key=visual_features,
            value=visual_features
        )

        return attended_features