import torch.nn as nn

class EmotionEmbedding(nn.Module):

    def __init__(
        self,
        num_emotions=4,
        embed_dim=384
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_emotions,
            embed_dim
        )

    def forward(self, emotion_ids):

        return self.embedding(emotion_ids)