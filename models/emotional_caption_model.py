import torch
import torch.nn as nn

from eye.vision_encoder import VisionEncoder
from brain.caption_decoder import CaptionDecoder
from heart.emotional_embedding import EmotionEmbedding
from heart.cross_attention import HeartCrossAttention


class EmotionalCaptionModel(nn.Module):
    """
    Eye → Heart → Brain Architecture

    Eye:
        MobileViT Vision Encoder

    Heart:
        Emotion Embedding
        Cross Attention

    Brain:
        Transformer Caption Decoder
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=384,
        num_emotions=4
    ):
        super().__init__()

        print("\nInitializing Emotional Caption Model...")

        # ------------------------
        # EYE
        # ------------------------
        self.eye = VisionEncoder(
            embed_dim=embed_dim,
            pretrained=True
        )

        # ------------------------
        # HEART
        # ------------------------
        self.heart_embed = EmotionEmbedding(
            num_emotions=num_emotions,
            embed_dim=embed_dim
        )

        self.heart_attention = HeartCrossAttention(
            embed_dim=embed_dim,
            num_heads=8
        )

        # ------------------------
        # BRAIN
        # ------------------------
        self.brain = CaptionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        print("✓ Eye initialized")
        print("✓ Heart initialized")
        print("✓ Brain initialized")

    def forward(
        self,
        images,
        captions,
        emotion_ids
    ):
        """
        images      : [B,3,224,224]
        captions    : [B,seq_len]
        emotion_ids : [B]

        Returns:
            logits : [B,seq_len,vocab_size]
        """

        # ==========================
        # EYE
        # ==========================

        visual_features = self.eye(images)

        # visual_features
        # [B,196,384]

        # ==========================
        # HEART
        # ==========================

        emotion_features = self.heart_embed(
            emotion_ids
        )

        # [B,384]

        emotion_features = emotion_features.unsqueeze(1)

        # [B,1,384]

        emotion_aware_features = self.heart_attention(
            visual_features,
            emotion_features
        )

        # [B,1,384]

        # ==========================
        # FUSION
        # ==========================

        memory = torch.cat(
            [
                visual_features,
                emotion_aware_features
            ],
            dim=1
        )

        # [B,197,384]

        # ==========================
        # BRAIN
        # ==========================

        logits = self.brain(
            memory,
            captions
        )

        return logits

    @torch.no_grad()
    def generate_caption(
        self,
        images,
        emotion_ids,
        max_length=30
    ):
        """
        Generate captions during inference

        images      : [B,3,224,224]
        emotion_ids : [B]
        """

        visual_features = self.eye(images)

        emotion_features = self.heart_embed(
            emotion_ids
        )

        emotion_features = emotion_features.unsqueeze(1)

        emotion_aware_features = self.heart_attention(
            visual_features,
            emotion_features
        )

        memory = torch.cat(
            [
                visual_features,
                emotion_aware_features
            ],
            dim=1
        )

        generated_tokens = self.brain.generate(
            memory,
            max_length=max_length
        )

        return generated_tokens