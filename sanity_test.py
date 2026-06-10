import torch
from models.emotional_caption_model import EmotionalCaptionModel

model = EmotionalCaptionModel(
    vocab_size=10000
)

images = torch.randn(2,3,224,224)
captions = torch.randint(0,10000,(2,30))
emotion_ids = torch.tensor([1,2])

output = model(
    images,
    captions,
    emotion_ids
)

print(output.shape)