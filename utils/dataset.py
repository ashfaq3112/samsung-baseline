import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms


class EmotionCaptionDataset(Dataset):
    """
    Dataset for Emotion-Aware Image Captioning

    CSV Format:
    image_path,emotion,intensity,caption
    """

    def __init__(
        self,
        csv_file,
        tokenizer,
        max_caption_len=30,
        transform=None
    ):
        self.data = pd.read_csv(csv_file)

        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len

        # Emotion Mapping
        self.emotion_map = {
            "neutral": 0,
            "joyful": 1,
            "curious": 2,
            "excited": 3
        }

        # Default Image Transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # ----------------------------------
        # Load Image
        # ----------------------------------
        image_path = row["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

        except Exception as e:
            print(f"Error loading image: {image_path}")
            print(e)

            image = torch.zeros(3, 224, 224)

        # ----------------------------------
        # Caption
        # ----------------------------------
        caption = str(row["caption"])

        caption_tokens = self.tokenizer.encode(
            caption,
            self.max_caption_len
        )

        # ----------------------------------
        # Emotion
        # ----------------------------------
        emotion = str(row["emotion"]).lower()

        emotion_id = self.emotion_map.get(
            emotion,
            self.emotion_map["neutral"]
        )

        # ----------------------------------
        # Intensity
        # ----------------------------------
        intensity = float(row["intensity"])

        return {
            "image": image,

            "caption": torch.tensor(
                caption_tokens,
                dtype=torch.long
            ),

            "emotion_id": torch.tensor(
                emotion_id,
                dtype=torch.long
            ),

            "intensity": torch.tensor(
                intensity,
                dtype=torch.float
            ),

            "image_path": image_path
        }


def create_dataloaders(
    train_csv,
    val_csv,
    tokenizer,
    batch_size=32,
    num_workers=0
):
    """
    Creates train and validation dataloaders
    """

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = EmotionCaptionDataset(
        train_csv,
        tokenizer,
        transform=train_transform
    )

    val_dataset = EmotionCaptionDataset(
        val_csv,
        tokenizer,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(
        f"✓ Data Loaders ready:"
        f" {len(train_dataset)} train samples,"
        f" {len(val_dataset)} val samples"
    )

    return train_loader, val_loader


if __name__ == "__main__":

    
    print("Dataset module loaded successfully.")