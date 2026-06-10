import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# Add project root to path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from utils.tokenizer import SimpleTokenizer
from utils.dataset import create_dataloaders
from models.emotional_caption_model import EmotionalCaptionModel


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training"
    )

    for batch in progress_bar:

        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        emotion_ids = batch["emotion_id"].to(device)

        # Teacher Forcing
        input_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        optimizer.zero_grad()

        outputs = model(
            images,
            input_captions,
            emotion_ids
        )

        loss = criterion(
            outputs.reshape(
                -1,
                outputs.size(-1)
            ),
            target_captions.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    for batch in dataloader:

        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        emotion_ids = batch["emotion_id"].to(device)

        input_captions = captions[:, :-1]
        target_captions = captions[:, 1:]

        outputs = model(
            images,
            input_captions,
            emotion_ids
        )

        loss = criterion(
            outputs.reshape(
                -1,
                outputs.size(-1)
            ),
            target_captions.reshape(-1)
        )

        running_loss += loss.item()

    return running_loss / len(dataloader)

def load_partial_checkpoint(model, checkpoint_path, device):

    if not os.path.exists(checkpoint_path):
        print("No baseline checkpoint found")
        return

    print("\nLoading compatible baseline weights...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    pretrained_dict = checkpoint["model_state_dict"]

    model_dict = model.state_dict()

    compatible_dict = {}

    for k, v in pretrained_dict.items():

        if (
            k in model_dict
            and model_dict[k].shape == v.shape
        ):
            compatible_dict[k] = v

    model_dict.update(compatible_dict)

    model.load_state_dict(model_dict)

    print(
        f"Loaded {len(compatible_dict)} compatible layers"
    )

def main():

    print("\n" + "=" * 60)
    print("EMOTIONAL IMAGE CAPTIONING TRAINING")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing Device: {device}")

    # --------------------------------------------------
    # Load Tokenizer
    # --------------------------------------------------

    tokenizer = SimpleTokenizer()

    tokenizer.load(
        "utils/tokenizer_emotional.pkl"
    )

    vocab_size = len(
        tokenizer.word2idx
    )

    print(
        f"Vocabulary Size: {vocab_size}"
    )

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------

    train_loader, val_loader = create_dataloaders(
        train_csv="data/emotional/train_small.csv",
        val_csv="data/emotional/val.csv",
        tokenizer=tokenizer,
        batch_size=4,
        num_workers=0
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = EmotionalCaptionModel(
    vocab_size=vocab_size,
    embed_dim=384
    ).to(device)

    total_params = sum(
    p.numel()
    for p in model.parameters()
    )

    trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
        )

    print(
    f"Trainable Parameters: {trainable_params:,}"
    )

    print(
    f"Total Parameters: {total_params:,}"
    )
    load_partial_checkpoint(
    model,
    "checkpoints/best_model.pth",
    device
    )   

    # --------------------------------------------------
    # Optimizer & Loss
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    # --------------------------------------------------
    # Training Config
    # --------------------------------------------------

    num_epochs = 1

    best_val_loss = float("inf")

    Path("checkpoints").mkdir(
        exist_ok=True
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(
        1,
        num_epochs + 1
    ):

        print(
            f"\nEpoch [{epoch}/{num_epochs}]"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                "checkpoints/emotional_model.pth"
            )

            print("Best model saved")

    print("\nTraining completed successfully")


if __name__ == "__main__":
    main()