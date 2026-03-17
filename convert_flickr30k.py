import pandas as pd
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
# Windows path to where you put the raw files
RAW_CSV_PATH = "data/raw/flickr30k/results.csv"
OUTPUT_FOLDER = "data/heart_annotations"

# We want relative paths so it works on Colab later
# (e.g., data/raw/flickr30k/flickr30k_images/1000092795.jpg)
IMAGE_ROOT_DIR = "data/raw/flickr30k/flickr30k_images"

def convert_data():
    print("🔧 Starting Flickr30k Conversion...")
    
    # 1. Load the messy CSV
    # Flickr30k uses '|' as a separator, not commas
    try:
        df = pd.read_csv(RAW_CSV_PATH, sep='|')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Clean Column Names (Remove extra spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # 3. Rename columns to match your Training Code
    # Usually Flickr30k has 'image_name' and ' comment'
    df = df.rename(columns={'image_name': 'image_path', 'comment': 'caption', ' comment': 'caption'})
    
    # 4. Clean the Data
    print("   Cleaning paths and captions...")
    # Add the folder path to the filename
    df['image_path'] = df['image_path'].apply(lambda x: f"{IMAGE_ROOT_DIR}/{str(x).strip()}")
    # Remove rows with missing captions
    df = df.dropna(subset=['caption'])
    # Clean caption text
    df['caption'] = df['caption'].astype(str).str.strip()
    
    # 5. Add Dummy Emotion (Your code requires this column)
    # Since Flickr30k doesn't have emotions, we set it to 'neutral'
    df['emotion'] = 'neutral'
    
    # 6. Split Train/Val (97% Train, 3% Val)
    # 30k images is a lot, so 1000 validation images is enough
    unique_images = df['image_path'].unique()
    random.shuffle(unique_images)
    
    val_imgs = unique_images[:1000]
    train_imgs = unique_images[1000:]
    
    val_df = df[df['image_path'].isin(val_imgs)]
    train_df = df[df['image_path'].isin(train_imgs)]
    
    # 7. Save
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    train_df.to_csv(f"{OUTPUT_FOLDER}/train_emotions.csv", index=False)
    val_df.to_csv(f"{OUTPUT_FOLDER}/val_emotions.csv", index=False)
    
    print(f"✅ Success!")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples:   {len(val_df)}")
    print(f"   Files saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    convert_data()