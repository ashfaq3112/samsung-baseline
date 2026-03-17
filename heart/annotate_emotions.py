import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add the project root to the path so we can import the detector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heart.emotion_detector import EmotionDetector

def annotate_split(csv_file, output_file, detector):
    """Annotate emotions for a specific dataset split"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file}")
    print(f"{'='*60}")
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        return
        
    df = pd.read_csv(csv_file)
    unique_images = df['image'].unique()
    
    print(f"Total captions: {len(df)}")
    print(f"Unique images: {len(unique_images)}")
    
    # Process each unique image
    image_emotions = {}
    
    # We use tqdm to show a progress bar
    for img_name in tqdm(unique_images, desc="Detecting emotions"):
        # Find the image path from the dataframe
        img_path = df[df['image'] == img_name].iloc[0]['image_path']
        
        # 1. Detect Face Emotion
        face_result = detector.detect_face_emotion(img_path)
        
        # 2. Detect Scene Emotion (Backup)
        scene_emotion = detector.detect_scene_emotion(img_path)
        
        # Store results
        image_emotions[img_name] = {
            'has_face': face_result['has_face'],
            'face_emotion': face_result['emotion'],
            'face_confidence': face_result['confidence'],
            'scene_emotion': scene_emotion
        }
    
    # Merge emotions back into the captions dataframe
    emotion_data = []
    for _, row in df.iterrows():
        emotions = image_emotions[row['image']]
        emotion_data.append({
            **row.to_dict(),
            **emotions
        })
    
    # Save the new annotated dataset
    output_df = pd.DataFrame(emotion_data)
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    # Print stats
    face_detected = sum(1 for e in image_emotions.values() if e['has_face'])
    print(f"\n📊 Statistics:")
    print(f"  • Images with faces: {face_detected}/{len(unique_images)}")
    print(f"  • Saved to: {output_file}")

def main():
    print("\n" + "="*60)
    print("STEP 3: HEART - Running Emotion Annotation")
    print("="*60)
    
    # Initialize the detector we built
    detector = EmotionDetector()
    
    # Define where to save results
    output_dir = 'data/heart_annotations'
    
    # Process all three splits (Train, Val, Test)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        csv_file = f'data/splits/{split}.csv'
        output_file = f'{output_dir}/{split}_emotions.csv'
        
        annotate_split(csv_file, output_file, detector)
    
    print("\n" + "="*60)
    print("✅ Emotion annotation complete! Proceed to Phase 3.")
    print("="*60)

if __name__ == "__main__":
    main()