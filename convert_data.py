import pandas as pd
import os

# Define the folder path
base_path = r'data/raw/flickr8k'
token_file = os.path.join(base_path, 'Flickr8k.token.txt')
output_file = os.path.join(base_path, 'captions.txt')

print(f"📂 Checking folder: {base_path}")

# Check if file exists
if not os.path.exists(token_file):
    print(f"❌ Error: Could not find 'Flickr8k.token.txt'")
    print("   (Windows might be hiding the .txt extension, which is fine)")
    # Try without .txt just in case
    token_file = os.path.join(base_path, 'Flickr8k.token')
    if os.path.exists(token_file):
        print("   Found it without .txt extension!")
    else:
        exit()

print("🔄 Converting text to CSV...")
data = []

try:
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            # The line format is: image_name.jpg#0   Caption Text
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Get the image ID (remove the #0 part)
                image_id = parts[0].split('#')[0]
                caption = parts[1]
                data.append({'image': image_id, 'caption': caption})

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Success! Created 'captions.txt' with {len(df)} captions.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
    