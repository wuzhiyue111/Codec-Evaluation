"""Dataset download: https://github.com/a43992899/MARBLE"""
import os
import re
import pandas as pd
from datasets import Dataset
def create_arrow_from_tsv(meta_dir, split, output_dir):

    tsv_path = os.path.join(meta_dir, f"data/splits/split-0/autotagging_genre-{split}.tsv")
    
    # manually read TSV and merge redundant fields
    data = []
    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')  # Read header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                print(f"Warning: Line {len(data)+2} does not have enough fields, skipping.")
                continue
            
            # Merge all fields after the fifth column into the sixth column (TAGS), concatenating them with spaces.
            tags = ' '.join(parts[5:])  # Use spaces to connect tags
            record = {
                'TRACK_ID': parts[0],
                'ARTIST_ID': parts[1],
                'ALBUM_ID': parts[2],
                'PATH': parts[3],
                'DURATION': parts[4],
                'TAGS': tags, 
            }
            data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Extract the number from PATH as id
    df['id'] = df['PATH'].apply(lambda x: re.search(r'(\d+)\.mp3', x).group(1))
    
    # Handle audio path
    df['audio_path'] = "MTG/audio-low/" + df['PATH'].str.replace('.mp3', '.low.mp3')
    
    # Add a split column
    df['split'] = split
    
    # Create a dataset
    dataset = Dataset.from_pandas(df)
    
    # Save in Arrow format
    output_path = os.path.join(output_dir, f"MTGGenre_{split}_dataset")
    dataset.save_to_disk(output_path)
    dataset.to_json(os.path.join(output_dir, f"MTGGenre_{split}_dataset.jsonl"))
    
    print(f"Successfully saved {split} dataset to: {output_path}")
    return dataset

meta_dir = "/sdb/data1/music/mix_music/marble_dataset/data/MTG/mtg-jamendo-dataset/"
output_dir = "/path/to/your/Codec-Evaluation/codec_evaluation/convert_dataset_arrow/MTG/MTG_dataset/MTGGenre_dataset"
os.makedirs(output_dir, exist_ok=True)

for split in ["train", "validation", "test"]:
    create_arrow_from_tsv(meta_dir, split, output_dir)