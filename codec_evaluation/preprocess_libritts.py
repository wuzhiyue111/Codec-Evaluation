import os
from datasets import Dataset, Features, Value, load_from_disk
from tqdm import tqdm
from codec_evaluation.utils.utils import find_audios
import argparse

def convert_libritts_to_arrow(
    audio_dir: str,          
    output_path: str,        
    limit: int = -1,         
    verify: bool = True      
):
    """Convert LibriTTS dataset to .arrow format
    Params:
        audio_dir: Audio directories (such as train-clean-100)
        output_path: .arrow output path
        limit: Limit the number of samples processed (-1 means all)
        verify: Verify that the text file exists
    .arrow file contains two columns:
        - audio_path: The relative path of the audio file
        - text: Audio text
    """
    # 1. Find all audio files
    all_audio_paths = find_audios(audio_dir)
    if limit > 0:
        all_audio_paths = all_audio_paths[:limit]
    print(f"{len(all_audio_paths)} audio files found.")
    
    # 2. Build dataset records
    records = []
    for audio_path in tqdm(all_audio_paths, desc="处理中"):
        # Generate the corresponding text file path
        text_path = audio_path.replace(".wav", ".normalized.txt")
        
        # Verify that the text file exists (optional)
        if verify and not os.path.exists(text_path):
            print(f"Warning: Text file does not exist: {text_path}")
            continue
        
        # Read text content
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error: Failed to read text file {text_path}: {e}")
            continue
        
        # Construct relative path (relative to audio_dir)
        relative_audio_path = os.path.relpath(audio_path, audio_dir)
        splicing_path = os.path.join(*audio_dir.split(os.sep)[-2:])
        path = os.path.join(splicing_path, relative_audio_path)
        
        records.append({
            "audio_path": path,  
            "text": text,
        })
    
    features = Features({
        "audio_path": Value("string"),  
        "text": Value("string"),          
    })

    # 3. Create a dataset and save it as .arrow
    dataset = Dataset.from_list(records, features=features)
    output_path = os.path.join(output_path, os.path.basename(audio_dir))
    dataset.save_to_disk(output_path)
    dataset.to_json(os.path.join(os.path.dirname(output_path), f"{os.path.basename(audio_dir)}.jsonl"))
    print(f"Successfully saved {len(records)} records to: {output_path}")
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LibriTTS dataset to HuggingFace Dataset format.")
    parser.add_argument("--audio_dir", 
                        type=str, required=True, help="Audio directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of samples processed (-1 means all)")
    parser.add_argument("--verify", action="store_true", help="Verify that the text file exists")
    args = parser.parse_args()

    dataset = convert_libritts_to_arrow(args.audio_dir, args.output_path, args.limit, args.verify)

    # Test loading the dataset
    dataset_loaded = load_from_disk(os.path.join(args.output_path, os.path.basename(args.audio_dir)))
    print(f"Dataset structure: \n{dataset_loaded}")
    print("\n The first three data contents:")
    for example in dataset_loaded.select(range(3)):
        print(f"\n{example}")

