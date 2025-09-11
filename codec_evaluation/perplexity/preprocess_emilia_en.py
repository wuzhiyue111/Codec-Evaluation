"""Convert the dataset to dataset format and download the dataset: https://opendatalab.com/Amphion/Emilia"""
import os
import pandas as pd
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import argparse

def convert_emilia_to_dataset(json_path: str, output_path: str):
    """Convert Emilia-en dataset from JSONL.GZ files to a Dataset object and save it to disk.
    Args:
        json_path (str): The path to the directory containing the JSONL.GZ files.
        output_path (str): The path to save the Dataset object.
    arrow_file:
        id: unique identifier
        wav: audio file path
        text: audio text
        duration: audio duration
        speaker: speaker ID
        language: language
        dnsmos: DNSMOS score, audio quality assessment metric
    """
    print(f"Dataset processing start:")
    # 1、 Collect all .jsonl.gz files from the specified directory
    jsonl_gz_files = [
        os.path.join(json_path, f) 
        for f in os.listdir(json_path) 
        if f.endswith(".jsonl.gz")
    ]

    if not jsonl_gz_files:
        raise ValueError(f"No .jsonl.gz files found in {json_path}")
    
    # 2、Read and merge all files
    dfs = []
    for gz_file in tqdm(jsonl_gz_files):
        df = pd.read_json(
            gz_file, 
            lines=True, 
            compression='gzip'
        )
        dfs.append(df)
        print(f"Read successfully {os.path.basename(gz_file)}({len(df)} rows)")
    
    if not dfs:
        raise ValueError("No dataframes were created from the input files.")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} rows from {len(dfs)} files.")

    # 3. Keep necessary fields (keep only required columns and delete redundant ones)
    required_columns = ['id', 'wav', 'text', 'duration', 'speaker', 'language', 'dnsmos']
    existing_columns = [col for col in required_columns if col in combined_df.columns]

    # 4. Create a dataset directly
    dataset = Dataset.from_pandas(combined_df[existing_columns])

    # 5. Save the dataset and JSONL format
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    dataset.to_json(os.path.join(output_path, "..", "emilia_en_dataset.jsonl"))

    print(f"Dataset saved to {output_path}")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Emilia-en dataset to HuggingFace Dataset format.")
    parser.add_argument("--json_path", 
                        type=str, 
                        required=True, 
                        help="Path to the directory containing JSONL.GZ files")
    parser.add_argument("--output_path", 
                        type=str, 
                        required=True, 
                        help="Path to save processed dataset")

    args = parser.parse_args()

    dataset = convert_emilia_to_dataset(args.json_path, args.output_path)
    print("Dataset conversion completed.\n")

    # Test loading the dataset
    dataset_loaded = load_from_disk(args.output_path)
    print(f"Dataset structure: \n{dataset_loaded}")
    print("\n The first three data contents:")
    for example in dataset_loaded.select(range(3)):
        print(f"\n{example}")
