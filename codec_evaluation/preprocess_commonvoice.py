"""Dataset download: https://commonvoice.mozilla.org/zh-CN/datasets Common Voice Corpus 18.0"""
import os
import pandas as pd
from datasets import Dataset, load_from_disk
import argparse

def create_arrow_from_tsv(meta_dir, output_dir, jsonl_output_dir=None):
    """
    Create an Arrow dataset from a TSV file
    
    params:
        meta_dir: The directory where the TSV file is located
        output_dir: Arrow data set saving directory
        jsonl_output_dir: JSONL file storage directory (optional)
    arrow_file:
        text: Audio text
        audio_path: Audio path
    """
    tsv_path = os.path.join(meta_dir, f'validated.tsv')
    
    # Check if the TSV file exists
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file does not exist: {tsv_path}")
    
    # Read TSV file and specify column names
    need_columns = ['sentence', 'path']
    
    try:
        df = pd.read_csv(
            filepath_or_buffer=tsv_path,
            sep='\t',
            usecols=need_columns,
            header=0,
            low_memory=False
        )
    except Exception as e:
        raise Exception(f"Failed to read TSV file: {e}")
    
    # Processing sentence columns
    df['text'] = df['sentence'].fillna('').astype(str)
    
    # Build audio path
    audio_base_path = "common_voice/cv-corpus-18.0-2024-06-14/en/clips/"
    df['audio_path'] = audio_base_path + df['path']

    # Create a dataset, keeping only the columns you need
    dataset = Dataset.from_pandas(df[['text', 'audio_path']])
    
    # Save in Arrow format
    output_path = os.path.join(output_dir, f'commonvoice_dataset')
    dataset.save_to_disk(output_path)
    print(f"Successfully saved Arrow dataset to: {output_path}")
    
    # Save in JSONL format (if output directory is specified)
    if jsonl_output_dir:
        os.makedirs(jsonl_output_dir, exist_ok=True)
        jsonl_path = os.path.join(jsonl_output_dir, f'commonvoice_dataset.jsonl')
        dataset.to_json(jsonl_path)
        print(f"Successfully saved JSONL dataset to: {jsonl_path}")
    
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Commonvoice Arrow datasets from TSV files.")
    parser.add_argument("--meta_dir", type=str, required= True, help="Directory containing TSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save Arrow dataset.")
    parser.add_argument("--jsonl_output_dir", type=str, default=None, help="Directory to save JSONL dataset (optional).")

    args = parser.parse_args()
    dataset = create_arrow_from_tsv(args.meta_dir, args.output_dir, args.jsonl_output_dir)

    # Test loading the dataset
    dataset_loaded = load_from_disk(os.path.join(args.output_dir, f'commonvoice_dataset'))
    print(f"Dataset structure: \n{dataset_loaded}")
    print("\n The first three data contents:")
    for example in dataset_loaded.select(range(3)):
        print(f"\n{example}")

