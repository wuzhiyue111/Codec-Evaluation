"""Dataset download: https://commonvoice.mozilla.org/zh-CN/datasets Common Voice Corpus 18.0"""
import os
import pandas as pd
from datasets import Dataset, load_from_disk
import argparse

def create_arrow_from_tsvs(train_tsv, dev_tsv, test_tsv, output_dir, jsonl_output_dir=None):
    """
    Create Arrow datasets from specified TSV files
    params:
        train_tsv: Path to train TSV file
        dev_tsv: Path to dev TSV file
        test_tsv: Path to test TSV file
        output_dir: Arrow data set saving directory
        jsonl_output_dir: JSONL file storage directory (optional)
    arrow_file:
        text: Audio text
        audio_path: Audio path
    """
    splits = {
        'train': train_tsv,
        'validation': dev_tsv,
        'test': test_tsv
    }
    audio_base_path = "common_voice/cv-corpus-18.0-2024-06-14/en/clips/"

    parent_dir = os.path.join(output_dir, "Commonvoice_dataset")
    os.makedirs(parent_dir, exist_ok=True)

    for split, tsv_path in splits.items():
        if not tsv_path or not os.path.exists(tsv_path):
            print(f"Warning: {tsv_path} does not exist, skipping {split}.")
            continue

        need_columns = ['sentence', 'path']
        df = pd.read_csv(tsv_path, sep='\t', usecols=need_columns, header=0, low_memory=False)
        df['text'] = df['sentence'].fillna('').astype(str)
        df['audio_path'] = audio_base_path + df['path']

        dataset = Dataset.from_pandas(df[['text', 'audio_path']])

        # Compare the number of valid rows in the original TSV with the number of rows after saving to Arrow
        tsv_count = len(df)
        arrow_count = len(dataset)
        if tsv_count != arrow_count:
            print(
                f"[CHECK] {split}: TSV rows={tsv_count}, Arrow rows={arrow_count} Different quantities!"
            )
        else:
            print(
                f"[CHECK] {split}: TSV rows={tsv_count}, Arrow rows={arrow_count} Same quantities!"
            )

        # Save Arrow
        split_output_path = os.path.join(parent_dir, f'commonvoice_{split}_dataset')
        dataset.save_to_disk(split_output_path, num_proc=16)
        print(f"Saved {split} Arrow dataset to: {split_output_path}")

        # Save JSONL
        if jsonl_output_dir:
            jsonl_parent = os.path.join(jsonl_output_dir, "Commonvoice_dataset")
            os.makedirs(jsonl_parent, exist_ok=True)
            jsonl_path = os.path.join(jsonl_parent, f"commonvoice_{split}_dataset.jsonl")
            dataset.to_json(jsonl_path)
            print(f"Saved {split} JSONL dataset to: {jsonl_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Commonvoice Arrow datasets from TSV files.")
    parser.add_argument("--train_tsv", 
                        type=str, 
                        required=True, 
                        help="Path to train TSV file")
    parser.add_argument("--dev_tsv", 
                        type=str, 
                        required=True, 
                        help="Path to dev TSV file")
    parser.add_argument("--test_tsv", 
                        type=str, 
                        required=True, 
                        help="Path to test TSV file")
    parser.add_argument("--output_dir", 
                        type=str, 
                        required=True, 
                        help="Directory to save Arrow datasets")
    parser.add_argument("--jsonl_output_dir", 
                        type=str, 
                        default=None, 
                        help="Directory to save JSONL datasets (optional)")
    args = parser.parse_args()

    create_arrow_from_tsvs(args.train_tsv, args.dev_tsv, args.test_tsv, args.output_dir, args.jsonl_output_dir)

    # Test loading the dataset
    splits = ['train', 'validation', 'test']
    parent_dir = os.path.join(args.output_dir, "Commonvoice_dataset") 

    for split in splits:
        split_path = os.path.join(parent_dir, f'commonvoice_{split}_dataset')
        
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping {split}.")
            continue

        dataset_loaded = load_from_disk(split_path)
        print(f"\n=== {split.upper()} dataset structure ===")
        print(dataset_loaded)
        
        print(f"\nFirst three examples of {split} dataset:")
        for example in dataset_loaded.select(range(min(3, len(dataset_loaded)))):
            print(example)


