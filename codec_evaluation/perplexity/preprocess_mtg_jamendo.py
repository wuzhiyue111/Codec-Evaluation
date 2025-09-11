"""Dataset download: https://mtg.github.io/mtg-jamendo-dataset/"""
import os
import pandas as pd
import argparse
from datasets import Dataset, load_from_disk, concatenate_datasets

def create_arrow_from_tsv(meta_dir, task, split, output_dir, keep_split=True, save_separate=True):
    """
    Read a TSV split file and return a Dataset object with TAGS, audio_path, optional split
    Save to structured directories if save_separate is True
    1、Separate dataset generation for downstream probes:
          python preprocess_mtg_jamendo.py     
          --meta_dir /sdb/data1/music/mix_music/mtg-jamendo-dataset     
          --output_dir /path/to/your/dataset    
          --splits train validation test     
          --task genre 
    2、The merged dataset is generated for ppl training:
          python preprocess_mtg_jamendo.py     
          --meta_dir /sdb/data1/music/mix_music/mtg-jamendo-dataset     
          --output_dir /path/to/your/dataset     
          --splits train validation test     
          --task genre
          --merge
    """
    tsv_path = os.path.join(meta_dir, f"data/splits/split-0/autotagging_{task}-{split}.tsv")
    
    data = []
    with open(tsv_path, 'r') as f:
        next(f)  
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                print(f"Warning: Line {len(data)+2} in {split} does not have enough fields, skipping.")
                continue
            tags = ' '.join(parts[5:])
            record = {'PATH': parts[3], 'TAGS': tags}
            data.append(record)
    
    df = pd.DataFrame(data)
    df['audio_path'] = "mtg-jamendo-dataset/downloaded_dataset/raw_30s_audio_low/" + df['PATH'].str.replace('.mp3', '.low.mp3')

    if keep_split:
        df['split'] = split

    columns = ['TAGS', 'audio_path']
    if keep_split:
        columns.append('split')
    df = df[columns]
    
    dataset = Dataset.from_pandas(df)
    
    if save_separate:
        # Build target directory：<output_dir>/MTG/MTG_dataset/MTG{task}_dataset/<split>/
        target_dir = os.path.join(output_dir, "MTG", "MTG_dataset", f"MTG{task.capitalize()}_dataset")
        os.makedirs(target_dir, exist_ok=True)
        
        # Saving Arrow datasets and JSONL
        dataset.save_to_disk(os.path.join(target_dir, f"MTG{task.capitalize()}_{split}_dataset"))
        dataset.to_json(os.path.join(target_dir, f"MTG{task.capitalize()}_{split}_dataset.jsonl"))
        print(f"Saved {task}-{split} dataset to: {target_dir}, {len(dataset)} samples.")
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MTG Arrow datasets from TSV files.")
    parser.add_argument("--meta_dir", 
                        type=str, 
                        required=True, 
                        help="Path to MTG meta directory")
    parser.add_argument("--output_dir", 
                        type=str, 
                        required=True, 
                        help="Path to save processed dataset")
    parser.add_argument("--splits", 
                        type=str, 
                        nargs="+", 
                        default=["train", "validation", "test"],
                        help="Which splits to process, default: train validation test")
    parser.add_argument("--merge", 
                        action="store_true", 
                        help="Whether to merge all splits into one dataset")
    parser.add_argument("--task", 
                        type=str, 
                        default="genre", 
                        help="Task type: genre, moodtheme, instrument, top50tags ...")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_datasets = []
    for split in args.splits:
        dataset = create_arrow_from_tsv(
            args.meta_dir, args.task, split, args.output_dir,
            keep_split=args.merge,
            save_separate=not args.merge
        )
        all_datasets.append(dataset)

    if args.merge:
        merged_dataset = concatenate_datasets(all_datasets)
        
        merged_dir = os.path.join(args.output_dir, "MTG", "MTG_dataset", f"MTG{args.task.capitalize()}_dataset")
        os.makedirs(merged_dir, exist_ok=True)
        
        merged_dataset.save_to_disk(os.path.join(merged_dir, "MTG_full_dataset"))
        merged_dataset.to_json(os.path.join(merged_dir, "MTG_full_dataset.jsonl"))
        
        print(f"\n Successfully saved merged dataset to: {merged_dir}, {len(merged_dataset)} samples.")
        
        # test merged dataset
        print("\nTesting merged dataset:")
        dataset_loaded = load_from_disk(os.path.join(merged_dir, "MTG_full_dataset"))
        print(f"dataset structure:\n{dataset_loaded}")
        print("first three data contents:")
        for example in dataset_loaded.select(range(3)):
            print(example)
    else:
        # test split dataset
        print("\nTesting individual splits:")
        for split in args.splits:
            split_dir = os.path.join(args.output_dir, "MTG", "MTG_dataset", f"MTG{args.task.capitalize()}_dataset", f"MTG{args.task.capitalize()}_{split}_dataset")
            dataset_loaded = load_from_disk(split_dir)
            print(f"\n{split} dataset structure:\n{dataset_loaded}")
            print(f"{split} first three data contents:")
            for example in dataset_loaded.select(range(3)):
                print(example)
