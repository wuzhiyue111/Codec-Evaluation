import os
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.utils import cut_or_pad
from datasets import load_from_disk

logger = RankedLogger(__name__, rank_zero_only=True)

class MELDdataset(Dataset):
    def __init__(
        self,
        sample_rate,
        target_sec,       
        is_mono,
        dataset_path,
        base_audio_dir,
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.base_audio_dir = base_audio_dir

        self.dataset = load_from_disk(dataset_path)

        self.classes = """anger, disgust, fear, joy, neutral, sadness, surprise""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            audio_path = self.dataset[index]["audio_path"]
            full_path = os.path.join(self.base_audio_dir, audio_path)
            logger.error(f"Error loading {full_path}: {e}")
            return None
        
    def get_item(self, index):
        """
        return:
            segments: [1,T] 
            labels: [1]
        """
        record = self.dataset[index]  
        relative_audio_path = record["audio_path"]  
        full_audio_path = os.path.join(self.base_audio_dir, relative_audio_path)  

        segments, pad_mask= self.load_audio(full_audio_path)
        label = torch.tensor([self.class2id[record["Emotion"]]], dtype=torch.int64)
        segments = torch.vstack(segments)

        return {"audio": segments, "labels":  label, "n_segments": len(pad_mask)}
    
    def load_audio(self, audio_path):
        """
        input:
            audio_path: one of audio_file path
        return:
            waveform:[1,T]
        """
        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)

        return waveform, pad_mask
    
    def collate_fn(self, batch):    
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]
        
        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.vstack(label_list).squeeze(1)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }
    
class MELDdataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_args,
            train_audio_dir,
            valid_audio_dir,
            test_audio_dir,
            codec_name,
            train_batch_size=32,
            valid_batch_size=2,
            test_batch_size=16,
            train_num_workers=8,
            valid_num_workers=4,
            test_num_workers=4
        ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_audio_dir = train_audio_dir
        self.valid_audio_dir = valid_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MELDdataset(dataset_path=self.test_audio_dir, **self.dataset_args)
            self.valid_dataset = MELDdataset(dataset_path=self.valid_audio_dir, **self.dataset_args)
        if stage == "val":
            self.valid_dataset = MELDdataset(dataset_path=self.valid_audio_dir, **self.dataset_args)
        if stage == "test":
            self.test_dataset = MELDdataset(dataset_path=self.test_audio_dir, **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.valid_num_workers,
            collate_fn=self.valid_dataset.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            collate_fn=self.test_dataset.collate_fn,
        )
