import os 
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.utils import cut_or_pad
from codec_evaluation.utils.logger import RankedLogger
from datasets import load_from_disk

logger = RankedLogger(__name__, rank_zero_only=True)

class VocalSoundDataset(Dataset):
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
        self.dataset_path = dataset_path
        self.base_audio_dir = base_audio_dir
    
        self.dataset = load_from_disk(dataset_path)

        self.classes = """/m/01j3sz, /m/07plz5l, /m/01b_21, /m/0dl9sf8, /m/01hsr_, /m/07ppn3j""".split(", ")
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
            segments: [n,T]
            labels: [n]
        """
        record = self.dataset[index]
        relative_audio_path = record["audio_path"] 
        full_audio_path = os.path.join(self.base_audio_dir, relative_audio_path) 

        segments, pad_mask = self.load_audio(full_audio_path)
        label = self.class2id[record["labels"]] 

        labels = []
        labels.extend([label] * len(pad_mask))

        labels = torch.tensor(labels)
        segments = torch.vstack(segments)

        return {"audio": segments, "labels": labels, "n_segments": len(pad_mask)}
    
    def load_audio(
        self,
        audio_file,
    ):
        """
        input:
            audio_file:one of audio_file path
        return:
            waveform:[n,T]
        """
        waveform, _ = torchaudio.load(audio_file)

        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)
        
        return waveform, pad_mask
    
    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]
        
        audio_tensor = torch.vstack(audio_list)
        label_tensor = torch.cat(label_list, dim=0)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }

class VocalSoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_args, 
        train_audio_dir,
        val_audio_dir,
        test_audio_dir,
        codec_name,
        train_batch_size=32,
        val_batch_size=2,
        test_batch_size=16,
        train_num_workers=8,
        val_num_workers=4,
        test_num_workers=4
    ):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_audio_dir = train_audio_dir
        self.val_audio_dir = val_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = VocalSoundDataset(dataset_path=self.test_audio_dir, **self.dataset_args)
            self.val_dataset = VocalSoundDataset(dataset_path=self.val_audio_dir, **self.dataset_args)
        if stage == "val":
            self.val_dataset = VocalSoundDataset(dataset_path=self.val_audio_dir, **self.dataset_args)
        if stage == "test":
            self.test_dataset = VocalSoundDataset(dataset_path=self.test_audio_dir,**self.dataset_args)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.train_num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.val_num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )

