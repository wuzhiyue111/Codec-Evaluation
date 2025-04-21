import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torchaudio
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.utils import find_audios, cut_or_pad


class GSdataset(Dataset):
    def __init__(
        self,
        split,
        sample_rate,
        target_sec,        
        is_mono,
        audio_dir,
        meta_path,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.audio_dir = audio_dir
        self.audio_files = find_audios(audio_dir)

        self.meta_path = meta_path
        with open(self.meta_path) as f:
            self.metadata = json.load(f)
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.audio_names_without_ext)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        """
        return:
            segments: [n_segments, segments_length]
            labels: [n_segments, 2]
        """
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'
        audio_file = os.path.join(self.audio_dir, audio_path)
        
        segments, pad_mask= self.load_audio(audio_file)
        label = self.class2id[self.metadata[audio_name_without_ext]['y']]
        labels = []
        for mask in pad_mask:
            if mask == 1:
                labels.append(label)
            else:
                labels.append(-100)
        labels = torch.tensor(labels)
        segments = torch.vstack(segments)

        return {"audio": segments, "labels":  labels, "n_segments": len(pad_mask)}

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
        label_tensor = torch.cat(label_list,dim=0)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }


class GSdataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_args, 
            codec_name,
            train_batch_size=32,
            valid_batch_size=2,
            test_batch_size=16,
            train_num_workers=8,
            valid_num_workers=4,
            test_num_workers=4):
        super().__init__()
        self.dataset_args = dataset_args
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.codec_name = codec_name
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GSdataset(split="train", **self.dataset_args)
            self.valid_dataset = GSdataset(split="valid", **self.dataset_args)
        if stage == "val":
            self.valid_dataset = GSdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = GSdataset(split="test", **self.dataset_args)

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
            dataset=self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            collate_fn=self.valid_dataset.collate_fn,
            num_workers=self.valid_num_workers,
        )
    
    def test_dataloader(self) :
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )