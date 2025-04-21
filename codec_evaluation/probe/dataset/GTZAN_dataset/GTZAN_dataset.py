import os
import torchaudio
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from codec_evaluation.utils.utils import cut_or_pad
from torch.nn.utils.rnn import pad_sequence
from codec_evaluation.utils.logger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

class GTZANdataset(Dataset):
    def __init__(
        self,
        split,
        audio_dir,
        meta_dir,
        sample_rate,
        target_sec,        
        is_mono ,
    ):
        """
            split: train, valid, test
            sample_rate: audio sample rate
            target_sec: target segment length (seconds)
            is_mono: whether to convert to mono
            is_normalize: whether to normalize the audio
            audio_dir: audio root path
            meta_dir: meta root path
        """
        self.split = split
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        if self.target_sec is not None:
            self.target_length = self.target_sec * self.sample_rate
        else:
            self.target_length = None

        self.is_mono = is_mono
        self.audio_dir = audio_dir
        self.meta_dir = meta_dir
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(meta_dir, f'{split}_filtered.txt'), 
                                    names = ['audio_path'])
        self.class2id = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            logger.error(f"Error loading audio file {self.metadata.iloc[index].iloc[0]}: {e}")

    def getitem(self, index):
        """
            return:
                segments: [n_segments, segments_length]
                labels: [n_segments, 10]
        """
        audio_path = self.metadata.iloc[index].iloc[0]
        audio_file = os.path.join(self.audio_dir,audio_path)

        segments, pad_mask= self.load_audio(audio_file)
        label = self.class2id[audio_path.split('/')[0]]
        labels = []
        for mask in pad_mask:
            if mask == 1:
                labels.append(label)
            else:
                labels.append(-100)
        labels = torch.tensor(labels)
        if self.target_length is not None:
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
            waveform:[T]
        """
        waveform, _ = torchaudio.load(audio_file)

        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.target_length is not None:
            waveform, pad_mask = cut_or_pad(waveform=waveform, target_length=self.target_length)
        else:
            pad_mask = [1]

        return waveform, pad_mask

    def collate_fn(self, batch):
        audio_list = [item["audio"] for item in batch if item is not None]
        label_list = [item["labels"] for item in batch if item is not None]
        n_segments_list = [item["n_segments"] for item in batch if item is not None]

        if self.target_length is not None:
            audio_tensor = torch.vstack(audio_list)
        else:
            audio_list = [audio.squeeze(0) for audio in audio_list]
            audio_tensor = pad_sequence(audio_list, batch_first=True)

        label_tensor = torch.cat(label_list,dim=0)

        return {
            "audio": audio_tensor,
            "labels": label_tensor,
            "n_segments_list": n_segments_list
        }


class GTZANdataModule(pl.LightningDataModule):
    def __init__(self,
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
            self.train_dataset = GTZANdataset(split="train", **self.dataset_args)
            self.valid_dataset = GTZANdataset(split="valid", **self.dataset_args) 
        if stage == "val":
            self.valid_dataset = GTZANdataset(split="valid", **self.dataset_args)
        if stage == "test":
            self.test_dataset = GTZANdataset(split="test", **self.dataset_args)

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