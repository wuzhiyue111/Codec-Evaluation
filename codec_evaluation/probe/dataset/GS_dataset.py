import os
import torch
import torchaudio
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import reduce

from codec_evaluation.utils.utils import find_audios, cut_or_pad


class GSdataset(Dataset):
    def __init__(
        self,
        sample_rate,
        target_sec,
        n_segments,
        is_mono,
        is_normalize,
        audio_dir,
        meta_path,
    ):
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.n_segments = n_segments
        self.target_length = self.target_sec * self.sample_rate
        self.is_mono = is_mono
        self.is_normalize = is_normalize
        self.audio_files = find_audios(audio_dir)

        self.meta_path = meta_path
        with open(self.meta_path) as f:
            self.metadata = json.load(f)
        self.audio_names_without_ext = [k for k in self.metadata.keys()]
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        """
        return:
            segments: [n_segments, segments_length]
            labels: [n_segments, 2]
        """
        
        audio_file = self.audio_files[index]
        
        waveform = self.load_audio(audio_file)
        label = self.load_label(index)

        segments = self.split_audio(waveform, self.n_segments)
        segments = torch.stack(segments, dim=0)

        labels = label.unsqueeze(0).repeat(self.n_segments, 1)
        return segments, labels

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
        if len(audio_file) == 0:
            raise FileNotFoundError("No audio files found in the specified directory.")

        try:
            waveform, _ = torchaudio.load(audio_file)
        except Exception as e:
            print(f"Error loading audio file {audio_file}: {e}")
            return None

        # Convert to mono if needed
        if waveform.shape[0] > 1 and self.is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize to [-1, 1] if needed
        if self.is_normalize:
            waveform = waveform / waveform.abs().max()

        waveform = cut_or_pad(waveform=waveform, target_length=self.target_length)

        waveform = waveform.squeeze(0)

        return waveform

    def load_label(self, index):
        """Load the label for a given audio from meta.json.
        input:
            index: the index of the audio
        return:
            label:[24]
        """

        audio_name_without_ext = self.audio_names_without_ext[index]
        # Extract the filename without the extension
        label = torch.tensor(self.class2id[self.metadata[audio_name_without_ext]['y']]).float()

        return label

    def split_audio(self, waveform, n_segments):
        """
        input:
            waveform:[T];
            n_segments: the number of segments per audio
        return:
            segments:[n_segments,segment_length]
        """

        segment_length = self.target_length // n_segments  # length of per segment
        segments = []

        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = waveform[start:end]
            segments.append(segment)

        return segments


class GSdataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, val_split, codec_name, num_workers):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_dataset = None
        self.valid_dataset = None
        self.codec_name = codec_name
        self.n_segments = dataset.n_segments
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_size = int((1 - self.val_split) * len(self.dataset))
            valid_size = len(self.dataset) - train_size
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, valid_size]
            )

    def train_collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * n_segments, length)
            labels_tensor:(batch_size * n_segments, 24)
            if codecname='semanticodec'
                labels_tensor:(batch_size * n_segments * 2, 24)
        """
        features, labels = zip(*batch)
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        labels_tensor = labels_tensor.squeeze(1)

        if self.codec_name == "semanticodec":
            labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

        return features_tensor, labels_tensor

    def valid_collate_fn(self, batch):
        """
        return:
            features_tensor:(batch_size * n_segments, length)
            labels_tensor:(batch_size , 2)
            if codecname='semanticodec'
                labels_tensor:(batch_size * 2, 2)
        """

        features, labels = zip(*batch)
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        labels_tensor = reduce(
            labels_tensor, "(b g) n -> b", reduction="mean", g=self.n_segments
        )

        if self.codec_name == "semanticodec":
            labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)


        return features_tensor, labels_tensor

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.valid_collate_fn,
            num_workers=self.num_workers,
        )
