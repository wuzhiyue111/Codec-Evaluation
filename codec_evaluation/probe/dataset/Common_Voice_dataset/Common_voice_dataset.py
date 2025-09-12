import torch
import os
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk

logger = RankedLogger(__name__, rank_zero_only=True)

class Common_voice_dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,    
        base_audio_dir: str,    
        target_samplerate=48000
        
    ):
        super().__init__()
        self.dataset = load_from_disk(dataset_path)  
        self.base_audio_dir = base_audio_dir         
        self.dataset_path = dataset_path
        self.target_samplerate = target_samplerate
        print(f"Found {len(self.dataset)} audio files in {base_audio_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            example = self.dataset[index]
            audio_path = example["audio_path"]
            logger.error(f"Error loading {audio_path}: {e}")
            return None
        
    def get_item(self, index):

        """
            audio:[1,T]
        """
        example = self.dataset[index]
        audio_path = os.path.join(self.base_audio_dir, example["audio_path"]) 
            
        audio, samplerate = torchaudio.load(audio_path)

        if samplerate != self.target_samplerate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=samplerate, 
                new_freq=self.target_samplerate
            )
            audio = resampler(audio)


        if audio.shape[1] > 20*48000:
            return None
        else:
            return {
                "audio": audio,
                "text": example["text"],
                "audio_length": audio.shape[1]
            }

    def collate_fn(self, batch):
        """
        return:
            audio: [B, T]
            text: [B]
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        text_list = [item["text"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]
        
        audio_tensor = pad_sequence(audio_list, batch_first=True)
        audio_length_tensor = torch.tensor(audio_length_list)
        
        return {
            "audio": audio_tensor,
            "text": text_list,
            "audio_length": audio_length_tensor,
        }

class Common_voice_module(pl.LightningDataModule):
    def __init__(
        self,
        train_audio_dir,
        val_audio_dir,
        test_audio_dir,
        base_audio_dir,
        target_samplerate,
        train_batch_size=16,
        val_batch_size=16,
        test_batch_size=1,
        train_num_workers=1,
        val_num_workers=1,
        test_num_workers=1,
        
    ):
        super().__init__()
        self.base_audio_dir = base_audio_dir
        self.train_audio_dir = train_audio_dir
        self.val_audio_dir = val_audio_dir
        self.test_audio_dir = test_audio_dir
        self.target_samplerate = target_samplerate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Common_voice_dataset(dataset_path=self.train_audio_dir, base_audio_dir=self.base_audio_dir, target_samplerate=self.target_samplerate)
            self.val_dataset = Common_voice_dataset(dataset_path=self.val_audio_dir, base_audio_dir=self.base_audio_dir, target_samplerate=self.target_samplerate)
        if stage == "val":
            self.val_dataset = Common_voice_dataset(dataset_path=self.val_audio_dir, base_audio_dir=self.base_audio_dir, target_samplerate=self.target_samplerate)
        if stage == "test":
            self.test_dataset = Common_voice_dataset(dataset_path=self.test_audio_dir, base_audio_dir=self.base_audio_dir, target_samplerate=self.target_samplerate)

    
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