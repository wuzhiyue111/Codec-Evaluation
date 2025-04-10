import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from codec_evaluation.utils.logger import RankedLogger
from torch.nn.utils.rnn import pad_sequence
from codec_evaluation.utils.utils import find_audios

logger = RankedLogger(__name__, rank_zero_only=True)

class LibriTTS_ctc_dataset(Dataset):
    def __init__(
        self,
        audio_dir,
    ):
        self.all_paths = find_audios(audio_dir)
        logger.info(f"Found {len(self.all_paths)} audio files in {audio_dir}")

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            logger.error(f"Error loading audio file {self.all_paths[index]}: {e}")
            return None

    def load_text(self, text_path):
        with open(text_path, "r") as f:
            text = f.read().strip()
        return text

    def get_item(self, index):
        """
        return:
            segments: [1, wavform_length]
            labels: [1, 20]
        """
        audio_path = self.all_paths[index]
        waveform, _ = torchaudio.load(audio_path)  # [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        text_path = audio_path.replace(".wav", ".normalized.txt")
        text = self.load_text(text_path)

        return {"audio": waveform, "text": text, "audio_length": waveform.shape[1]}

    def collate_fn(self, batch):
        """
        return:
            audio: [B, T]
            text: [B]
        """
        audio_list = [item["audio"].squeeze(0) for item in batch if item is not None]
        text_list = [item["text"] for item in batch if item is not None]
        audio_length_list = [item["audio_length"] for item in batch if item is not None]

        audio_tensor = pad_sequence(audio_list, batch_first=True) # [B, T]
        audio_length_tensor = torch.tensor(audio_length_list)

        return {
            "audio": audio_tensor,
            "text": text_list,
            "audio_length": audio_length_tensor,
        }


class LibriTTS_ctc_module(pl.LightningDataModule):
    def __init__(
        self,
        train_audio_dir=None,
        valid_audio_dir=None,
        test_audio_dir=None,
        train_batch_size=16,
        valid_batch_size=16,
        test_batch_size=16,
        train_num_workers=4,
        valid_num_workers=4,
        test_num_workers=4,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.train_audio_dir = train_audio_dir
        self.valid_audio_dir = valid_audio_dir
        self.test_audio_dir = test_audio_dir
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.test_num_workers = test_num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LibriTTS_ctc_dataset(self.train_audio_dir)
            self.valid_dataset = LibriTTS_ctc_dataset(self.valid_audio_dir)
        if stage == "val":
            self.valid_dataset = LibriTTS_ctc_dataset(self.valid_audio_dir)
        if stage == "test":
            self.test_dataset = LibriTTS_ctc_dataset(self.test_audio_dir)

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.test_num_workers,
        )