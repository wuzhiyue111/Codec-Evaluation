import argparse
import random
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.reconstruction_eval.gtzan_dataset.gtzan_dataset import GTZANdataset
from tqdm import tqdm
from codec_evaluation.codecs.init_codecs import init_codec
from typing import Optional
from codec_evaluation.reconstruction_eval.utils import (
    calculate_pesq,
    calculate_stoi,
    calculate_visqol,
    calculate_mel_distance,
)

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

seed_all(666)

class CodecEvaluation:
    def __init__(
        self,
        codec_name: str,
        model_ckpt_dir: str,
        device: str,
        sample_rate: int,
        dataset_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        mode: str = "reconstruct",
        visqol_mode: str = "audio",
        **kwargs,
    ):
        """
        codec_name: codec name, such as encodec, dac, etc.
        model_ckpt_dir: codec model checkpoint directory
        device: GPU device, such as cuda:0
        sample_rate: dataset sample rate
        dataset_path: dataset .arrow file path
        batch_size: batch size
        num_workers: number of workers for dataloader
        mode: codec mode, such as reconstruct, encode_decode, etc.
        """
        self.codec = init_codec(
            modelname=codec_name,
            sample_rate=sample_rate,
            mode=mode,
            model_ckpt_dir=model_ckpt_dir,
            device=device,
            freeze=True,
            need_resample=False,  # return the audio resampled to codec sample rate
            **kwargs,
        )

        self.codec_sample_rate = self.codec.orig_sample_rate
        self.sample_rate = sample_rate
        self.visqol_mode = visqol_mode

        self.batch_size = batch_size
        self.device = device
        dataset = GTZANdataset(dataset_path=dataset_path)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )

    @torch.inference_mode()
    def evaluate(self):
        # first get the reconstruction audio
        gt_audio_list = []
        rec_audio_list = []
        print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
        for batch in tqdm(self.dataloader, desc="reconstruct audio"):
            gt_audio_test = batch["audio"].clone()
            # resample to codec sample rate
            if self.codec_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    self.sample_rate, self.codec_sample_rate
                ).to(gt_audio_test.device)
                gt_audio_test = resampler(gt_audio_test)

            gt_audio = batch["audio"].to(self.device)
            max_length = gt_audio.shape[-1]

            if self.codec_sample_rate != self.sample_rate:
                max_length = int(
                    max_length * (self.codec_sample_rate / self.sample_rate)
                )

            # prevent the more than max_length
            rec_audio = self.codec(gt_audio)[:, :max_length]

            # prevent the less than gt_audio
            min_length = min(gt_audio_test.shape[-1], rec_audio.shape[-1])

            # save to list
            for gt_audio_t, rec_a in zip(gt_audio_test, rec_audio):
                gt_audio_list.append(gt_audio_t[:min_length].to("cpu"))
                rec_audio_list.append(rec_a[:min_length].to("cpu"))

        torch.cuda.empty_cache()
        print(f"reconstruct done, now start to compute metrics")

        # compute metrics
        stoi_list = []
        pesq_list = []
        visqol_list = []
        mel_distance_list = []

        data_length = len(gt_audio_list)
        for i in tqdm(
            range(0, data_length, 50), desc="compute metrics"
        ):  # per 50 samples to compute metrics
            if i + 50 < data_length:
                tmp_gt_audio_list = gt_audio_list[i : i + 50]
                tmp_rec_audio_list = rec_audio_list[i : i + 50]
            else:
                tmp_gt_audio_list = gt_audio_list[i:]
                tmp_rec_audio_list = rec_audio_list[i:]

            tmp_gt_audio = pad_sequence(tmp_gt_audio_list, batch_first=True).to(
                self.device
            )
            tmp_rec_audio = pad_sequence(tmp_rec_audio_list, batch_first=True).to(
                self.device
            )

            # stoi
            stoi_list.append(
                calculate_stoi(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    sample_rate=self.codec_sample_rate,
                )
            )
            print(f"stoi: {stoi_list[-1]}")

            # pesq
            pesq_list.append(
                calculate_pesq(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    sample_rate=self.codec_sample_rate,
                )
            )
            print(f"pesq: {pesq_list[-1]}")

            # visqol
            visqol_list.append(
                calculate_visqol(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    sample_rate=self.codec_sample_rate,
                    visqol_mode=self.visqol_mode,
                )
            )
            print(f"visqol: {visqol_list[-1]}")

            # mel_distance
            mel_distance_list.append(
                calculate_mel_distance(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    sample_rate=self.codec_sample_rate,
                )
            )
            print(f"mel_distance: {mel_distance_list[-1]}")

        avg_stoi = sum(stoi_list) / len(stoi_list)
        avg_pesq = sum(pesq_list) / len(pesq_list)
        avg_visqol = sum(visqol_list) / len(visqol_list)
        avg_mel_distance = sum(mel_distance_list) / len(mel_distance_list)
        print(f"compute metrics done, now start to save results")

        print(f"stoi: {avg_stoi}")
        print(f"pesq: {avg_pesq}")
        print(f"visqol: {avg_visqol}")
        print(f"mel_distance: {avg_mel_distance}")
        return {
            "stoi": avg_stoi,
            "pesq": avg_pesq,
            "visqol": avg_visqol,
            "mel_distance": avg_mel_distance,
        }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", 
                        type=str, 
                        required=True, 
                        help="Name of the audio codec model to be used (e.g., 'encodec', 'dac').")
    parser.add_argument("--model_ckpt_dir",
                        type=str,
                        required=True, 
                        help="Directory containing the pretrained checkpoint files for the specified audio codec.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", 
                        type=int, 
                        required=True,
                        help="The sample rate of the dataset audio files.")
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True, 
                        help="The huggingface dataset path obtained using the script.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mode", type=str, default="reconstruct")
    parser.add_argument("--use_vocos", 
                        type=bool, 
                        default=False,
                        help="Whether to use Vocos to post-process the audio after decoding.")
    parser.add_argument("--vocos_ckpt_dir", 
                        type=Optional[str], 
                        default=None,
                        help="The directory containing the vocos checkpoint files.")
    parser.add_argument("--visqol_mode", 
                        type=str, 
                        default="audio",
                        help="Mode for VISQOL metric calculation, either 'speech' or 'audio'.")
    args = parser.parse_args()
    print(f"args: {args}")
    codec_eval = CodecEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
        visqol_mode=args.visqol_mode,
    )
    result = codec_eval.evaluate()
    print(f"result: {result}")

    return 0


if __name__ == "__main__":
    cli()
