import os
import random
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.probe.dataset.GTZAN_dataset.GTZAN_dataset import (
    GTZANdataset,
)
from tqdm import tqdm
from codec_evaluation.init_codecs import init_codec
from typing import Optional
from codec_evaluation.reconstruction_eval.utils import (
    calculate_pesq,
    calculate_stoi,
)
from tqdm import tqdm


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
        dataset_audio_dir: str,
        dataset_meta_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        mode: str = "reconstruct",
        **kwargs,
    ):
        """
        codec_model_safetensors_path: codec absolute path model.safetensors
        asr_model_path_or_name: asr model path or name for wer compute
        wav2vec_model_path_or_name: wav2vec model for computing spk_sim
        dataset_meta_path: absolute path to dataset_meta json
        dataset_audio_path: where the audio root path is
        sample_rate: audio sample rate
        device: cuda:0 or cpu
        batch_size: batch size
        codec_trunk_size: the trunk size (seconds) of extract code
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

        self.batch_size = batch_size
        self.device = device
        dataset = GTZANdataset(
            audio_dir=dataset_audio_dir,
            split="test",
            meta_dir=dataset_meta_dir,
            sample_rate=sample_rate,
            target_sec=None,
        )
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

        avg_stoi = sum(stoi_list) / len(stoi_list)
        avg_pesq = sum(pesq_list) / len(pesq_list)
        print(f"compute metrics done, now start to save results")

        print(f"stoi: {avg_stoi}")
        print(f"pesq: {avg_pesq}")
        return {
            "stoi": avg_stoi,
            "pesq": avg_pesq,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="encodec")
    parser.add_argument(
        "--model_ckpt_dir",
        type=str,
        default="/sdb/model_weight/codec_evaluation/codec_ckpt/encodec/models--facebook--encodec_24khz",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument(
        "--dataset_audio_dir",
        type=str,
        default="/sdb/data1/music/mix_music/marble_dataset/data/GTZAN/genres",
    )
    parser.add_argument(
        "--dataset_meta_dir",
        type=str,
        default="/sdb/data1/music/mix_music/marble_dataset/data/GTZAN",
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mode", type=str, default="reconstruct")
    parser.add_argument("--use_vocos", type=bool, default=False)
    parser.add_argument("--vocos_ckpt_dir", type=Optional[str], default=None)
    args = parser.parse_args()
    print(f"args: {args}")
    codec_eval = CodecEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        dataset_audio_dir=args.dataset_audio_dir,
        dataset_meta_dir=args.dataset_meta_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
    )
    result = codec_eval.evaluate()
    print(f"result: {result}")
