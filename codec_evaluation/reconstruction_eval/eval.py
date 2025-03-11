import os
import random
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
# from speechbrain.inference.speaker import EncoderClassifier
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.probe.dataset.LibriTTS_dataset.libritts_ctc import (
    LibriTTS_ctc_dataset,
)
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from codec_evaluation.init_codecs import init_codec
from typing import Optional
from codec_evaluation.reconstruction_eval.utils import (
    calculate_f0_corr,
    calculate_pesq,
    calculate_si_snr,
    calculate_spk_sim,
    calculate_stoi,
    wer,
    cer,
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
        asr_model_path_or_name: str,
        dataset_audio_dir: str,
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

        # wer model
        self.asr_processor = WhisperProcessor.from_pretrained(asr_model_path_or_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            asr_model_path_or_name, device_map=device
        )
        self.asr_model.config.forced_decoder_ids = None

        # embedding model
        # self.wav2vec_model = EncoderClassifier.from_hparams(source=wav2vec_model_path_or_name)

        self.batch_size = batch_size
        self.device = device
        dataset = LibriTTS_ctc_dataset(audio_dir=dataset_audio_dir)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )

    def get_wer(
        self,
        gt_audio_test,
        rec_audio,
        text_list,
        asr_processor,
        asr_model,
        sample_rate=24000,
    ):
        with torch.no_grad():
            return wer(
                gt_audio_test,
                rec_audio,
                text_list,
                asr_processor,
                asr_model,
                sample_rate=sample_rate,
            )

    def get_speaker_sim(self, gt_audio, rec_audio, processor, model, sample_rate=24000):
        with torch.no_grad():
            return calculate_spk_sim(
                gt_audio, rec_audio, processor, model, sample_rate=sample_rate
            )

    def get_f0_corr(self, gt_audio, rec_audio):
        with torch.no_grad():
            return calculate_f0_corr(gt_audio, rec_audio)

    def get_si_snr(self, gt_audio, rec_audio):
        with torch.no_grad():
            return calculate_si_snr(gt_audio, rec_audio)

    def get_text_from_path(self, posix_path_list):
        text_list = []
        for path in posix_path_list:
            abs_path = os.path.join(self.audio_root_path, str(path))
            text_abs_path = abs_path.replace(".wav", ".normalized.txt")
            with open(text_abs_path, "r") as f:
                text_list.append(f.read())
        return text_list

    def evaluate(self):
        # first get the reconstruction audio
        gt_audio_list = []
        rec_audio_list = []
        text_list = []
        print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
        for batch in tqdm(self.dataloader, desc="reconstruct audio"):
            gt_audio_test = batch["audio"].clone()
            # resample to codec sample rate
            if self.codec_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    self.sample_rate, self.codec_sample_rate
                ).to(gt_audio_test.device)
                gt_audio_test = resampler(gt_audio_test)

            # save to list
            texts = batch["text"]
            for text in texts:
                text_list.append(text)

            gt_audio = batch["audio"].to(self.device)
            audio_lengths = batch["audio_length"]
            max_length = audio_lengths.max().item()

            if self.codec_sample_rate != self.sample_rate:
                max_length = int(max_length * (self.codec_sample_rate / self.sample_rate))

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
        wer_rec_list = []
        wer_gt_list = []
        cer_rec_list = []
        cer_gt_list = []
        stoi_list = []
        pesq_list = []
        usage_entropy_list = []

        with torch.no_grad():
            data_length = len(gt_audio_list)
            for i in tqdm(range(0, data_length, 100), desc="compute metrics"):  # per 100 samples to compute metrics
                if i + 100 < data_length:
                    tmp_gt_audio_list = gt_audio_list[i : i + 100]
                    tmp_rec_audio_list = rec_audio_list[i : i + 100]
                    tmp_text_list = text_list[i : i + 100]

                else:
                    tmp_gt_audio_list = gt_audio_list[i:]
                    tmp_rec_audio_list = rec_audio_list[i:]
                    tmp_text_list = text_list[i:]
                tmp_gt_audio = pad_sequence(tmp_gt_audio_list, batch_first=True).to(self.device)
                tmp_rec_audio = pad_sequence(tmp_rec_audio_list, batch_first=True).to(self.device)
                # wer
                wer_gt, wer_rec = wer(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    gt_text=tmp_text_list,
                    processor=self.asr_processor,
                    model=self.asr_model,
                    device=self.device,
                    sample_rate=self.codec_sample_rate,
                )
                wer_rec_list.append(wer_rec)
                wer_gt_list.append(wer_gt)
                print(f"wer_gt: {wer_gt}, wer_rec: {wer_rec}")

                # cer
                cer_gt, cer_rec = cer(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    gt_text=tmp_text_list,
                    processor=self.asr_processor,
                    model=self.asr_model,
                    device=self.device,
                    sample_rate=self.codec_sample_rate,
                )
                cer_rec_list.append(cer_rec)
                cer_gt_list.append(cer_gt)
                print(f"cer_gt: {cer_gt}, cer_rec: {cer_rec}")

                # speaker_sim
                # speaker_sim_list.append(
                #     self.get_speaker_sim(
                #         gt_audio,
                #         rec_audio,
                #         self.wav2vec_feature_extractor,
                #         self.wav2ver_model,
                #     )
                # )
                # print(f"speaker_sim: {speaker_sim_list[-1]}")

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

        avg_wer_gt = sum(wer_gt_list) / len(wer_gt_list)
        avg_wer_rec = sum(wer_rec_list) / len(wer_rec_list)
        avg_cer_gt = sum(cer_gt_list) / len(cer_gt_list)
        avg_cer_rec = sum(cer_rec_list) / len(cer_rec_list)
        avg_stoi = sum(stoi_list) / len(stoi_list)
        avg_pesq = sum(pesq_list) / len(pesq_list)
        print(f"compute metrics done, now start to save results")
        print(f"wer_gt: {avg_wer_gt}")
        print(f"wer_rec: {avg_wer_rec}")
        print(f"cer_gt: {avg_cer_gt}")
        print(f"cer_rec: {avg_cer_rec}")
        print(f"stoi: {avg_stoi}")
        print(f"pesq: {avg_pesq}")
        return {
            "wer_gt": avg_wer_gt,
            "cer_gt": avg_cer_gt,
            # "speaker_sim": np.mean(np.array(speaker_sim_list)),
            "wer_rec": avg_wer_rec,
            "cer_rec": avg_cer_rec,
            "stoi": avg_stoi,
            "pesq": avg_pesq,
            # "codebook_usage": np.mean(per_codebook_usage, axis=0) / np.log2(self.effective_codebook_num)
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="dac")
    parser.add_argument("--model_ckpt_dir", type=str, default="/sdb/model_weight/codec_evaluation/codec_ckpt/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--asr_model_path_or_name", type=str, default="/sdb/model_weight/whisper-base")
    parser.add_argument("--dataset_audio_dir", type=str, default="/sdb/data1/speech/24kHz/LibriTTS/test-other")
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
        asr_model_path_or_name=args.asr_model_path_or_name,
        dataset_audio_dir=args.dataset_audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
    )
    result = codec_eval.evaluate()
    print(f"result: {result}")
