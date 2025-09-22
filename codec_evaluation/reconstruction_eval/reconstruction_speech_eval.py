import argparse
import random
import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from speechbrain.inference.speaker import EncoderClassifier
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.reconstruction_eval.libritts_dataset.libritts_dataset import LibriTTS_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from codec_evaluation.codecs.init_codecs import init_codec
from typing import Optional
from codec_evaluation.reconstruction_eval.utils import (
    calculate_pesq,
    calculate_spk_sim,
    calculate_stoi,
    calculate_visqol,
    calculate_mel_distance,
    wer,
    cer,
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
        asr_model_path_or_name: str,
        dataset_path: str,
        base_audio_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        mode: str = "reconstruct",
        wav2vec_model_path_or_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        visqol_mode: str = "speech",
        **kwargs,
    ):
        """
        codec_name: codec name, such as encodec, dac, etc.
        model_ckpt_dir: codec model checkpoint directory
        device: GPU device, such as cuda:0
        sample_rate: dataset sample rate
        asr_model_path_or_name: asr model path or name for wer compute
        dataset_path: dataset .arrow file path
        base_audio_dir: base audio dir for dataset load
        batch_size: batch size  
        num_workers: number of workers for dataloader
        mode: codec mode, such as reconstruct, encode_decode, etc.
        wav2vec_model_path_or_name: wav2vec model for computing spk_sim
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

        # wer model
        self.asr_processor = WhisperProcessor.from_pretrained(asr_model_path_or_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            asr_model_path_or_name, device_map=device
        )
        self.asr_model.config.forced_decoder_ids = None

        # embedding model
        self.wav2vec_model = EncoderClassifier.from_hparams(source=wav2vec_model_path_or_name)
        self.wav2vec_model.to(device)
        self.wav2vec_model.device = device

        self.batch_size = batch_size
        self.device = device
        dataset = LibriTTS_dataset(dataset_path=dataset_path, base_audio_dir=base_audio_dir)
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
            max_length = gt_audio_test.shape[-1]

            # save to list
            texts = batch["text"]
            for text in texts:
                text_list.append(text)

            gt_audio = batch["audio"].to(self.device)

            # prevent the more than max_length | prevent rectify the gt_audio
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
        speaker_sim_list = []
        usage_entropy_list = []
        visqol_list = []
        mel_distance_list = []

        data_length = len(gt_audio_list)
        for i in tqdm(range(0, data_length, 50), desc="compute metrics"):  # per 50 samples to compute metrics
            if i + 50 < data_length:
                tmp_gt_audio_list = gt_audio_list[i : i + 50]
                tmp_rec_audio_list = rec_audio_list[i : i + 50]
                tmp_text_list = text_list[i : i + 50]
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
            speaker_sim_list.append(
                calculate_spk_sim(
                    gt_audio=tmp_gt_audio,
                    rec_audio=tmp_rec_audio,
                    model=self.wav2vec_model,
                )
            )
            print(f"speaker_sim: {speaker_sim_list[-1]}")
            
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

        avg_wer_gt = sum(wer_gt_list) / len(wer_gt_list)
        avg_wer_rec = sum(wer_rec_list) / len(wer_rec_list)
        avg_cer_gt = sum(cer_gt_list) / len(cer_gt_list)
        avg_cer_rec = sum(cer_rec_list) / len(cer_rec_list)
        avg_stoi = sum(stoi_list) / len(stoi_list)
        avg_pesq = sum(pesq_list) / len(pesq_list)
        avg_speaker_sim = sum(speaker_sim_list) / len(speaker_sim_list)
        avg_visqol = sum(visqol_list) / len(visqol_list)
        avg_mel_distance = sum(mel_distance_list) / len(mel_distance_list)
        print(f"compute metrics done, now start to save results")
        print(f"speaker_sim: {avg_speaker_sim}")
        print(f"wer_gt: {avg_wer_gt}")
        print(f"wer_rec: {avg_wer_rec}")
        print(f"cer_gt: {avg_cer_gt}")
        print(f"cer_rec: {avg_cer_rec}")
        print(f"stoi: {avg_stoi}")
        print(f"pesq: {avg_pesq}")
        print(f"visqol: {avg_visqol}")
        print(f"mel_distance: {avg_mel_distance}")
        return {
            "wer_gt": avg_wer_gt,
            "cer_gt": avg_cer_gt,
            "speaker_sim": avg_speaker_sim,
            "wer_rec": avg_wer_rec,
            "cer_rec": avg_cer_rec,
            "stoi": avg_stoi,
            "pesq": avg_pesq,
            "visqol": avg_visqol,
            "mel_distance": avg_mel_distance,
            # "codebook_usage": np.mean(per_codebook_usage, axis=0) / np.log2(self.effective_codebook_num)
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
    parser.add_argument("--asr_model_path_or_name", 
                        type=str, 
                        required=True, 
                        help="Path of the pre-trained ASR model to be used for evaluation.")
    parser.add_argument("--dataset_path", 
                        type=str, 
                        required=True, 
                        help="The huggingface dataset path obtained using the script.")
    parser.add_argument("--base_audio_dir", 
                        type=str, 
                        required=True, 
                        help="The root directory where the raw audio files are stored.(Used to splice the complete audio path)")
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
    parser.add_argument("--wav2vec_model_path_or_name", 
                        type=str, 
                        required=True, 
                        help="Path of the pre-trained wav2vec model to be used for evaluation.")
    parser.add_argument("--visqol_mode", 
                        type=str, 
                        default="speech",
                        help="Mode for VISQOL metric calculation, music use 'audio', speech use 'speech'.")
    args = parser.parse_args()
    print(f"args: {args}")
    codec_eval = CodecEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        asr_model_path_or_name=args.asr_model_path_or_name,
        dataset_path=args.dataset_path,
        base_audio_dir=args.base_audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
        wav2vec_model_path_or_name=args.wav2vec_model_path_or_name,
        visqol_mode=args.visqol_mode,
    )
    result = codec_eval.evaluate()
    print(f"result: {result}")

    return 0


if __name__ == "__main__":
    cli()
