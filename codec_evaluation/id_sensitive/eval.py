from codec_evaluation.init_codecs import init_codec
import torchaudio
import random
import numpy as np
import torch
import os
import codec_evaluation
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
root_path = codec_evaluation.__path__[0]
from codec_evaluation.probe.dataset.LibriTTS_dataset.libritts_ctc import (
    LibriTTS_ctc_dataset,
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

class IDSensitiveEvaluation:
    def __init__(
        self,
        codec_name: str,
        model_ckpt_dir: str,
        sample_rate: int,
        num_codebooks: int,
        need_resample: bool,
        dataset_audio_dir: str,
        device: str,
        task: str,
        batch_size: int = 24,
        num_workers: int = 8,
        shift_time: int = 2,
        **kwargs,
    ):
        """
            codec_name: codec model name
            model_ckpt_dir: codec model checkpoint directory
            sample_rate: audio sample rate
            num_codebooks: number of codebooks
            need_resample: boolean, whether to resample the audio after decoding
            dataset_audio_dir: dataset audio directory
            task: task name "MRC" or "OS2"  MRC: multi-round reconstruction, OS2: offset-2ms 
            num_workers: number of workers
            shift_time: shift time in ms
            kwargs: other arguments
        """
        self.device = device
        self.codec_encode = init_codec(
            modelname=codec_name,
            mode="encode",
            sample_rate=sample_rate,
            device=self.device,
            num_codebooks=num_codebooks,
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )
        self.codec_encode.sample_rate = self.codec_encode.orig_sample_rate

        self.codec_reconstruct = init_codec(
            modelname=codec_name,
            mode="reconstruct",
            sample_rate=sample_rate,
            device=self.device,
            num_codebooks=num_codebooks,
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir,
            **kwargs,
        )

        self.num_codebooks = num_codebooks
        self.codec_name = codec_name
        self.codec_sample_rate = self.codec_encode.orig_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = device
        self.shift_time = shift_time
        self.task = task
        dataset = LibriTTS_ctc_dataset(audio_dir=dataset_audio_dir)
        dataset = torch.utils.data.Subset(dataset, range(1200))
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.dataset.collate_fn,
        )
    def get_offset_2ms_audio(self, audio_tensor, sample_rate):
        shift_samples = int(sample_rate * self.shift_time / 1000)
        y_shifted = torch.roll(audio_tensor, shifts=shift_samples, dims=1)
        y_shifted[:, :shift_samples] = 0
        return y_shifted

    def get_same_id(self, gt_id, rec_id):
        results = []
        for i in range(gt_id.shape[1]):
            same = (gt_id[0, i, :] == rec_id[0, i, :]).sum().item()
            results.append(same)
        return results


    def get_id(self, gt, rec):
        gt = gt.to(self.device)
        rec = rec.to(self.device)
        gt_id, _ = self.codec_encode(gt, length=None)
        gt_id = gt_id.movedim(-1, -2)
        rec_id, _ = self.codec_encode(rec, length=None)
        rec_id = rec_id.movedim(-1, -2)
        return gt_id, rec_id

    def multi_round_reconstruction(self, gt_audio_clone, gt_audio, max_length, rec_audio_dict, gt_audio_dict):
        for i in range(10):
            rec_audio = self.codec_reconstruct(gt_audio)[:, :max_length]
            min_length = min(gt_audio.shape[-1], rec_audio.shape[-1])
            for idx, rec_a in enumerate(rec_audio):
                rec_audio_dict[f"round_{i + 1}"].append(rec_a[:min_length].unsqueeze(0).to("cpu"))
                gt_audio_dict[f"round_{i + 1}"].append(gt_audio_clone[idx][:min_length].unsqueeze(0).to("cpu"))
            gt_audio = rec_audio
            max_length = gt_audio.shape[-1]

        return rec_audio_dict, gt_audio_dict

    def data_process(self, same_id_dict_round):
        round_list = [f'round_{i+1}' for i in range(10)]

        avg_same_id_dict = {} # {round_1: [avg_codebook_1, avg_codebook_2, ...], round_2: [], ...}
        for i in range(10): # round
            avg_same_id_dict[round_list[i]] = []

        for key, value_same_id_dict in same_id_dict_round.items(): # round
            for _, value_same_id in value_same_id_dict.items(): # codebook
                if value_same_id != []:
                    numeric_values = [float(val.rstrip('%')) for val in value_same_id]
                    average = sum(numeric_values) / len(numeric_values)
                    average_with_percentage = "{:.2f}%".format(average)
                    avg_same_id_dict[key].append(average_with_percentage)
        
        result_codebook_same_id = {}
        for i in range(self.num_codebooks):
            codebook_key = f'codebook{i+1}'
            result_codebook_same_id[codebook_key] = []
            for round_values in avg_same_id_dict.values():
                result_codebook_same_id[codebook_key].append(round_values[i])

        return result_codebook_same_id

    def plot_mrc_avg_same_id(self, avg_same_id_dict):
        num_codebooks = len(avg_same_id_dict)
        num_rounds = len(next(iter(avg_same_id_dict.values())))
        bar_width = 0.8 / num_rounds
        x_labels = list(avg_same_id_dict.keys())
        x_positions = np.arange(num_codebooks)

        for i in range(num_rounds):
            values = [float(avg_same_id_dict[codebook][i].strip('%')) for codebook in x_labels]
            positions = x_positions + i * bar_width
            plt.bar(positions, values, width=bar_width, label=f"Round {i + 1}")

        plt.figure(figsize=(12, 6))
        plt.xticks(x_positions + (bar_width * (num_rounds - 1)) / 2, x_labels)
        plt.xlabel("Codebooks")
        plt.ylabel("Codebook ID Proportion(%)", rotation=90, labelpad=3, fontweight='bold')
        plt.title(f"{self.codec_name} - {self.task} - Average Codebook Same ID Across 10 Rounds")
        plt.legend(title="Reconstruction Rounds")
        plt.ylim(0, 100)
        y_ticks = np.arange(0, 101, 10)
        plt.yticks(y_ticks, [f"{tick}%" for tick in y_ticks])
        save_dir = os.path.join(root_path, "id_sensitive", f"{self.task}_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.codec_name}_mrc_avg_same_id.png")
        plt.savefig(save_path)

    def plot_os2_avg_same_id(self, percent_same_id_avg_list):
        x_labels = ["Codebook Same ID"]
        codebook_labels = [f"codebook{i + 1}" for i in range(self.num_codebooks)]
        bar_width = 0.1  
        percent_same_id_avg_list = [float(p.strip('%')) / 100 for p in percent_same_id_avg_list]
        _, ax = plt.subplots()

        total_display_width = 0.8
        total_width = bar_width * self.num_codebooks
        offset = (total_display_width - total_width) / 2  

        for i in range(self.num_codebooks):
            positions = [offset + j + i * bar_width for j in range(len(x_labels))]
            values = [percent_same_id_avg_list[i]]
            bars = ax.bar(positions, values, width=bar_width, label=codebook_labels[i])
            ax.bar_label(bars, padding=3, labels=[f"{percent_same_id_avg_list[i] * 100:.1f}"])

        ax.set_xticks([offset + i + (bar_width * (self.num_codebooks - 1)) / 2 for i in range(len(x_labels))])
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Codebook ID Proportion(%)", rotation=90, labelpad=3, fontweight='bold')
        ax.set_title(f"{self.codec_name} - {self.task} - Codebook Same Id Average(%)")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.set_yticklabels([f"{i * 20}" for i in range(6)])
        save_dir = os.path.join(root_path, "id_sensitive", f"{self.task}_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.codec_name}_os2_avg_same_id_results.png")
        plt.savefig(save_path)

    @torch.inference_mode
    def evaluate(self, task):
        if task == "MRC":
            gt_audio_dict = {}
            rec_audio_dict = {}
            for i in range(10):
                gt_audio_dict[f"round_{i + 1}"] = []
                rec_audio_dict[f"round_{i + 1}"] = []

            print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
            gt_audio_lengths = []
            for batch in tqdm(self.dataloader, desc="reconstruct audio"):
                gt_audio_test = batch["audio"].clone().to(self.device)  # shape [B, T]
                # resample to codec sample rate
                if self.codec_sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, self.codec_sample_rate
                    ).to(self.device)
                    gt_audio_test = resampler(gt_audio_test)
                    resample_ratio = self.sample_rate / self.codec_sample_rate
                    gt_audio_length = (batch["audio_length"] / resample_ratio).long()
                else:
                    gt_audio_length = batch["audio_length"]

                gt_audio_lengths.append(gt_audio_length)
                max_length = gt_audio_test.shape[-1]
                gt_audio = batch["audio"].to(self.device)

                rec_audio_dict, gt_audio_dict = self.multi_round_reconstruction(
                    gt_audio_clone=gt_audio_test,
                    gt_audio=gt_audio,
                    max_length=max_length,
                    rec_audio_dict=rec_audio_dict,
                    gt_audio_dict=gt_audio_dict,
                )

            torch.cuda.empty_cache()
            print(f"Reconstruct done! Next, carry out the MRC task.")

            same_id_dict_round = {}
            for i in range(10): # round
                same_id_dict_round[f"round_{i + 1}"] = {}
                for j in range(self.num_codebooks): # codebook
                    same_id_dict_round[f"round_{i + 1}"][f"codebook_{j + 1}"] = []

            all_lengths = [length for sublist in gt_audio_lengths for length in sublist]
            for key, value in gt_audio_dict.items(): # round
                gt_audio_list = value
                rec_audio_list = rec_audio_dict[key]
                for i, (gt, rec) in enumerate(zip(gt_audio_list, rec_audio_list)): # bs
                    if i < len(all_lengths):
                        gt_audio_length = all_lengths[i]
                        gt = gt[:, :gt_audio_length]
                        rec = rec[:, :gt_audio_length]
                        id_1, id_2 = self.get_id(gt, rec)   
                        same_id_result_list = self.get_same_id(id_1, id_2)    
                        for idx,same_id in enumerate(same_id_result_list): # codebook
                            same_id_dict_round[key][f"codebook_{idx + 1}"].append("{:.2f}%".format((same_id / id_1.shape[2]) * 100))
            avg_same_id_dict = self.data_process(same_id_dict_round)
            self.plot_mrc_avg_same_id(avg_same_id_dict)
            return avg_same_id_dict
        
        elif task == "OS2":
            same_id_sums = [0] * self.num_codebooks
            total_samples = 0
            for batch in tqdm(self.dataloader, desc="dataloader audio"):
                gt_audio_test = batch["audio"].to(self.device)  # shape [B, T]
                all_gt_audio_lengths = batch["audio_length"]
                for i, gt_audio in enumerate(gt_audio_test):
                    gt_audio = gt_audio_test[i].to(self.device)
                    gt_audio_length = all_gt_audio_lengths[i]
                    gt_audio = gt_audio.unsqueeze(0)
                    gt_audio = gt_audio[:, :gt_audio_length]
                    os_audio = self.get_offset_2ms_audio(gt_audio, self.sample_rate)
                    gt_id, rec_id = self.get_id(gt_audio, os_audio)
                    same_id_results = self.get_same_id(gt_id, rec_id)
                    total_samples += 1
                    for j in range(self.num_codebooks):
                        same_id_sums[j] += (same_id_results[j] / gt_id.shape[2])
            percent_same_id_avg_list = [f"{(val / total_samples) * 100:.2f}%" for val in same_id_sums]
            self.plot_os2_avg_same_id(percent_same_id_avg_list)
            return f"codebook same id: {percent_same_id_avg_list}"

if __name__ == "__main__":
    """
    if task == "MRC"
        test sample_rate = codec's sample_rate
    elif task == "OS2":
        test sample_rate = example.wav's sample_rate
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, default="speechtokenizer")
    parser.add_argument(
        "--model_ckpt_dir",
        type=str,
        default="/sdb/model_weight/codec_evaluation/codec_ckpt/speechtokenizer",
    )
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--need_resample", type=bool, default=False)
    parser.add_argument("--task", type=str, default="MRC")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shift_time", type=int, default=2)
    parser.add_argument(
        "--dataset_audio_dir",
        type=str,
        default="/sdb/data1/speech/24kHz/LibriTTS/test-other",
    )
    parser.add_argument("--use_vocos", type=bool, default=False)
    parser.add_argument("--vocos_ckpt_dir", type=Optional[str], default=None)
    args = parser.parse_args()

    codec_eval = IDSensitiveEvaluation(
        codec_name=args.codec_name,
        model_ckpt_dir=args.model_ckpt_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        num_codebooks=args.num_codebooks,
        need_resample=args.need_resample,
        dataset_audio_dir=args.dataset_audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_vocos=args.use_vocos,
        vocos_ckpt_dir=args.vocos_ckpt_dir,
        shift_time=args.shift_time,
        task=args.task
    )
    result = codec_eval.evaluate(args.task)
    print(result)
