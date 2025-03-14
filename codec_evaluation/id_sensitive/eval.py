from codec_evaluation.init_codecs import init_codec
import torchaudio
import random
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from codec_evaluation.probe.dataset.LibriTTS_dataset.libritts_ctc import (
    LibriTTS_ctc_dataset,
)
from codec_evaluation.id_sensitive.utils import (
    longest_common_substring_strict,
)
from pandas import DataFrame

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
        batch_size: int = 32,
        num_workers: int = 8,
        shift_time: int = 2,
        save_pandas_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        codec_name: codec model name
        model_ckpt_dir: codec model checkpoint directory
        sample_rate: audio sample rate
        mode_encode: codec mode for encoding
        mode_reconstruct: codec mode for reconstruction
        num_codebooks: number of codebooks
        need_resample: boolean, whether to resample the audio after decoding
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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

        self.codec_name = codec_name
        self.codec_sample_rate = self.codec_encode.orig_sample_rate
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = device
        self.shift_time = shift_time
        self.save_pandas_dir = save_pandas_dir
        dataset = LibriTTS_ctc_dataset(audio_dir=dataset_audio_dir)
        dataset = torch.utils.data.Subset(dataset, range(1200))
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=dataset.dataset.collate_fn,
        )

    # def get_offset_2ms_audio(self, audio_tensor, sample_rate):
    #     y = audio_tensor.cpu().numpy()
    #     shift_samples = int(sample_rate * 2 / 1000)
    #     y_shifted = np.roll(y, shift_samples, axis=1)
    #     y_shifted[:, :shift_samples] = 0
    #     y_shifted = torch.from_numpy(y_shifted).to(self.device)
    #     return y_shifted

    def get_offset_2ms_audio(self, audio_tensor, sample_rate):
        shift_samples = int(sample_rate * self.shift_time / 1000)
        y_shifted = torch.roll(audio_tensor, shifts=shift_samples, dims=1)
        y_shifted[:, :shift_samples] = 0
        return y_shifted

    def get_same_id(self, id_1, id_2):
        results = []
        for i in range(id_1.shape[1]):
            same = (id_1[0, i, :] == id_2[0, i, :]).sum().item()
            results.append(same)
        return results

    def get_lcs(self, emb1_id, emb2_id):
        lcs_lengths = []
        for i in range(emb1_id.shape[1]):
            emb1_codebook = emb1_id[0, i, :].tolist()
            emb2_codebook = emb2_id[0, i, :].tolist()
            max_length = longest_common_substring_strict(emb1_codebook, emb2_codebook)
            lcs_lengths.append(max_length)
        return lcs_lengths

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

    def data_process(self, same_id_dict_round, lcs_dict_round):
        round_list = [f'round_{i+1}' for i in range(10)]

        # avg 
        avg_same_id_dict = {} # {round_1: [avg_codebook_1, avg_codebook_2, ...], round_2: [], ...}
        avg_lcs_dict = {} # {round_1: [avg_codebook_1, avg_codebook_2, ...], round_2: [], ...}
        for i in range(10): # round
            avg_same_id_dict[round_list[i]] = []
            avg_lcs_dict[round_list[i]] = []

        for key, value_same_id_dict in same_id_dict_round.items(): # round
            for _, value_same_id in value_same_id_dict.items(): # codebook
                if value_same_id != []:
                    avg_same_id_dict[key].append(sum(value_same_id) / len(value_same_id))

        for key, value_lcs_dict in lcs_dict_round.items(): # round
            for _, value_lcs in value_lcs_dict.items(): # codebook
                if value_lcs != []:
                    avg_lcs_dict[key].append(sum(value_lcs) / len(value_lcs))

        data = {
            'round': [],  # 存储轮次（1-10）
            'codec': [],  # 存储模型名称（dac, encodec等）
            'average_same_id': []  # 存储每轮对应的平均值
        }
        df = DataFrame(data)
        if self.save_pandas_dir is not None:
            df.to_csv(self.save_pandas_dir, index=False)

    def plot_avg_same_id(self, df):
        pivot_df = df.pivot_table(values='average_same_id', index='round', columns='codec')
        # 绘制柱状图
        plt.figure(figsize=(14, 8))
        pivot_df.plot(kind='bar', width=0.8)

        # 美化图表
        plt.xlabel('Reconstruction Round (Round_1 to Round_10)', fontsize=12)
        plt.ylabel('Average Same Codebook ID Count', fontsize=12)
        plt.title('Codebook Same ID Count Across Rounds by Codec', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Codec Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("avg_same_id_results.png")
        

    @torch.inference_mode
    def evaluate(self, task):
        if task == "MRC":
            gt_audio_dict = {}
            rec_audio_dict = {}
            for i in range(10):
                gt_audio_dict[f"round_{i + 1}"] = []
                rec_audio_dict[f"round_{i + 1}"] = []

            print(f"dataset length: {len(self.dataloader)}, now start to reconstruct")
            for batch in tqdm(self.dataloader, desc="reconstruct audio"):
                gt_audio_test = batch["audio"].clone().to(self.device)  # shape [B, T]
                # resample to codec sample rate
                if self.codec_sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, self.codec_sample_rate
                    ).to(self.device)
                    gt_audio_test = resampler(gt_audio_test)
                max_length = gt_audio_test.shape[-1]

                gt_audio = batch["audio"].to(self.device)

                rec_audio_dict, gt_audio_dict = self.multi_round_reconstruction(
                    gt_audio_clone=gt_audio_test,
                    gt_audio=gt_audio,
                    max_length=max_length,
                    rec_audio_dict=rec_audio_dict,
                    gt_audio_dict=gt_audio_dict,
                )
                break

            torch.cuda.empty_cache()
            print(f"reconstruct done!")

            same_id_dict_round = {}
            lcs_dict_round = {}
            for i in range(10): # round
                same_id_dict_round[f"round_{i + 1}"] = {}
                lcs_dict_round[f"round_{i + 1}"] = {}
                for j in range(8): # codebook
                    same_id_dict_round[f"round_{i + 1}"][f"codebook_{j + 1}"] = []
                    lcs_dict_round[f"round_{i + 1}"][f"codebook_{j + 1}"] = []

            for key, value in gt_audio_dict.items(): # round
                gt_audio_list = value
                rec_audio_list = rec_audio_dict[key]
                for gt, rec in zip(gt_audio_list, rec_audio_list): # bs
                    id_1, id_2 = self.get_id(gt, rec)
                    same_id_result_list = self.get_same_id(id_1, id_2)
                    lcs_results_list = self.get_lcs(id_1, id_2)
                    for idx, (same_id, same_lcs) in enumerate(zip(same_id_result_list, lcs_results_list)): # codebook
                        same_id_dict_round[key][f"codebook_{idx + 1}"].append(same_id)
                        lcs_dict_round[key][f"codebook_{idx + 1}"].append(same_lcs)


            


            plt.figure(figsize=(10, 6))
            plt.plot(same_id_list, marker="o")
            plt.title("Same ID Results")
            plt.xlabel("Round")
            plt.ylabel("Same ID Value")
            plt.grid(True)
            plt.savefig("same_id_results.png")
            plt.close()

            # 绘制 LCS Results 图表
            plt.figure(figsize=(10, 6))
            plt.plot(lcs_list, marker="o")
            plt.title("LCS Results")
            plt.xlabel("Round")
            plt.ylabel("LCS Value")
            plt.grid(True)
            plt.savefig("lcs_results.png")
            plt.close()

            return {
                "task": task,
                "same_id_list": same_id_list,
                "lcs_list": lcs_list,
            }

        elif task == "OB2":
            same_id_list = []
            lcs_list = []
            for batch in tqdm(self.dataloader, desc="dataloader audio"):
                gt_audio_test = batch["audio"].clone()
                if self.sample_rate != self.codec_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.sample_rate, self.codec_sample_rate
                    ).to(self.device)
                    gt_audio_test = resampler(gt_audio_test)

                for i in range(len(gt_audio_test)):
                    sig1 = gt_audio_test[i].to(self.device)
                    sig1 = sig1.unsqueeze(0)
                    sig2 = self.get_offset_2ms_audio(sig1, self.sample_rate)
                    id_1, id_2 = self.get_id(sig1, sig2)
                    same_id_results = self.get_same_id(id_1, id_2)
                    lcs_results = self.get_lcs(id_1, id_2)
                    same_id_list.append(same_id_results)
                    lcs_list.append(lcs_results)
                print("Same ID Results:", same_id_results)
                print("LCS Results:", lcs_results)

            return {
                "task": task,
                "same_id_list": same_id_list,
                "lcs_list": lcs_list,
            }


if __name__ == "__main__":
    """
    if task == "MRC":
        test sample_rate = codec's sample_rate
    elif task == "OB2":
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=24000)
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
    parser.add_argument("--save_pandas_dir", type=Optional[str], default=None)
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
        save_pandas_dir=args.save_pandas_dir,
    )
    result = codec_eval.evaluate(args.task)
    print(result)
