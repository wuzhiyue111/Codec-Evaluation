import jiwer
import numpy as np
import torch
import torchaudio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

def transform_text_list_for_wer(text_list):
    """
    处理文本列表，消除大小写和标点符号的影响

    Args:
        text_list: 文本列表

    Returns:
        list: 处理后的文本列表
    """
    def clean_text(text):
        # 转小写
        text = text.lower()
        # 移除标点符号 (可以根据需要添加更多标点符号)
        for punct in ',.!?;:""\'"()[]{}、，。！？；：""' "【】《》-":
            text = text.replace(punct, " ")
        # 移除多余空格
        text = " ".join(text.split())
        return text
    return [clean_text(text) for text in text_list]

def transform_text_list_for_cer(text_list):
    """
    处理文本列表，消除大小写和标点符号的影响

    Args:
        text_list: 文本列表

    Returns:
        list: 处理后的文本列表, 每个字符之间都有一个空格
    """
    def clean_text(text):
        # 转小写
        text = text.lower()
        # 移除标点符号 (可以根据需要添加更多标点符号)
        for punct in ',.!?;:""\'"()[]{}、，。！？；：""' "【】《》-":
            text = text.replace(punct, " ")
        # 移除多余空格
        text = "".join(text.split())
        return text
    return [clean_text(text) for text in text_list]

def asr(audios, processor, model, sample_rate, device):
    # audios: [B, T]
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        audios = resampler(audios)
    input_features = processor(
        [audio.numpy() for audio in audios.cpu()],
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    predicted_ids = model.generate(input_features.to(device))
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

def wer(gt_audio, rec_audio, gt_text, processor, model, device, sample_rate=24000):

    gt_transcription = asr(audios=gt_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)
    rec_transcription = asr(audios=rec_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)

    try:  # if no words are predicted
        gt_transcription_clean = transform_text_list_for_wer(gt_transcription)
        rec_transcription_clean = transform_text_list_for_wer(rec_transcription)
        gt_text_clean = transform_text_list_for_wer(gt_text)
        wer_gt = jiwer.wer(reference=gt_text_clean, hypothesis=gt_transcription_clean)
        wer_rec = jiwer.wer(reference=gt_text_clean, hypothesis=rec_transcription_clean)
    except ValueError:
        wer_gt = None
        wer_rec = None
    return wer_gt, wer_rec

def cer(gt_audio, rec_audio, gt_text, processor, model, device, sample_rate=24000):
    gt_transcription = asr(audios=gt_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)
    rec_transcription = asr(audios=rec_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)

    try:  # if no words are predicted
        gt_transcription_clean = transform_text_list_for_cer(gt_transcription)
        rec_transcription_clean = transform_text_list_for_cer(rec_transcription)
        gt_text_clean = transform_text_list_for_cer(gt_text)
        cer_gt = jiwer.cer(reference=gt_text_clean, hypothesis=gt_transcription_clean)
        cer_rec = jiwer.cer(reference=gt_text_clean, hypothesis=rec_transcription_clean)
    except ValueError:
        cer_gt = None
        cer_rec = None

    return cer_gt, cer_rec


def calculate_stoi(gt_audio: torch.Tensor, rec_audio: torch.Tensor, sample_rate=24000):
    stoi = ShortTimeObjectiveIntelligibility(sample_rate).to(gt_audio.device)
    return stoi(rec_audio, gt_audio).item()


def calculate_spk_sim(
    gt_audio: torch.Tensor,
    rec_audio: torch.Tensor,
    model: EncoderClassifier,
    sample_rate: int = 24000,
):
    # gt_audio: [B, T]
    # rec_audio: [B, T]
    # if sample_rate != 16000:
    #     resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
    #     gt_audio = resampler(gt_audio)
    #     rec_audio = resampler(rec_audio)

    gt_embedding = model.encode_batch(gt_audio)
    rec_embedding = model.encode_batch(rec_audio)

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_sim(gt_embedding, rec_embedding)

    return similarity.mean().item()


def compute_codebook_usage(all_codes: torch.Tensor, audio_mask: torch.Tensor | None):
    """
    all_codes: torch.tensor.shape = [B, codebooks, T]
    audio_mask: torch.tensor.shape = [B, T]
        if audio_mask is None, then codes_mask is all ones
    """
    if audio_mask is None:
        codes_mask = torch.ones(size=(all_codes.shape[0], all_codes.shape[2])).to(all_codes.device)
    else:
        codes_mask = torch.nn.functional.interpolate(audio_mask, size=(all_codes.shape[-1],), mode="nearest").squeeze(1)

    with torch.no_grad():
        entropy = []
        for codebook_id in range(all_codes.shape[1]):
            codes_ = all_codes[:, codebook_id, :]
            counts = torch.bincount(codes_[codes_mask == 1])
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))
        return entropy


def calculate_pesq(gt_audio: torch.Tensor, rec_audio: torch.Tensor, sample_rate=24000):
    # gt_audio: [B, T]
    # rec_audio: [B, T]
    pesq = PerceptualEvaluationSpeechQuality(16000, "wb")

    # PESQ要求采样率为16k或8k
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(gt_audio.device)
        gt_audio = resampler(gt_audio)
        rec_audio = resampler(rec_audio)

    gt_audio_cal = gt_audio.clone()
    rec_audio_cal = rec_audio.clone()
    assert gt_audio_cal.shape == rec_audio_cal.shape

    pesq_list = []
    for i in range(gt_audio_cal.shape[0]):
        try:
            pesq_list.append(pesq(rec_audio_cal[i], gt_audio_cal[i]))
        except Exception as e:
            print(f"gt_audio.shape = {gt_audio_cal.shape}, rec_audio.shape = {rec_audio_cal.shape},")
            print(f"pesq error: {e}")
    return sum(pesq_list) / len(pesq_list)