import jiwer
import numpy as np
import torch
import torchaudio
import os
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from speechbrain.inference.speaker import EncoderClassifier

from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
import gc

def transform_text_list_for_wer(text_list):
    """
    Process text lists to eliminate the effects of capitalization and punctuation.

    Args:
        text_list: Text list

    Returns:
        list: Processed text list
    """
    def clean_text(text):
        # Convert to lower case
        text = text.lower()
        # Remove punctuation (you can add more if needed)
        for punct in ',.!?;:""\'"()[]{}、，。！？；：""' "【】《》-":
            text = text.replace(punct, " ")
        # Remove extra spaces
        text = " ".join(text.split())
        return text
    return [clean_text(text) for text in text_list]

def transform_text_list_for_cer(text_list):
    """
    Process text lists to eliminate the effects of capitalization and punctuation

    Args:
        text_list: Text list

    Returns:
        list: Processed text list, each character is separated by a space
    """
    def clean_text(text):
        # Convert to lower case
        text = text.lower()
        # Remove punctuation (you can add more if needed)
        for punct in ',.!?;:""\'"()[]{}、，。！？；：""' "【】《》-":
            text = text.replace(punct, " ")
        # Remove extra spaces
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

def wer(gt_audio = None, rec_audio = None, gt_text = None, processor = None, model = None, device = None, sample_rate = 24000):
    gt_transcription = None
    rec_transcription = None
    if gt_audio is not None and gt_text is not None:
        gt_transcription = asr(audios=gt_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)
    if rec_audio is not None and gt_text is not None:
        rec_transcription = asr(audios=rec_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)

    if gt_transcription is not None or rec_transcription is None:
        try:  # if no words are predicted
            wer_gt = None
            wer_rec = None
            gt_text_clean = transform_text_list_for_wer(gt_text)

            if gt_transcription is not None:
                gt_transcription_clean = transform_text_list_for_wer(gt_transcription)
                wer_gt = jiwer.wer(reference=gt_text_clean, hypothesis=gt_transcription_clean)

            if rec_transcription is not None:
                rec_transcription_clean = transform_text_list_for_wer(rec_transcription)
                wer_rec = jiwer.wer(reference=gt_text_clean, hypothesis=rec_transcription_clean)
        except ValueError:
            wer_gt = None
            wer_rec = None
    else:
        wer_gt = None
        wer_rec = None
    return wer_gt, wer_rec

def cer(gt_audio = None, rec_audio = None, gt_text = None, processor = None, model = None, device = None, sample_rate = 24000):
    gt_transcription = None
    rec_transcription = None
    if gt_audio is not None and gt_text is not None:
        gt_transcription = asr(audios=gt_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)
    if rec_audio is not None and gt_text is not None:
        rec_transcription = asr(audios=rec_audio, processor=processor, model=model, sample_rate=sample_rate, device=device)

    if gt_transcription is not None or rec_transcription is None:
        try:  # if no words are predicted
            cer_gt = None
            cer_rec = None
            gt_text_clean = transform_text_list_for_cer(gt_text)
            if gt_transcription is not None:
                gt_transcription_clean = transform_text_list_for_cer(gt_transcription)
                cer_gt = jiwer.cer(reference=gt_text_clean, hypothesis=gt_transcription_clean)
            if rec_transcription is not None:
                rec_transcription_clean = transform_text_list_for_cer(rec_transcription)
                cer_rec = jiwer.cer(reference=gt_text_clean, hypothesis=rec_transcription_clean)
        except ValueError:
            cer_gt = None
            cer_rec = None
    else:
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
):
    # gt_audio: [B, T]
    # rec_audio: [B, T]
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

    # PESQ requires a sampling rate of 16k or 8k
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

def calculate_visqol(gt_audio: torch.Tensor, rec_audio: torch.Tensor, sample_rate=24000, visqol_mode = "speech"):
    # gt_audio: [B, T]
    # rec_audio: [B, T]
    # mode = "speech" or "audio", "audio" is for music, "speech" is for speech, audio sample rate is 48k, speech sample rate is 16k

    assert visqol_mode in ["speech", "audio"], f"visqol_mode must be 'speech' or 'audio', but got {visqol_mode}"
    config = visqol_config_pb2.VisqolConfig()
    if visqol_mode == "audio":
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    else:
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"

    if sample_rate != config.audio.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, config.audio.sample_rate).to(gt_audio.device)
        gt_audio = resampler(gt_audio)
        rec_audio = resampler(rec_audio)

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    visqol_scores = []

    for i in range(gt_audio.shape[0]):
        ref = gt_audio[i].cpu().numpy().astype(np.float64)
        deg = rec_audio[i].cpu().numpy().astype(np.float64)
        result = api.Measure(ref, deg)
        visqol_scores.append(result.moslqo)

    if len(visqol_scores) == 0:
        return 0.0
    return sum(visqol_scores) / len(visqol_scores)


def calculate_mel_distance(gt_audio: torch.Tensor, rec_audio: torch.Tensor, sample_rate=24000):
    """
    gt_audio, rec_audio: [B, T]
    """
    # Reference dac
    window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    num_mels = [5, 10, 20, 40, 80, 160, 320]

    # Pre-built MelSpectrograms of different scales
    mel_transforms = [
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            n_mels=n_mels
        ).to(gt_audio.device)
        for n_fft, n_mels in zip(window_lengths, num_mels)
    ]

    B = gt_audio.shape[0]
    msd_list = []

    for b in range(B):
        gt_wav  = gt_audio[b].unsqueeze(0)  # [1, T]
        rec_wav = rec_audio[b].unsqueeze(0)

        total_l1 = 0.0
        num_scales = len(mel_transforms)
        for mel_spec in mel_transforms:
            # Convert waveform to Mel spectrum
            gt_mel  = mel_spec(gt_wav)
            rec_mel = mel_spec(rec_wav)

            # First, clamp to ensure that the amplitude is not less than 1e-5 to avoid log(0).
            # Then take the log and get the logarithmic power spectrum
            gt_db = torch.log(torch.clamp(gt_mel, min=1e-5))
            rec_db = torch.log(torch.clamp(rec_mel, min=1e-5))

            total_l1 += torch.mean(torch.abs(gt_db - rec_db))

        mean_l1 = (total_l1 / num_scales).item()
        msd_list.append(mean_l1)

    return sum(msd_list) / len(msd_list)
