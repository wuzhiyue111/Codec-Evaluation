from codec_evaluation.codecs.dac import DAC
from codec_evaluation.codecs.encodec import Encodec
from codec_evaluation.codecs.mimi import Mimi
from codec_evaluation.codecs.semanticodec import SemantiCodec
from codec_evaluation.codecs.speechtokenizer import SpeechTokenizer
from codec_evaluation.codecs.wavlm_kmeans import WavLMKmeans
from codec_evaluation.codecs.wavtokenizer import WavTokenizer


def init_codec(modelname, sample_rate, mode, model_path, model_ckpt_dir, vocos_ckpt_dir, model_path_dir, device, freeze = False):
    """
        Codec initialization
        input:
            modelname: codecname
            sample_rate: The sample rate of the input audio
            mode: "quantized_emb" "unquantized_emb",etc.
            model_path: The path of the model
            model_ckpt_dir: The path of the model checkpoint
            vocos_ckpt_dir: The path of the vocos checkpoint
            model_path_dir: The path of the model path
            device: Select the device to use
            freeze: Whether to calculate the gradient
        return:
            model: Initialied codec

    """
    modes = ["encode", "decode", "reconstruct", "unquantized_emb","quantized_emb"]
    if mode not in modes:
        raise ValueError(f"Mode must be one of the following: {modes}")

    if modelname == 'dac':
        model = DAC(
            sample_rate=sample_rate, 
            orig_sample_rate=24000,
            mode=mode,
            num_codebooks=8,
            need_resample=True,
            model_path=model_path
        ).to(device)  
    elif modelname == 'encodec':
        model = Encodec(
            sample_rate=sample_rate, 
            orig_sample_rate=24000,
            mode=mode,
            num_codebooks=8,
            use_vocos=False,
            model_ckpt_dir=model_ckpt_dir,
            vocos_ckpt_dir=vocos_ckpt_dir,
            need_resample=True
        ).to(device)
    elif modelname == 'mimi':
        model = Mimi(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=8,
            model_ckpt_dir=model_ckpt_dir,
            need_resample=True
        ).to(device)
    elif modelname == 'semanticodec':
        model = SemantiCodec(
            sample_rate=sample_rate, 
            mode=mode,
            token_rate=100,
            semantic_vocab_size=8192,
            ddim_sample_step=50,
            cfg_scale=2.0,
            model_path_dir=model_path_dir,
            need_resample=True
        ).to(device)
    elif modelname =='speechtokenizer':
        model = SpeechTokenizer(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=8, 
            need_resample=True,
            model_ckpt_dir=model_ckpt_dir
        ).to(device)
    elif modelname =='wavlm_kmeans':
        model = WavLMKmeans(
            sample_rate=sample_rate, 
            mode=mode,
            layer_ids=(6,),
            need_resample=True
        ).to(device)
    elif modelname =='wavtokenizer':
        model = WavTokenizer(
            sample_rate=sample_rate, 
            need_resample=True,
            mode=mode,
            model_ckpt_dir=model_ckpt_dir,
            source="novateur/WavTokenizer-large-unify-40token",
            config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            checkpoint="wavtokenizer_large_unify_600_24k.ckpt"
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {modelname}")

    if freeze: 
        for name, params in model.named_parameters():
            params.requires_grad = False

    return model.eval()