from codec_evaluation.codecs.dac import DAC
from codec_evaluation.codecs.encodec import Encodec
from codec_evaluation.codecs.mimi import Mimi
from codec_evaluation.codecs.semanticodec import SemantiCodec
from codec_evaluation.codecs.speechtokenizer import SpeechTokenizer
from codec_evaluation.codecs.wavlm_kmeans import WavLMKmeans
from codec_evaluation.codecs.wavtokenizer import WavTokenizer

def init_codec(
        modelname: str, 
        sample_rate: int, 
        mode: str, 
        model_ckpt_dir: str,  
        device: str = 'cpu', 
        num_codebooks: int = 8,
        vocos_ckpt_dir: str | None = None,
        use_vocos: bool = False,
        freeze: bool = False,
        need_resample: bool = True,
        ):
    """
        Codec initialization
        input:
            modelname: codecname
            sample_rate: The sample rate of the input audio
            mode: "quantized_emb" "unquantized_emb",etc.
            model_ckpt_dir: The path of the model checkpoint
            device: Select the device to use
            num_codebooks: The number of codebooks
            freeze: Whether to calculate the gradient(Default is False)
            need_resample: Boolean, whether to resample the audio after decoding(Default is True)
        return:
            model: Initialied codec

    """
    modes = ["encode", "decode", "reconstruct", "unquantized_emb","quantized_emb"]
    if mode not in modes:
        raise ValueError(f"Mode must be one of the following: {modes}")

    if modelname == 'dac':
        model = DAC(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=num_codebooks,
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir
        ).to(device)  
    elif modelname == 'encodec':
        model = Encodec(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=num_codebooks,
            use_vocos=use_vocos,
            vocos_ckpt_dir=vocos_ckpt_dir,  
            model_ckpt_dir=model_ckpt_dir,
            need_resample=need_resample
        ).to(device)
    elif modelname == 'mimi':
        model = Mimi(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=num_codebooks,
            model_ckpt_dir=model_ckpt_dir,
            need_resample=need_resample
        ).to(device)
    elif modelname == 'semanticodec':
        model = SemantiCodec(
            sample_rate=sample_rate, 
            mode=mode,
            token_rate=100,
            semantic_vocab_size=8192,
            ddim_sample_step=50,
            cfg_scale=2.0,
            model_ckpt_dir=model_ckpt_dir,
            need_resample=need_resample
        ).to(device)
    elif modelname =='speechtokenizer':
        model = SpeechTokenizer(
            sample_rate=sample_rate, 
            mode=mode,
            num_codebooks=num_codebooks, 
            need_resample=need_resample,
            model_ckpt_dir=model_ckpt_dir
        ).to(device)
    elif modelname =='wavlm_kmeans':
        model = WavLMKmeans(
            sample_rate=sample_rate, 
            mode=mode,
            layer_ids=(6,),
            need_resample=need_resample
        ).to(device)
    elif modelname =='wavtokenizer':
        model = WavTokenizer(
            sample_rate=sample_rate, 
            need_resample=need_resample,
            mode=mode,
            model_ckpt_dir=model_ckpt_dir,
        ).to(device)
    else:
        raise ValueError(f"Invalid model name: {modelname}")

    if freeze: 
        for _, params in model.named_parameters():
            params.requires_grad = False

    return model.eval()