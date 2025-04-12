import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
import codec_evaluation
import json
import os

from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.print_config import print_config_tree

root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path=f"{root_path}/probe/config/LibriTTS_dataset/qwen2encoder_pca", config_name="qwen2encoder_pca_inference.yaml", version_base=None)
def main(config: DictConfig) -> None:
    print_config_tree(config)

    pl.seed_everything(config.seed)

    logger.info(f"Instantiating datamodule <{config.data._target_}>.")
    datamodule = hydra.utils.instantiate(config.data, _convert_="partial")

    logger.info(f"Instantiating model <{config.model._target_}>.")
    model = hydra.utils.instantiate(config.model, _convert_="partial")

    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating datamodule <{cb_conf._target_}>.")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks, 
        _convert_="partial",
    )

    logger.info(f"Loading checkpoint: {config.probe_ckpt_path}")
    if not os.path.exists(config.probe_ckpt_path):
        logger.error(f"Checkpoint file not found: {config.probe_ckpt_path}")
        return

    model = model.load_from_checkpoint(config.probe_ckpt_path)
    model.eval()

    logger.info("Starting testing...")
    trainer.test(model=model, datamodule=datamodule)
    
    if hasattr(model, 'test_metrics'):
        logger.info(f"Test WER: {model.test_metrics['wer']}")
        logger.info(f"Test CER: {model.test_metrics['cer']}")
        
        # Save results to file if specified
        if config.save_asr_result:
            os.makedirs(os.path.dirname(config.save_asr_result), exist_ok=True)
            with open(config.save_asr_result, 'w') as f:
                json.dump(model.test_metrics, f, indent=4)
            logger.info(f"Results saved to {config.save_asr_result}")
    else:
        logger.warning("No test metrics found in model.")

if __name__ == "__main__":
    main()
