import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import codec_evaluation
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.print_config import print_config_tree
from codec_evaluation.utils.utils import find_lastest_ckpt
import os

codec_evaluation_root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

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

    logger.info("Instantiating tensorboard_logger...")
    tensorboard_logger = hydra.utils.instantiate(config.tensorboard_logger, _convert_="partial")

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks, 
        logger=tensorboard_logger, 
        _convert_="partial",
    )

    latest_ckpt_path = find_lastest_ckpt(config.get("lm_ckpt_dir", None))
    logger.info(f"start_training, latest_ckpt_path: {latest_ckpt_path}")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=latest_ckpt_path,
    )
    logger.info("training_finished")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec_name", type=str, required=True)
    parser.add_argument("--base_audio_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--codec_ckpt_dir", type=str, required=True)
    parser.add_argument("--ppl_ckpt_dir", type=str, required=True)
    parser.add_argument("--tensorboard_save_dir", type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(f"{codec_evaluation_root_path}/perplexity/config/{args.codec_name}_ppl.yaml")
    config.ppl_ckpt_dir = args.ppl_ckpt_dir
    config.tensorboard_save_dir = args.tensorboard_save_dir
    config.codec_name = args.codec_name
    config.data.base_audio_dir = args.base_audio_dir
    config.data.dataset_path = args.dataset_path
    config.codec_ckpt_dir = args.codec_ckpt_dir
    config.model.ppl_model_config.pretrained_model_name_or_path = os.path.join(codec_evaluation_root_path, config.model.ppl_model_config.pretrained_model_name_or_path)

    main(config)