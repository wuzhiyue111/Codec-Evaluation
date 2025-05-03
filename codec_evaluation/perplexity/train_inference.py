import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import codec_evaluation
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.print_config import print_config_tree
from codec_evaluation.utils.utils import find_lastest_ckpt

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
    parser.add_argument("--codec_name", type=str, default="mimi")
    args = parser.parse_args()
    config = OmegaConf.load(f"{codec_evaluation_root_path}/perplexity/config/{args.codec_name}_ppl.yaml")
    main(config)