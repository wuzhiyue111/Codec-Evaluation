import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import codec_evaluation
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.print_config import print_config_tree
from codec_evaluation.utils.utils import find_lastest_ckpt
from codec_evaluation.probe.dataset.MELD_dataset.MELD_dataset import MELDdataModule

root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path=f"{root_path}/probe/config/MELD_dataset", config_name="encodec.yaml", version_base=None) # 需要更改
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
    tensorboard_logger = hydra.utils.instantiate(config.tensorboard, _convert_="partial")

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks, 
        logger=tensorboard_logger, 
        _convert_="partial" # Custom bucket sampler, the use_distributed_sampler need to be set to False
    )

    
    latest_ckpt_path = None
    logger.info(f"start_training, latest_ckpt_path: {latest_ckpt_path}")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=latest_ckpt_path,
    )
    logger.info("training_finished")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    latest_ckpt_path = find_lastest_ckpt(config.get("probe_ckpt_dir", None))
    if latest_ckpt_path is None:
        logger.error("No checkpoint found for testing!")
        return

    logger.info(f"start_Testing, latest_ckpt_path: {latest_ckpt_path}")
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=latest_ckpt_path,
    )
    logger.info("test_finished!")
    logger.info(f"acc: {model.test_step_outputs['acc']}")
    # # 保存结果
    # if config.save_asr_result is not None:
    #     with open(f"{config.save_asr_result}", "w") as f:
    #         for r in model.test_step_outputs["acc"]:
    #             json.dump(r, f, indent=4)
    #             f.write("\n")

if __name__ == "__main__":
    main()