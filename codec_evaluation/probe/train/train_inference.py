import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import codec_evaluation
import argparse
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.utils import find_lastest_ckpt
from codec_evaluation.utils.print_config import print_config_tree

root_path = codec_evaluation.__path__[0]

def main(dataset_name, config_name):
    with hydra.initialize_config_dir(
        config_dir=f"{root_path}/probe/config/{dataset_name}",
        version_base=None
    ):
        config = hydra.compose(config_name = config_name)

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
            _convert_="partial")

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

    checkpoint = torch.load(latest_ckpt_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    logger.info(f"start_Testing, latest_ckpt_path: {latest_ckpt_path}")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    logger.info("testing_finished")
    logger.info(f"{model.test_step_outputs=}")

    # 保存结果
    # if config.save_asr_result is not None:
    #     with open(f"{config.save_asr_result}", "w") as f:
    #         for r in model.test_step_outputs["result"]:
    #             json.dump(r, f, indent=4)
    #             f.write("\n")

if __name__ == "__main__":
    logger = RankedLogger(__name__, rank_zero_only=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="EMO_dataset", help='dataset name')
    parser.add_argument('--config_name', type=str, default="dac", help='congig name')

    args = parser.parse_args()
    main(args.dataset_name, args.config_name)