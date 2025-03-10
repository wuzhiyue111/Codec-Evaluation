import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import codec_evaluation
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.print_config import print_config_tree
import json
import torch

root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path=f"{root_path}/probe/config/LibriTTS_dataset/mimi", config_name="mimi_inference.yaml", version_base=None) # 需要更改
def main(config: DictConfig) -> None:

    print_config_tree(config)
    pl.seed_everything(config.seed)

    logger.info(f"Instantiating datamodule <{config.data._target_}>.")
    datamodule = hydra.utils.instantiate(config.data, _convert_="partial")

    logger.info(f"Instantiating model <{config.model._target_}>.")
    model = hydra.utils.instantiate(config.model, _convert_="partial")
    model.load_state_dict(torch.load(config.probe_ckpt_path)['state_dict'], strict=False)
    model.eval()

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
        use_distributed_sampler=False, # Custom bucket sampler, the use_distributed_sampler need to be set to False
    )

    logger.info(f"start_testing")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    logger.info("testing_finished")
    logger.info(f"wer: {model.test_step_outputs['wer']}")
    logger.info(f"cer: {model.test_step_outputs['cer']}")

    # 保存结果
    if config.save_asr_result is not None:
        with open(f"{config.save_asr_result}", "w") as f:
            for r in model.test_step_outputs["result"]:
                json.dump(r, f, indent=4)
                f.write("\n")

if __name__ == "__main__":
    main()