import argparse
import logging
import os
from pathlib import Path
import time

import codec_evaluation
from codec_evaluation.utils.logger import RankedLogger
from codec_evaluation.utils.utils import find_lastest_ckpt
from codec_evaluation.utils.print_config import print_config_tree
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch


root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)
logging.basicConfig(level=logging.INFO)

if os.environ.get("DEBUG", "false").lower() == "true":
    logger.setLevel(logging.DEBUG)


def main(dataset_name: str,
         config_name: str,
         mode: str,
         devices: str,
         probe_ckpt_dir: str,
         tensorboard_save_dir: str,
         output_file: str):
    with hydra.initialize_config_dir(
        config_dir=f"{root_path}/probe/config/{dataset_name}",
        version_base=None
    ):
        config: DictConfig = hydra.compose(config_name=config_name,
                                           overrides=[f"mode={mode}",
                                                      f"probe_ckpt_dir={probe_ckpt_dir}"])

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
                    logger.info(f"Instantiating callback <{cb_conf._target_}>.")
                    callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

        logger.info("Instantiating tensorboard_logger...")
        tensorboard_logger = hydra.utils.instantiate(config.tensorboard,
                                                     save_dir=tensorboard_save_dir,
                                                     _convert_="partial")

        logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
        trainer = hydra.utils.instantiate(
            config.trainer, 
            callbacks=callbacks, 
            logger=tensorboard_logger, 
            devices=devices,
            _convert_="partial")

    latest_ckpt_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    latest_ckpt_path = find_lastest_ckpt(config.get("probe_ckpt_dir", None))
    if latest_ckpt_path is None:
        logger.error("No checkpoint found for testing!")
        return

    checkpoint = None
    try:
        with open(latest_ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    logger.info(f"Testing start, using ckpt: {latest_ckpt_path}")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    logger.info("Testing finished")

    # 保存结果
    if output_file is not None:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(f"{output_file}", "w") as f:
                result = None
                if "result" in model.test_step_outputs:
                    result = model.test_step_outputs["result"]
                else:
                    result = model.test_step_outputs
                if result is None:
                    raise ValueError("No result found in model.test_step_outputs")

                # 将结果转换为 key: value 格式
                if not isinstance(result, dict):
                    raise ValueError("Result is not a dictionary")

                for key, value in result.items():
                    # 如果是tensor，取出item
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    cur_line = f"{key}: {value}"
                    f.write(cur_line)
                    logger.info(cur_line)

                logger.info(f"Save result to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save result to {output_file}: {e}")

def cli():
    # 获取所有数据集目录
    dataset_choices = sorted([d.name.replace("_dataset", "") for d in (Path(root_path) / "probe" / "dataset").iterdir() if d.is_dir()])
    
    parser = argparse.ArgumentParser()
    
    # 第一步：只添加 dataset_name 参数
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help=f'Dataset name', 
                        choices=dataset_choices)
    
    # 首先解析 dataset_name
    args, _ = parser.parse_known_args()

    dataset_name = args.dataset_name + "_dataset"
    
    # 根据选择的数据集获取对应的配置文件选项
    config_path = Path(root_path) / "probe" / "config" / dataset_name
    config_choices = sorted([f.stem for f in config_path.glob("*.yaml")])  # 只获取 yaml 文件
    
    # 第二步：添加其他参数
    if not config_choices:
        raise ValueError(f"No config files found for dataset {dataset_name}")
    
    parser.add_argument('--config_name',
                        type=str, 
                        required=True,
                        help=f'Config name',
                        choices=config_choices)
    
    args, _ = parser.parse_known_args()

    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        choices=["encode", "decode", "reconstruct", "unquantized_emb", "quantized_emb"],
                        help=f'Mode')
    
    args, _ = parser.parse_known_args()

    parser.add_argument('--devices',
                        type=str,
                        default="0,",
                        help=f'Devices, e.g. "1" (gpu count), "0,1,2,3" (gpu ids)')
    
    parser.add_argument('--probe_ckpt_dir',
                        type=str,
                        default=f"probe/probe_ckpt/{dataset_name}/{args.config_name}/{args.mode}",
                        help=f'Probe ckpt dir')
    
    parser.add_argument("--tensorboard_save_dir",
                        type=str,
                        default=f"probe/probe_tb_log/{dataset_name}/{args.config_name}/{args.mode}",
                        help=f'Tensorboard save dir')
    
    default_output_file = f"probe/probe_result/{dataset_name}/{args.config_name}/{args.mode}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    parser.add_argument('--output_file', 
                        type=str, 
                        default=default_output_file,
                        help=f'Output file, default: {default_output_file}')
    
    # 解析所有参数
    args = parser.parse_args()
    
    # 打印选择的配置信息
    logger.info(f"Selected dataset: {args.dataset_name}")
    logger.info(f"Available configs for this dataset: {config_choices}")
    logger.info(f"Selected config: {args.config_name}")
    
    main(dataset_name,
         args.config_name,
         args.mode,
         args.devices,
         args.probe_ckpt_dir,
         args.tensorboard_save_dir,
         args.output_file)

if __name__ == "__main__":
    cli()
