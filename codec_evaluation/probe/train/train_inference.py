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
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch


root_path = codec_evaluation.__path__[0]
logger = RankedLogger(__name__, rank_zero_only=True)
logging.basicConfig(level=logging.INFO)


def main(config: DictConfig, output_file: str):

    print_config_tree(config, resolve=True)

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
                                                    _convert_="partial")

    logger.info(f"Instantiating trainer <{config.trainer._target_}>.")
    trainer = hydra.utils.instantiate(
        config.trainer, 
        callbacks=callbacks, 
        logger=tensorboard_logger,
        _convert_="partial")

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=None,
    )
    logger.info("Training finished")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    latest_ckpt_path = find_lastest_ckpt(config.probe_ckpt_dir)
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

    # save results
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

                # Convert the result to key: value format
                if not isinstance(result, dict):
                    raise ValueError("Result is not a dictionary")

                for key, value in result.items():
                    # If it is a tensor, take out the item
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    cur_line = f"{key}: {value}"
                    f.write(cur_line)
                    f.write("\n")
                    logger.info(cur_line)

                logger.info(f"Save result to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save result to {output_file}: {e}")

def cli():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help=f'The dataset name is the same as the folder name under codec_evaluation.probe.config, such as "EMO_dataset", "GS_dataset"')
    
    parser.add_argument('--model_name',
                        type=str, 
                        required=True,
                        help=f'Model name, such as "speechtokenizer"')

    parser.add_argument('--mode',
                        type=str,
                        default="quantized_emb",
                        choices=["unquantized_emb", "quantized_emb", "encode"],
                        help=f'Mode')

    parser.add_argument("--codec_ckpt_dir",
                        type=str,
                        required=True,
                        help=f'Codec checkpoint dir')

    parser.add_argument('--devices',
                        type=str,
                        default="0,",
                        help=f'Devices, e.g. "1" (gpu count), "0,1,2,3" (gpu ids)')

    args, _ = parser.parse_known_args()
    parser.add_argument('--weights_save_dir',
                        type=str,
                        default=os.path.join(root_path, 
                                             "probe", 
                                             f"codec_eval_probe_{args.model_name}_{args.mode}_{args.dataset_name}"),
                        help=f'Weights save dir')

    parser.add_argument("--tensorboard_save_dir",
                        type=str,
                        default=os.path.join(root_path, 
                                             "probe", 
                                             f"codec_eval_probe_tb_log_{args.model_name}_{args.mode}_{args.dataset_name}"),
                        help=f'Tensorboard save dir')

    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help=f'Path to the huggingface format dataset.(e.g., /sdb/data1/huggingface_dataset/GTZAN/GTZAN_dataset/GTZAN_train_dataset)')

    parser.add_argument("--base_audio_dir",
                        type=str,
                        required=True,
                        help=f'The root directory where the raw audio files are stored.(Used to splice the complete audio path).')

    parser.add_argument('overrides',
                        nargs='*',
                        help='Any key=value arguments to override config values (e.g., model.tokenizer.pretrained_model_name_or_path=/sdb/model_weight/s2t-small-librispeech-asr).')

    default_output_file = os.path.join(root_path, 
                                       "probe", 
                                       "probe_result", 
                                       f"codec_eval_probe_result_{args.model_name}_{args.mode}_{args.dataset_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    os.makedirs(args.weights_save_dir, exist_ok=True)
    os.makedirs(args.tensorboard_save_dir, exist_ok=True)
    os.makedirs(os.path.join(root_path, "probe", "probe_result"), exist_ok=True)

    
    parser.add_argument('--output_file',
                        type=str, 
                        default=default_output_file,
                        help=f'Output file, default: {default_output_file}')
    
    # Parse all parameters
    args = parser.parse_args()
    
    # Print selected configuration information
    config_name = os.path.join(root_path, "probe", "config", args.dataset_name, f"{args.model_name}.yaml")
    logger.info(f"Selected config: {config_name}")
    config = OmegaConf.load(config_name)
    config.probe_ckpt_dir = args.weights_save_dir
    config.trainer.devices = [int(d) for d in args.devices.split(',') if d]
    config.data.dataset_path = args.dataset_path
    config.data.base_audio_dir = args.base_audio_dir
    config.mode = args.mode
    config.model.model_ckpt_dir = args.codec_ckpt_dir
    config.tensorboard.save_dir = args.tensorboard_save_dir

    main(config=config,
         output_file=args.output_file)

if __name__ == "__main__":
    cli()
