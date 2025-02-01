from argparse import ArgumentParser
from pathlib import Path

import yaml

import benchmark as bench
from benchmark.utils.config_utils import add_constructors
from benchmark.utils.parser_utils import add_extract_args
from benchmark.utils.parser_utils import add_probe_args
from benchmark.utils.parser_utils import add_finetune_args
from benchmark.extract import main as extract_main
from benchmark.probe import main as probe_main
from benchmark.finetune import main as finetune_main

# Add all the constructors we need for yaml files.
add_constructors()

parser = ArgumentParser(
    description='MIR-Benchmark: Benchmarking Music Representation Learning Models',
)
parser.set_defaults(
    main_func=lambda _: parser.print_help()
)

# We have lots of main scripts to be executed.
subparsers = parser.add_subparsers()

################################
# Handling Feature Extraction #
################################
run_extract_parser = subparsers.add_parser(
    'extract',
    help=f'''Extract pretrain / handcrafted features from audio dir and save to output dir. 
    You need to specify a config file for the extraction. 
    See example in `benchmark/tasks/GTZAN/GTZAN_base_config.yaml`.
    You can define your own config file, but it must follow the same structure as the example.
    Currently supported representations: {bench.constants.model_constants.SUPPORTED_REPRESENTATIONS}.
    Run `python . extract -h` for more details.
    ''',
)
run_extract_parser = add_extract_args(run_extract_parser)
run_extract_parser.set_defaults(
    main_func=extract_main,
)

################################
# Handling Probing #
################################
run_probe_parser = subparsers.add_parser(
    'probe',
    help=f'''Probe the representations extracted by the model.
    Run `python . probe -h` for more details.
    You need to specify a config file for the probing.
    See example in `benchmark/tasks/GTZAN/GTZAN_base_config.yaml`.
    You can define your own config file, but it must follow the same structure as the example.
    Currently supported probing tasks: {bench.constants.task_constants.SUPPORTED_TASKS}.
    Currently supported representations: {bench.constants.model_constants.SUPPORTED_REPRESENTATIONS}.
    ''',
)
run_probe_parser = add_probe_args(run_probe_parser)
run_probe_parser.set_defaults(
    main_func=probe_main,
)

################################
# Handling Finetuning #
################################
run_finetune_parser = subparsers.add_parser(
    'finetune',
    help=f'''Finetune the pretrained model on the specified task.
    You need to specify a config file for the finetuning.
    See example in `benchmark/tasks/GTZAN/GTZAN_base_config.yaml`.
    You can define your own config file, but it must follow the same structure as the example.
    Currently supported finetuning tasks: {bench.constants.task_constants.SUPPORTED_TASKS}.
    Currently supported representations: {bench.constants.model_constants.SUPPORTED_FINETUNING_MODELS}.
    Run `python . finetune -h` for more details.
    ''',
)
run_finetune_parser = add_finetune_args(run_finetune_parser)
run_finetune_parser.set_defaults(
    main_func=finetune_main,
)


################################################### End of subparsers ###########################################################

args = parser.parse_args()
# Execute the specified main function. 
args.main_func(args) #选择probe or extract

#run python . extract -c configs/mert/MERT-v1-95M/EMO.yaml
#python . probe -c configs/mert/MERT-v1-95M/EMO.yaml