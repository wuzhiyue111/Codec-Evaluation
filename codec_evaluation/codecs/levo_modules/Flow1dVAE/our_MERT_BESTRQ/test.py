import os
import sys
import torch
from dataclasses import dataclass
from logging import getLogger
import fairseq.utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task

logger = getLogger(__name__)

@dataclass
class UserDirModule:
    user_dir: str

def find_project_root(current_path: str, target_folder: str = "Codec-Evaluation"):
    path = os.path.abspath(current_path)
    while True:
        if os.path.basename(path) == target_folder:
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError(f"Cannot find project root folder '{target_folder}' from {current_path}")
        path = parent

def load_model(model_dir, checkpoint_dir):
    '''Load Fairseq SSL model'''
    project_root = find_project_root(os.path.dirname(__file__), target_folder="Codec-Evaluation")
    mert_path = os.path.join(project_root, "codec_evaluation", "codecs", model_dir)
    # model_dir 已经是完整目录到 mert_fairseq
    mert_path = os.path.abspath(mert_path)
    
    if not os.path.exists(mert_path):
        raise FileNotFoundError(f"Cannot find mert_fairseq in {mert_path} or {fixed_path}")

    # 加入 sys.path
    if mert_path not in sys.path:
        sys.path.insert(0, mert_path)

    # import_user_module
    module_args = UserDirModule(user_dir=mert_path)
    fairseq.utils.import_user_module(module_args)

    # 载入 checkpoint
    model, cfg, task = load_model_ensemble_and_task([checkpoint_dir], strict=False)
    model = model[0]

    return model

