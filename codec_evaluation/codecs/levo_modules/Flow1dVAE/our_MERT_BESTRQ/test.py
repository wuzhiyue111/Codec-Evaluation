import torch
from dataclasses import dataclass
from logging import getLogger
import torch.nn.functional as F
import fairseq.utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task

logger = getLogger(__name__)

@dataclass
class UserDirModule:
    user_dir: str

def load_model(model_dir, checkpoint_dir):
    '''Load Fairseq SSL model'''

    #导入模型所在的代码模块
    # codeclm/tokenizer/Flow1dVAE/our_MERT_BESTRQ/mert_fairseq
    model_path = UserDirModule(model_dir)
    # if "codeclm" in model_dir:
    #     model_path.user_dir = os.path.join(model_path.user_dir, "mert_fairseq")
    fairseq.utils.import_user_module(model_path)
    
    #载入模型的checkpoint
    model, cfg, task = load_model_ensemble_and_task([checkpoint_dir], strict=False)
    model = model[0]

    return model
