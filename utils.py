import os
import torch
import numpy as np
import random
import logging
from transformers.utils import logging as t_logging
import transformers
from pathlib import Path
from models.CLIP import CLIP
from datasets.mscoco.MSCOCO import MSCOCO
from models.CLIPOpenAI import CLIPOpenAI

PROJECT_PATH = Path(os.path.dirname(__file__))
EXPERIMENTS_PATH = Path(PROJECT_PATH / "experiments")

def prep_experiment_dir(experiment):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(experiment))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)]) + 1
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return Path(experiment_dir)

def set_seed(seed):
    # Randomly seed pytorch to get random model weights.
    # Don't use torch.seed() because of
    # https://github.com/pytorch/pytorch/issues/33546
    torch.manual_seed(seed)
    # Make rest of experiments deterministic (almost, see
    # https://pytorch.org/docs/stable/notes/randomness.html)
    random.seed(seed)
    np.random.seed(seed)

def configure_logger(experiment_dir, level_str):
    if level_str == "INFO":
        logging_level = logging.INFO
    elif level_str == "DEBUG":
        logging_level = logging.DEBUG
    else:
        raise Exception(f"Unkown logging level '{level_str}'.")
    logging.basicConfig(filename=str(experiment_dir / 'output.log'), level=logging_level) # NOTE: encoding option removed
    logging.getLogger().addHandler(logging.StreamHandler())
    # since this is only supported in a newer version than euler's python_gpu

def fill_config_defaults(config):
    if "eval_only_small_model" not in config:
        config["eval_only_small_model"] = False
    return config

def load_model(name, args):
    if name == 'CLIP':
        return CLIP(**args)
    elif name == 'CLIPOpenAI':
        return CLIPOpenAI(**args)
    else:
        raise Exception(f"Unknown model '{name}'.")

def load_dataset(name, args):
    if name == 'MSCOCO':
        return MSCOCO(**args)
    else:
        raise Exception(f"Unknown dataset '{name}'.")
