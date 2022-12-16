import os
import torch
from importlib import import_module
from ..huggingface_clip import HuggingFaceCLIP
from models.huggingface_pruned_clip import modeling_pruned_clip
from .modeling_pruned_clip import CLIPModel as CLIPModel_pruned
import sys
# Next line is necessary for unpickling (in torch.load) to work
sys.modules['modeling_pruned_clip'] = modeling_pruned_clip
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")


class HuggingFacePrunedCLIP(HuggingFaceCLIP):
    def __init__(self, model_str:str, pruning_version:str='v0', cache_dir:str=CACHE_DIR):
        super().__init__(model_str, cache_dir)
        self.model = torch.load(f'models/huggingface_pruned_clip/pruned_models/{pruning_version}')
        self.pruning_version = pruning_version

    @property
    def image_encoder_str(self):
        return f"PrunnedHuggingFaceCLIP<{self.model_str},{self.pruning_version}>.ImageEncoder"

