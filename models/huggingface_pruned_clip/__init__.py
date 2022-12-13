import os
from importlib import import_module
from ..huggingface_clip import HuggingFaceCLIP

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")


class HuggingFacePrunedCLIP(HuggingFaceCLIP):
    def __init__(self, model_str:str, pruning_version:str='v0', cache_dir:str=CACHE_DIR):
        super().__init__(model_str, cache_dir)
        path = '.' + 'huggingface_pruned_clip' + '.' +  pruning_version
        pruned = import_module(path, 'models')
        self.model           = pruned.CLIPModel_pruned.from_pretrained(model_str, cache_dir=cache_dir)
        self.pruning_version = pruning_version

    @property
    def image_encoder_str(self):
        return f"PrunnedHuggingFaceCLIP<{self.model_str},{self.pruning_version}>.ImageEncoder"
    
