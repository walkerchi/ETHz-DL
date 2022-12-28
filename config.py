import os
import numpy as np
import random
import toml
import logging
import torch
from datetime import datetime
from transformers.utils import logging as t_logging
import transformers
from pathlib import Path

import models
import datasets

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DatasetConfig:
    """Configuration for dataset
        name:   str
                the name of dataset
        kwargs: dict
                the key-word arguments for the initialization of the dataset
    """
    def __init__(self,config):
        self.name   = config['name']
        self.kwargs = config['kwargs']
        assert isinstance(self.name, str)
        assert isinstance(self.kwargs, dict)
    def to_dict(self):
        return {
            'name'  :self.name,
            'kwargs':self.kwargs
        }
    def build(self):
        return self.__call__()
    def __call__(self):
        """Build and init the dataset and return
            Returns
            -------
                dataset instance from the datasets folder
        """
        logging.info("loading dataset...")
        return getattr(datasets,self.name)(**self.kwargs)
        
class ModelsConfig:
    """Configuration for models
        name:   List[str]
                the class name of each CLIP model 
        kwargs: List[dict]
                the key-word arguments for each CLIP model,
                the length should be the same as name 
        device: str
                the runtime device for each CLIP
        cache_type:str
                the cache_type for CasCLIP
    """

    def __init__(self,config, device, cache_type):
       
        self.device = device
        self.cache_type = cache_type
        self.name   = config['name']
        self.topm   = config.get('topm', None)
        self.kwargs = []
        for i in range(len(self.name)):
            if str(i) in config and "kwargs" in config[str(i)]:
                self.kwargs.append(config[str(i)]["kwargs"])
            else:
                self.kwargs.append({})
        # self.kwargs = [config[str(i)]['kwargs'] for i in range(len(self.name))]
        assert isinstance(self.device, str)
        for i in range(len(self)):
            assert isinstance(self.name[i], str)
            assert isinstance(self.kwargs[i], dict)
    def to_dict(self):
        result = {'name':self.name}
        for i, kwargs in enumerate(self.kwargs):
            result.update({str(i):kwargs})
        return result
    def __len__(self):
        return len(self.name)
    def build(self):
        return self.__call__()
    def __call__(self):
        """Build the CLIP models and wrap them in CasCLIP
            Return
            ------
                CasCLIP instance 
        """
        logging.info("loading models...")
        return models.CasCLIP([getattr(models,self.name[i])(**self.kwargs[i]).to(self.device) for i in range(len(self))], cache_type=self.cache_type)
    def __getitem__(self, index):
        return ModelsConfig({
            "name":[self.name[index]],
            "0":{"kwargs":self.kwargs[index]}
        }, self.device, self.cache_type)

class Config:
    """Configuration for the experiment
        
        topm:       Optional[list[int]], default `None`
                    the number of candidate selected by each CLIP except the last one
                    if None, the CasCLIP is a one-layer CLIP
        topk:       int, required
                    the topk related images selected by the model, 
                    it can also be seen as the number of candidate selected by the last CLIP
        seed:       int, default  123456789
                    the random seed
        logging_level:str, default `INFO`
                    the logging level argument for logging module
        device:     str, default `cpu` if nvidia-gpu is not avaiable else `gpu`
                    the runtime device for each CLIP, the result will be default in `cpu`
        batch_size: Optional[int], default None
                    the runtime batch_size for images encoding.
                    If None, a for loop will be applied
                    Others, a batched dataloader will be applied
        filename:   str, required
                    The filename of the configure file.
                    It's used for logging, 
                    the log will be writen to a timed-named file under the folder f`.log\{filename}` 
        f: float, default 1.0
                    query rate for the experiment, it will be different in differet experiments
                    in `speedup` experiment, it will calcuate the speed up assuming that only a fraction of f images will ever be returned by a search engine.
        cache_type: str, default `sparse`
                    the cache type in CasCLIP, choose from [`sparse`, `dense`]
        experiment: str, default `topk`
                    the experiment to run, choose from [`topk`, `speedup`]
                    if `topk`, it will run calcuate the `topk` result of the model on the given dataset 
                    if `speedup`, it will compare the time cost for the large model(last layer) and the small model(first layer) to compute the speed up
                        the speedup is given by $speedup = large\_time / (small\_time + query\_ratio * large\_time)$
        n_reps:      int, default 3
                    how often to repeat timing measurements for a `speedup` experiment
        base_model: str, required if experiment == `speedup`
                    model to use as a baseline comparison for `speedup` experiments
    """
    def __init__(self,config, init_logging:bool=True):
      
        self.topm         = config['topm']         if 'topm'          in config else None
        self.topk         = config['topk']         if 'topk'          in config else [1]  
        self.seed         = config['seed']         if 'seed'          in config else 123456789
        self.logging_level=config['logging_level'] if 'logging_level' in config else 'INFO'
        self.device       = config['device']       if 'device'        in config else DEFAULT_DEVICE
        self.batch_size   = config['batch_size']   if 'batch_size'    in config else None
        self.filename     = config['filename']
        self.f            =config['f']    if 'f'    in config else 1.0  # different meaning in two experiments
        self.cache_type   =config['cache_type']    if 'cache_type'    in config else "sparse"
        self.experiment   =config['experiment']    if 'experiment'    in config else "topk"
        if self.experiment == "speedup":
            self.base_model = ModelsConfig(config['base_model'], self.device, self.cache_type)
        else:
            self.base_model = None
        self.n_reps        =config['n_reps']         if 'n_reps'         in config else 3

        self.dataset = DatasetConfig(config['dataset'])
        self.models  = ModelsConfig(config['models'], self.device, self.cache_type)
        
        assert self.device in ["cpu", "cuda"]
        assert self.cache_type in ["sparse", "dense"]
        assert self.experiment in ["topk", "speedup", "distill"]
        assert self.topm is None or isinstance(self.topm, list), f"`topm` in configure file should be list of int or `None`, but got {self.topm}"
        assert isinstance(self.topk, list), f"`topk` in configure file should be list of int, but got {self.topk}"
        assert isinstance(self.logging_level, str), f"`logging_level` in configure file should be str, but got {self.logging_level}"
        assert self.topm is None or len(self.topm) + 1 == len(self.models), f"The length of `topm` in configure file is expected to be {len(self.models)-1}, but got {len(self.topm)}" 

        self.init_random_seed()
        if init_logging:
            self.init_logging()

    def to_dict(self):
        return {
            "dataset":  self.dataset.to_dict(),
            "models":   self.models.to_dict(),
            "base_model":   self.base_model.to_dict() if self.base_model is not None else "",
            "topm":     self.topm,
            "topk":     self.topk,
            "seed":     self.seed,
            "logging_level":self.logging_level,
            "device":   self.device,
            "batch_size":self.batch_size,
            "f":self.f,
            "cache_type":self.cache_type,
            # "n_reps":    self.n_reps
        }

    def init_random_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def init_logging(self):
        log_path = ".log"
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_path = os.path.join(log_path, self.filename)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        now      = datetime.now()
        log_path = os.path.join(log_path, now.strftime('%Y-%m-%d_%H-%M-%M')+'.log')
        logging.basicConfig(
            level       = getattr(logging,self.logging_level)
        ) 
        logger = logging.getLogger(self.filename)
        logger.setLevel(getattr(logging, self.logging_level))
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(log_path))
        logger.info("\n"+toml.dumps(self.to_dict()))



