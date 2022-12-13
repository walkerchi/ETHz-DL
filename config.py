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
        self.kwargs = [config[str(i)]['kwargs'] for i in range(len(self.name))]
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
    def __call__(self):
        """Build the CLIP models and wrap them in CasCLIP
            Return
            ------
                CasCLIP instance 
        """
        logging.info("loading models...")
        return models.CasCLIP([getattr(models,self.name[i])(**self.kwargs[i]).to(self.device) for i in range(len(self))], cache_type=self.cache_type)

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
        query_rate: Optional[float], default None
                    query rate for the experiment,
                    if 0.1, the experiment will carry out on 10% of the texts
        cache_type: str, default `sparse`
                    the cache type in CasCLIP, choose from [`sparse`, `dense`]
    """
    def __init__(self,config):
        
        self.topm    = config['topm']              if 'topm'          in config else None
        self.topk    = config['topk']           
        self.seed    = config['seed']              if 'seed'          in config else 123456789
        self.logging_level=config['logging_level'] if 'logging_level' in config else 'INFO'
        self.device  = config['device']            if 'device'        in config else DEFAULT_DEVICE
        self.batch_size = config['batch_size']     if 'batch_size'    in config else None
        self.filename= config['filename']
        self.query_rate   =config['query_rate']    if 'query_rate'    in config else None
        self.cache_type   =config['cache_type']    if 'cache_type'    in config else "sparse"

        self.dataset = DatasetConfig(config['dataset'])
        self.models  = ModelsConfig(config['models'], self.device, self.cache_type)
        
        assert self.topm is None or isinstance(self.topm, list), f"`topm` in configure file should be list of int or `None`, but got {self.topm}"
        assert isinstance(self.topk, list), f"`topk` in configure file should be list of int, but got {self.topk}"
        assert isinstance(self.logging_level, str), f"`logging_level` in configure file should be str, but got {self.logging_level}"
        assert self.topm is None or len(self.topm) + 1 == len(self.models), f"The length of `topm` in configure file is expected to be {len(self.models)-1}, but got {len(self.topm)}" 

        self.init_random_seed()
        self.init_logging()

    def to_dict(self):
        return {
            "dataset":  self.dataset.to_dict(),
            "models":   self.models.to_dict(),
            "topm":     self.topm,
            "topk":     self.topk,
            "seed":     self.seed,
            "logging_level":self.logging_level,
            "device":   self.device,
            "batch_size":self.batch_size,
            "query_rate":self.query_rate,
            "cache_type":self.cache_type
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
            filename    = log_path, 
            level       = getattr(logging,self.logging_level)
        ) 
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("\n"+toml.dumps(self.to_dict()))

