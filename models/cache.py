import numpy as np
import os
import re
import torch
from typing import Union,List,Optional
from functools import lru_cache
from collections import OrderedDict

class SparseCache():
    """
        Usage:
        >>> capacity = 10
        >>> indexes  = [1,4,5]
        >>> values   = torch.rand(3,  128)
        >>> cache    = SparseCache(capcity, indexes,  values)
        >>> len(cache)
        3
        >>> 2 in cache
        False
        >>> 1 in cache
        True
        >>> cache[1]
        torch.Tensor([...]) # shape = [128]
        >>> cache[[1,4]]
        torch.Tensor([[...],[...]]) # shape = [2,128]
        >>> cache[3] = torch.rand(128)
        >>> len(cache)
        4
        >>> cache[[7,4,9]] = torch.rand(3,128)
        >>> len(cache)
        6
    """
    def __init__(self, capacity:int, keys:List[int], values:torch.Tensor):
        """
            capacity:   int
                        the capacity of the cache
            keys:       List[int]
                        initial keys 
            values:     torch.FloatTensor[batch, n_emb]
                        initial embedding
                        
        """
        assert values.dim()== 2
        assert len(keys) == values.shape[0] 
        self.capacity = capacity
        self.cache = {k:v for k, v in zip(keys, values)}
    def __len__(self):
        return  len(self.cache)
    def __contains__(self, key:int):
        """
            Parameters
            ----------
                key:        int

            Returns
            -------
                bool, whether contain the key
        """
        return key in self.cache
    def __getitem__(self, key:Union[int, List[int]]):
        """
            Parameters
            ----------
                key:        int | List[int]
            
            Returns
            -------
                torch.FloatTensor[n_emb] | [batch, n_emb]
        """
        if isinstance(key, list):
            return torch.stack([self.cache[k] for k in key], 0)
        else:
            return self.cache[key]
    def __setitem__(self, key:Union[int, List[int]], value:torch.Tensor):
        """
            Parameters
            ----------
                key:        int | List[int]

                value:      torch.FloatTensor[n_emb] | [batch,n_emb]

        """
        if isinstance(key, list):
            assert value.dim() == 2 and value.shape[0] == len(key)
            for k,v in zip(key,value):
                self.cache[k] = v 
        else:
            assert value.dim() == 1
            self.cache[key] = value
        

class DenseCache:
    def __init__(self, capacity:int, keys:List[int], values:torch.Tensor):
        assert values.dim()== 2
        assert len(keys) == values.shape[0] 
        self.capacity = capacity
        self.cache = torch.zeros([capacity, values.shape[1]])
        self.cache[keys] = values
        self.keys  = set(keys)
    def __len__(self):
        return len(self.keys)
    def __contains__(self, key:int):
        return key in self.keys 
    def __getitem__(self, key:Union[int, List[int]]):
        return self.cache[key]
    def __setitem__(self, key:Union[int, List[int]], value:torch.Tensor):
        if isinstance(key, list):
            self.keys.update(set(key))
        else:
            self.keys.add(key)
        self.cache[key] = value
       
        
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")

class _LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    def __contains__(self, key):
        return key in self.cache
    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]
    def __setitem__(self, key, value):
        self.cache[key] = value 
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

class DiskCache:
    def __init__(self, capacity:int, keys:Optional[List[int]]=None, values:Optional[torch.Tensor]=None, cache_dir:str=CACHE_DIR, name:Union[str, List[str]]="diskcache", pin_memory_size:int=128):
        self.name_pad_len = int(np.ceil(np.log10(capacity)))
        self.pin_memory = _LRUCache(pin_memory_size)
        self.keys = set() if keys is None else set(keys)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        path = cache_dir
        if isinstance(name, str):
            name = [name]
        for n in name:
            path = os.path.join(path, n)
            if not os.path.exists(path):
                os.mkdir(path)
        self.path = path
        if keys is not None:
            self[keys] = values
        else:
            filenames = os.listdir(self.path)
            self.keys = set(int(re.findall("\d+",filename)[0]) for filename in filenames)
            

    def __len__(self):
        return len(self.keys)
    def __contains__(self, key:int):
        return key in self.keys
    def get_filepath(self, key:int):
        filepath = os.path.join(self.path, f"{str(key).zfill(self.name_pad_len)}.npy")
        return filepath
    def load(self, key:int):
        if key not in self.pin_memory:
            filepath = self.get_filepath(key)
            value =  torch.tensor(np.load(filepath))
            self.pin_memory[key] = value
        else:
            value = self.pin_memory[key]
        return value
        
    def save(self, key:int, value:torch.Tensor):
        self.pin_memory[key] = value
        filepath = self.get_filepath(key)
        np.save(filepath,value.cpu().numpy())

    def __getitem__(self, key:Union[int, List[int]]):
        if isinstance(key, (list,np.ndarray)):
            xs = []
            for k in key:
                assert k in self.keys
                xs.append(self.load(k))
            return torch.stack(xs, 0)
        else:
            return self.load(key)

    def __setitem__(self, key:Union[int, List[int]], values:torch.Tensor):
        if isinstance(key, (list,np.ndarray)):
            for k, v in zip(key, values):
                self.save(k, v)
        else:
            self.save(key, values)


if __name__ == '__main__':
    c = DenseCache(3, (4,))
    index = c.not_contain(torch.tensor([0,1]))
    print(len(c))
    print(index)
    c[index] = torch.rand(2,4)
    print(len(c))
    
