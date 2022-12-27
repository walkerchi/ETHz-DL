import torch
from typing import Union,List

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
       
        



if __name__ == '__main__':
    c = DenseCache(3, (4,))
    index = c.not_contain(torch.tensor([0,1]))
    print(len(c))
    print(index)
    c[index] = torch.rand(2,4)
    print(len(c))
    
