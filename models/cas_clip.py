import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Optional
from PIL.ImageFile import ImageFile as PILImage
import logging

from .cache import DenseCache, SparseCache


class CasCLIP(nn.Module):
    def __init__(self,
        models:List[nn.Module],
        cache_type:str = "sparse"
        ):
        super().__init__()
        assert cache_type in ["sparse", "dense"]
        self.models             = models
        self.images             = None
        self.base_images_emb    = None
        self.cache_images_emb   = None
        self.cache_type         = cache_type

        for model in self.models:
            model.eval()
            model.set_no_grad()

    def __len__(self):
        return len(self.models)

    def build(self, images:List[PILImage], batch_size:Optional[int]=None, verbose:bool=True):
        self.images             = images
        self.base_images_emb    = self.models[0].encode_images(images, batch_size=batch_size, verbose=verbose)
        self.cache_images_emb   = {}
        self.cache_text_emb     = {}
        return self

    def get_text_embed(self, index:int, text:str):
        """
            Parameters
            ----------
                index:  int
                        index of the layer
                text:   str
                        text for encoding

            Returns
            -------
                torch.FloatTensor[n_emb]
        """
        model = self.models[index]
        if model.text_encoder_str not in self.cache_text_emb:
            # if text_emb need compute
            text_emb = model.encode_texts(text)
            self.cache_text_emb[model.text_encoder_str] = text_emb
        else:
            # if text_emb is calcuated before, use it from the cache
            text_emb = self.cache_text_emb[model.text_encoder_str]
        return text_emb

    def get_images_embed(self, index:int, images_index:Optional[List[int]], batch_size:Optional[int]=None):
        """
            Parameters
            ----------
                index:  int
                        index of the layer
                images_index: List[int] | torch.LongTensor
                        images index for encoding

            Returns
            -------
                torch.FloatTensor[n_image, n_emb]
        """
        if isinstance(images_index, torch.Tensor):
            images_index = images_index.tolist()
        model = self.models[index]
        if index == 0:
            images_emb = self.base_images_emb
        else:
            if model.image_encoder_str not in self.cache_images_emb:
                assert images_index is not None, f"You should call the `build` function first to precompute the embedding for first model"
                images = [self.images[i] for i in images_index]
                images_emb = model.encode_images(images, batch_size=batch_size)
                # cache[name] = NewCache(len(self.images), keys,  values)
                self.cache_images_emb[model.image_encoder_str] = {
                    "sparse":SparseCache,
                    "dense":DenseCache
                }[self.cache_type](len(self.images), images_index, images_emb)
            else:
                cache = self.cache_images_emb[model.image_encoder_str]
                # some of image embedding is not computed as it's sparsed stored
                indexs              = [i for i in images_index  if i not in cache]
                if len(indexs) > 0:
                    images              = [self.images[index] for index in indexs]
                    partial_images_emb  = model.encode_images(images, batch_size=batch_size)
                    # cache.update(indexes, partial_image_emb)
                    cache[indexs]       = partial_images_emb
                    # cache.get(images_indexs)
                images_emb          = cache[images_index]
        return images_emb


    def query(self, text:str, topk:int=3, topm:Optional[List[int]]=None, batch_size:Optional[int]=None):
        """
            Parameters
            ----------
                text:       str
                            the query text
                topk:       int
                            the topk images for return
                topm:       Optional[List[int]], default `None`
                            `None` means `[]`
                            the topm for each cascade encoder
            Returns
            -------
                torch.LongTensor[topk]
        """
        assert self.images is not None, "You should call `build` before call `query`"
        if topm is None:
            n_candidates = []
        else:
            n_candidates = topm.copy()
        n_candidates.append(topk)
        for i in range(len(n_candidates)-1):
            assert n_candidates[i] >= n_candidates[i+1], f"topm should decrease for each iteration, got topm[{i}]={n_candidates[i]} and topm[{i+1}]={n_candidates[i+1]}"
        assert n_candidates[0]   <= len(self.images),f"topm number should be less than total images number, got topm number {n_candidates[0]}, and  {len(self.images)}"
        assert len(n_candidates) == len(self), f"Expected number of models equal to the len(topm)+1, but got {len(self)} and {len(n_candidates)}({n_candidates[:-1]})"
        self.cache_text_emb = {}

        candidate = None # if `None` means select all, else should be torch.LongTensor of 1-dim (global index)
        for i, n_candidate in enumerate(n_candidates):

            # compute text embedding
            text_emb = self.get_text_embed(i, text)

            # compute images embedding
            images_emb = self.get_images_embed(i, candidate)

            # get topm or topk
            scores      = torch.cosine_similarity(images_emb,  text_emb[None, :])
            if candidate is None:
                candidate = scores.topk(n_candidate).indices.flatten()
            else:
                candidate = candidate[scores.topk(n_candidate).indices.flatten()]

        return candidate


    def log_cache(self, logger=None):
        if logger is None:
            logger = logging
        logger.info("\n<========CasCLIP Cache=========>")
        for k, v in self.cache_images_emb.items():
            logger.info(f"{k}:{int(len(v)/v.capacity*100):2d}%")

