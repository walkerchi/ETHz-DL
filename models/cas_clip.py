import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List,Optional
from PIL.ImageFile import ImageFile as PILImage

class CasCLIP(nn.Module):
    def __init__(self,
        *models:List[nn.Module]
        ):
        super().__init__()

        self.models             = models
        self.images             = None
        self.base_images_emb    = None
        self.cache_images_emb   = None

        for model in self.models:
            model.eval()
            model.set_no_grad()

    def __len__(self):
        return len(self.models)

    def build(self, images:List[PILImage], batch_size:Optional[int]=None, verbose:bool=True):
        self.images             = images
        images_emb              = self.models[0].encode_images(images, batch_size=batch_size, verbose=verbose)
        self.cache_images_emb   = {self.models[0].image_encoder_str:images_emb}
        return self

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
        cache_text_emb = {}
        
        candidate = None # if `None` means select all, else should be torch.LongTensor of 1-dim (global index)
        for n_candidate, model in zip(n_candidates,self.models):

            # compute text embedding
            if model.text_encoder_str not in cache_text_emb:
                # if text_emb need compute
                text_emb = model.encode_texts(text)
                cache_text_emb[model.text_encoder_str] = text_emb
            else:
                # if text_emb is calcuated before, use it from the cache
                text_emb = cache_text_emb[model.text_encoder_str]
        
            # compute images embedding
            if model.image_encoder_str not in self.cache_images_emb:
                assert candidate is not None, f"You should call the `build` function first to precompute the embedding for first model"
                images = [self.images[c] for c in candidate]
                images_emb = model.encode_images(images, batch_size=batch_size)
                self.cache_images_emb[model.image_encoder_str] = {c.item():emb for c, emb in zip(candidate, images_emb)}
            else:
                images_emb = self.cache_images_emb[model.image_encoder_str]
                if isinstance(images_emb, dict):
                    # some of image embedding is not computed as it's sparsed stored
                    indexs              = [c.item() for c in candidate  if c not in self.cache_images_emb]
                    images              = [self.images[index] for index in indexs]
                    partial_images_emb  = model.encode_images(images, batch_size=batch_size)
                    for index, emb in zip(indexs, partial_images_emb):
                        images_emb[index] = emb
                    images_emb = torch.stack([images_emb[c.item()] for c in candidate], 0)
            # get topm or topk
            scores      = torch.cosine_similarity(images_emb,  text_emb[None, :])
            if candidate is None:
                candidate = scores.topk(n_candidate).indices.flatten()
            else:
                candidate = candidate[scores.topk(n_candidate).indices.flatten()]
           
        return candidate



