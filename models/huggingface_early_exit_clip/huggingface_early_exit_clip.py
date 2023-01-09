import os
import time
import operator
import logging
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from itertools import chain
from functools import reduce
from tqdm import tqdm
from typing import List, Optional, Union
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from PIL.ImageFile import ImageFile as PILImage
from transformers import CLIPProcessor, CLIPModel

from ..huggingface_clip import HuggingFaceImageEncoder,HuggingFaceCLIP,HuggingFaceTextEncoder
from ..cache import DiskCache


CACHE_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".cache")

EARLY_EXIT_DIR = os.path.join(os.path.dirname(__file__),"weight")




class HuggingFaceImageEarlyExitEncoderStage(nn.Module):
    def __init__(self, layers, projection, projection_fixed:bool=False):
        super().__init__()
        self.layers     = layers
        self.projection = projection
        self.projection_fixed = projection_fixed

    def project(self, x):
        x = x[:, 0, :]
        return self.projection(x)

    def forward(self, x, project:bool=True):
        """
            Parameters:
            -----------
                x:      torch.Tensor[B, n_grid^2, n_embed]
                project:bool, default  `True`
                        whether apply projection
            Returns
            -------
                if project = `True`
                    the return shape will be [B, n_embed]
                else:
                    the return shape will be [B, n_grid^2, n_embed] 

        """
        assert x.dim() == 3
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x, None, None)
                x = x[0]
        if project:
            if self.projection_fixed:
                with torch.no_grad():
                    x = self.project(x)
            else:
                x = self.project(x)
        return x

class HuggingFaceEarlyExitImageEncoder(HuggingFaceImageEncoder):
    """
        Usage:
            >>> m1 = HuggingFaceEarlyExitImageEncoder(from_stage=None, to_stage=2)
            >>> m2 = HuggingFaceEarlyExitImageEncoder(from_stage=2, to_stage=4)
            >>> x = torch.rand(2,3,224,224)
            >>> m1(x).shape    # x->embedding->stages[0]->stage[1]
            [2, 197, 512]
            >>> x = m1(x)
            >>> x = m2(x)      # x->stages[2]->stages[3]
            >>> x.shape
            [2, 197, 512] 
           
    """
    def __init__(self, model_str: str, 
                cache_dir: str = CACHE_DIR, 
                from_stage:Optional[int]=None, 
                to_stage:int=4, 
                epoch:Optional[int]=None, 
                lr:Optional[float]=None):
        super(HuggingFaceImageEncoder, self).__init__()
        self.processor = CLIPProcessor.from_pretrained(model_str, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        self.embedding = nn.Sequential(
            model.vision_model.embeddings,
            model.vision_model.pre_layrnorm
        )
        encoder = model.vision_model.encoder
        n_out, n_embed = model.visual_projection.weight.shape
        self.cache_dir  = cache_dir
        self.from_stage = from_stage
        self.to_stage   = to_stage
        self.model_str = model_str

        stages = []
        for i in range(0 if from_stage is None else from_stage, to_stage):
            stages.append(
                HuggingFaceImageEarlyExitEncoderStage(
                encoder.layers[i*3:(i+1)*3], 
                nn.Sequential(nn.LayerNorm(n_embed), nn.Linear(n_embed, n_out)) if i < 3 else nn.Sequential(model.vision_model.post_layernorm, model.visual_projection),
                projection_fixed=True if i < 3 else False
                )
            )
        self.stages = nn.ModuleList(stages)

        self.epoch = epoch 
        self.lr    = lr
        if self.epoch is None:
            assert self.lr is None
        else:
            assert self.lr is not None

    @property
    def name(self):
        return f"{self.model_str},{self.from_stage},{self.to_stage}"
    @property
    def load_context_key(self):
        if self.from_stage is None:
            return None
        else:
            return f"stage{self.from_stage}"
    @property
    def save_context_key(self):
        if self.to_stage == 4:
            return None
        else:
            return f"stage{self.to_stage}"

    def get_filename(self, stage:int):
        return f"{self.model_str.replace('/','-')}_stage{stage}.pt"

    def save(self, path: str = EARLY_EXIT_DIR):
        if not os.path.exists(path):
            os.mkdir(path)
        for i,stage in zip(range(0 if self.from_stage is None else self.from_stage, self.to_stage),self.stages):
            if i < 3:
                torch.save(stage.projection.state_dict(), os.path.join(path, self.get_filename(i+1)))
    def load(self, path: str = EARLY_EXIT_DIR):
        for i, stage in zip(range(0 if self.from_stage is None else self.from_stage, self.from_stage), self.stages):
            path = os.path.join(path, self.get_filename(i+1))
            if not os.path.exists(path):
                print(f"path:{path}")
                raise FileNotFoundError(
                    "Weight File not exist, You should train the early exit it first to have the weight")
            weight = torch.load(path)
            stage.projection.load_state_dict(weight)
        return self

    def forward(self, x:torch.Tensor, project:bool=True, save_ctx:bool=False):
        """
            Parameters:
            -----------
                x:                  torch.Tensor[B,C,H,W] or torch.Tensor[B, n_gird^2, n_embed]
                                    if x is of former shape, the `from_stage` should be `None`
                project:            bool, default=`True`
                                    whether last layer should be projected
                save_ctx:      bool, default=`False`
                                    whether return the embedding before projection
            Returns:
            --------
                x:                  if project=`True` torch.Tensor[B,n_out]
                                    else              torch.Tensor[B, n_grid^2, n_out]
        """
        if self.from_stage is None:
            with torch.no_grad():
                x = self.embedding(x)
        for stage in self.stages[:-1]:
            x = stage(x, project=False)
        if save_ctx:
            return x, self.stages[-1](x, project=project)
        else:
            return self.stages[-1](x, project=project) 
    
    def train_early_exit(self, images_path: List[str], batch_size: Optional[int], device: str = "cpu", verbose: bool = True, logger=logging):


        class ImageLoader(DataLoader):
            def __init__(self, images_path:List[str],processor,batch_size=batch_size):
                super().__init__(np.arange(len(images_path)), batch_size=batch_size, collate_fn=self.collate_fn)
                self.processor = processor
                self.images_path = images_path
            def collate_fn(self, index:List[int]):
                images = []
                for i in index:
                    image = Image.open(self.images_path[i])
                    image.load()
                    images.append(image)
                images = self.processor(images=images, return_tensors="pt")["pixel_values"]
                index  = np.array(index)
                return index,images

        class FeatureMapLoader(DataLoader):
            def __init__(self, cache_dir, total_len:int=len(images_path), batch_size=batch_size):
                self.caches = [DiskCache(capacity=total_len, cache_dir=cache_dir, name=["early_exit",f"stage{i+1}"]) for i in range(4)]
                self.total_len = total_len 
                super().__init__(np.arange(total_len), batch_size=batch_size, collate_fn=self.collate_fn)

            def collate_fn(self, index):
                index = np.stack(index)
                return tuple(self.caches[i][index] for i in range(4))

            def __iter__(self):
                for cache in self.caches:
                    assert len(cache) == self.total_len, f"all feature map should be computed first, but only compute {v}/{self.total_len} for {k}"
                return super().__iter__()



        assert len(self.stages) == 4

        if batch_size is None:
            images = [Image.open(image_path) for image_path in images_path]
            images = self.preprocess(images)
        else:
            images = ImageLoader(images_path, self.processor, batch_size=batch_size)

        if verbose:
            loader = tqdm(images, total=len(images),
                            desc=f"Extract inner feature map...")
        else:
            loader = images

        if batch_size is None:
            feature_loader = FeatureMapLoader(self.cache_dir, len(images_path), batch_size=1)
        else:
            feature_loader = FeatureMapLoader(self.cache_dir, len(images_path), batch_size=batch_size)

        self.eval()
        with torch.no_grad():
            for index, images in loader:
                partial_index = [i for i,ind in enumerate(index) if ind not in feature_loader.caches[0]]
                if len(partial_index) == 0:
                    continue
                images = images[partial_index]
                index = index[partial_index]
                images = images.to(device)
                x = self.embedding(images)
                for i in range(4):
                    x = self.stages[i](x, project=False if i < 3 else True)
                    feature_loader.caches[i][index[partial_index]] = x
        
        self.train()

        optimizers = [
            torch.optim.Adam(
                self.stages[i].projection.parameters(), 
                lr=self.lr
            ) 
            for i in range(3)
            ]

        loss_fn = lambda y, p: F.cosine_embedding_loss(
                    y, p, torch.ones(len(y)).to(device))

        for ep in range(self.epoch):
            if verbose:
                loader = tqdm(feature_loader, total=len(feature_loader),
                              desc=f"Epoch[{ep:2}/{self.epoch}]")
            else:
                loader = feature_loader

            l_loss = {f"stage{i+1}":[] for i in range(3)}
            start = time.process_time()

            for *features,stage4 in loader:
                for i, feature in enumerate(features):
                    if feature.device != torch.device(device):
                        feature = feature.to(device)

                    loss = loss_fn(stage4, self.stages[i].project(feature))
                    loss.backward()
                    optimizers[i].step()
                    optimizers[i].zero_grad()
                    l_loss[f"stage{i+1}"].append(loss.item())

            end = time.process_time()
            
            for k,v in l_loss.items():
                l_loss[k] = np.array(v)
            loss_str = reduce(operator.add, 
                        [f"loss_stage{i+1}:{v.mean():5.3f}({v.std():5.2f}) " 
                        for i,v in enumerate(l_loss.values())]
                        )
            logger.info(
                f"Epoch[{ep}/{self.epoch}] {loss_str} time:{end-start:7.3f}s")

        self.save()


class HuggingFaceEarlyExitCLIP(HuggingFaceCLIP):
    def __init__(self, model_str: str, cache_dir: str = CACHE_DIR, from_stage:Optional[int]=None, to_stage:int=4, **kwargs):
        super(HuggingFaceCLIP, self).__init__()
        self.image_encoder = HuggingFaceEarlyExitImageEncoder(
            model_str=model_str, cache_dir=cache_dir, from_stage=from_stage, to_stage=to_stage, **kwargs)
        self.text_encoder = HuggingFaceTextEncoder(model_str, cache_dir)
        self.model_str = model_str

    @property
    def image_encoder_str(self):
        return f"HuggingFaceEarlyExit<{self.image_encoder.name}>.ImageEncoder"

    def encode_image(self, image: torch.Tensor, project:bool=True):
        return self.image_encoder(image, project=project, save_ctx=True)

    def encode_text(self, input_ids: torch.tensor, attention_mask: torch.Tensor):
        return self.text_encoder(input_ids, attention_mask)

    def encode_images(self, images: Union[List[PILImage], PILImage, torch.Tensor], batch_size: Optional[int] = None,
                      device: str = 'cpu', verbose: bool = False, return_timing: bool = False) -> torch.Tensor:
        is_single = False
        if isinstance(images, PILImage):
            images = [images]
            is_single = True
        if isinstance(images[0], PILImage):
            images = self.preprocess_images(images)
        

        if batch_size is not None:
            images = DataLoader(images, batch_size=batch_size)
        if verbose:
            images = tqdm(images, total=len(images), desc="Image Encoding")

        emb_images = []
        ctx_images = []

        start_time = time.process_time()

        for image in images:
            if batch_size is None:
                image = image[None, ...]
            if self.no_grad:
                with torch.no_grad():
                    ctx_batch, emb_batch = self.encode_image(image)
            else:
                ctx_batch, emb_batch = self.encode_image(image)
            if emb_batch.device != torch.device(device):
                emb_batch = emb_batch.to(device)
            emb_images.append(emb_batch)
            ctx_images.append(ctx_batch)

        end_time = time.process_time()

        if return_timing:
            return end_time - start_time

        emb_images = torch.cat(emb_images, 0)
        ctx_images = torch.cat(ctx_images, 0)

        if is_single:
            return ctx_images[0], emb_images[0]
        else:
            return ctx_images, emb_images