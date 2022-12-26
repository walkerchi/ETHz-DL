import os
import time
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Union, Optional 
from PIL.ImageFile import ImageFile as PILImage

from ..openai_clip.openai_clip import load,tokenize
from ..openai_clip import OpenAICLIP

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")

class TextLoader(DataLoader):
    def __init__(self, texts:List[str], batch_size:int, **kwargs):
        super().__init__(
            dataset    = texts,
            batch_size = batch_size,
            **kwargs
        )    
    def collate_fn(self, texts:List[str]):
        return tokenize(["a photo of " + text for text in texts])


class MAEImageEncoder(nn.Module):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        assert p>=0.0 and p<=1.0, f"the drop probability p should be between [0,1]"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        model,  self.preprocessor = load(model_str, device="cpu", download_root=cache_dir)
        visual  = model.visual
        
        self.p                  = 1 - p
        self.resolution         = visual.input_resolution       
        self.feature_embedding  = visual.conv1                  # 3 -> n_hid
        self.class_embedding    = visual.class_embedding        # shape = [n_hid]
        self.positional_embedding = visual.positional_embedding # shape = [grid**2+1, n_hid]
        self.transformer        = visual.transformer
        self.ln_pre             = visual.ln_pre                 # n_hid -> n_hid
        self.ln_post            = visual.ln_post                # n_hid -> n_hid
        self.proj               = visual.proj                   # n_hid -> n_emb

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    @property
    def device(self):
        return next(iter(self.parameters())).device   
    def forward(self, x:torch.Tensor):
        """
            Parameters
            ----------
                x:      torch.FloatTensor[B, C, H, W]
            Returns
            -------
                torch.FloatTensor[n_batch, n_embed]
        """
        x = x.type(self.dtype).to(self.device)
        x = self.feature_embedding(x)  # shape = [n_batch, n_hid, grid, grid]
        x = x.reshape(*x.shape[:2],-1) # shape = [n_batch, n_hid, grid**2]
        m = torch.randperm(x.shape[-1]) < int(self.p*x.shape[-1])
        x = x[:, :, m]
        x = x.permute(0, 2, 1)         # shape = [n_batch, n_grid, n_hid]
        class_emb    = self.class_embedding[None,None,:].repeat(x.shape[0], 1, 1).type(x.dtype).to(x.device)
        position_emb = torch.cat([self.positional_embedding[:-1][m],self.positional_embedding[-1][None,:]],0).type(x.dtype).to(x.device)
        x = torch.cat([class_emb, x], 1) # shape = [n_batch, n_grid+1, n_hid]
        x = x + position_emb
        x = self.ln_pre(x)               # shape = [n_batch, n_grid+1, n_hid]

        x = x.permute(1, 0, 2)   # NLD -> LND
        x = self.transformer(x)          
        x = x.permute(1, 0, 2)   # LND -> NLD

        x = self.ln_post(x[:, 0, :])  # shape = [n_batch, n_hid]
        x = x @ self.proj             # shape = [n_batch, n_emb]
        return x
    def preprocess(self, images:List[PILImage]):
        """
            Parameters
            ----------
                images:     List[PILImage]
            Returns
            -------
                torch.FloatTensor[n_batch, C, H, W]
        """
        return torch.stack([self.preprocessor(image) for image in images],0)


class OpenAICLIPTextEncoder(nn.Module):
    def __init__(self, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        model, _                  = load(model_str, device="cpu", download_root=cache_dir)

        self.transformer          = model.transformer
        self.vocab_size           = model.vocab_size
        self.token_embedding      = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final             = model.ln_final
        self.text_projection      = model.text_projection
        self.logit_scale          = model.logit_scale
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    def forward(self, text:torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    def preprocess(self, texts:List[str]):
        return tokenize(["a photo of " + text for text in texts])


class OpenAIFLIP(OpenAICLIP):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super(OpenAICLIP, self).__init__()
        assert model_str.startswith("ViT"), f"FLIP can only be used to optimize ViT not CNN"
        assert p>=0.0 and p<=1.0, f"the drop probability p should be between [0,1]"
        self.image_encoder = MAEImageEncoder(p, model_str, cache_dir)
        self.text_encoder  = OpenAICLIPTextEncoder(model_str, cache_dir)
        self.model_str     = model_str 
        self.p             = p

    @property
    def image_encoder_str(self):
        return f"OpenAIFLIP<{self.model_str},{self.p}>.ImageEncoder"

    def preprocess_images(self, images:List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts:List[str]):
        return self.text_encoder.preprocess(texts)

    def encode_image(self, image):
        return self.image_encoder(image)
    
    def encode_text(self, text):
        return self.text_encoder(text)
