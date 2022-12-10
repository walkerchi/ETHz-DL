import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Union, Optional 
from PIL.ImageFile import ImageFile as PILImage

from ..openai_clip.openai_clip import load,tokenize

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")

class ImageLoader(DataLoader):
    def __init__(self,  resolution:int, images:List[PILImage], batch_size:int, **kwargs):
        super().__init__(
            dataset     = images, 
            batch_size  = batch_size,
            **kwargs
        )
        self.transform = Compose([
            Resize(resolution, interpolation=BICUBIC),
            CenterCrop(resolution),
            lambda image:image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    def collate_fn(self, images:List[PILImage]):
        results = []
        for image in images:
            results.append(self.transform(image))
        results = torch.stack(results,0)
        return results



class MAEImageEncoder(nn.Module):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        assert p>=0.0 and p<=1.0, f"the drop probability p should be between [0,1]"
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        model,  self.preprocessor = load(model_str, device="cpu", download_root=cache_dir)
        visual  = model.visual
        
        self.p                  = p
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


class OpenAIFLIP(nn.Module):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        assert model_str.startswith("ViT"), f"FLIP can only be used to optimize ViT not CNN"
        assert p>=0.0 and p<=1.0, f"the drop probability p should be between [0,1]"
        self.image_encoder = MAEImageEncoder(p, model_str, cache_dir)
        self.text_encoder  = OpenAICLIPTextEncoder(model_str, cache_dir)
        self.model_str     = model_str 
        self.p             = p
        self.no_grad       = True

    def set_no_grad(self, state=True):
        self.no_grad = state
        return self

    @property
    def image_encoder_str(self):
        return f"OpenAIFLIP<{self.model_str},{self.p}>.ImageEncoder"
    
    @property
    def text_encoder_str(self):
        return f"OpenAICLIP<{self.model_str}>.TextEncoder"

    def preprocess_images(self, images:List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts:List[str]):
        return self.text_encoder.preprocess(texts)

    def encode_images(self, images:Union[List[PILImage], PILImage], batch_size:Optional[int]=None, device:str='cpu', verbose:bool=False)->torch.Tensor:
        """
            Parameters
            ----------
                images:     Union[List[PILImage], PILImage]
                            could input a list of PIL.Image or a single.
                            If input a single, the output shape will be [n_emb]
                            else the output will be [n_image, n_emb]
                batch_size: Optional[int]
                            if batch_size is `None`, it will visit the image iteratively,
                            else it will grab them in a dataloader and do it in batch
                device:     str
                            The output device for the embedding
                            As the embeding is so big, so sometimes we should store them in cpu rather than gpu
                            Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`
                verbose:    bool
                            if verbose, the tqdm progress bar will be showed 
                            else, the encoding process will keep silent
            
            Returns
            -------
                emb_images: torch.FloatTensor[n_image, n_emb] or [e_emb]
                            the embedding of the encoded images
        """
        is_single = False
        if isinstance(images, PILImage):
            images = [images]
            is_single = True
        emb_images = []
        if batch_size is None:
            if verbose:
                images = tqdm(images, total=len(images), desc="Image Encoding")
            for image in images:
                image      = self.preprocess_images([image])
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch = self.image_encoder(image)
                else:
                    emb_batch  = self.image_encoder(image)
                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_images.append(emb_batch)
            emb_images = torch.cat(emb_images, 0)
        else:
            images = ImageLoader(self.resolution, images, batch_size)
            if verbose:
                images = tqdm(images, total=len(images), desc="Image Encoding")
            for batch in images:
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch = self.image_encoder(batch)
                else:
                    emb_batch  = self.image_encoder(batch)
                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_images.append(emb_batch)
            emb_images = torch.cat(emb_batch, 0)
       
        if is_single:
            return emb_images[0] 
        else:
            return emb_images

    def encode_texts(self, texts:Union[List[str],str], batch_size:Optional[int]=None, device:str='cpu', verbose:bool=False)->torch.Tensor:
        """
            Parameters
            ----------
                texts:      Union[List[str], str]
                            could input a list of str or a single.
                            If input a single, the output shape will be [n_emb]
                            else the output will be [n_text, n_emb]
                batch_size: Optional[int]
                            if batch_size is `None`, it will visit the text iteratively,
                            else it will grab them in a dataloader and do it in batch
                device:     str
                            The output device for the embedding
                            As the embeding is so big, so sometimes we should store them in cpu rather than gpu
                            Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`
                verbose:    bool
                            if verbose, the tqdm progress bar will be showed 
                            else, the encoding process will keep silent
            Returns
            -------
                emb_texts:  torch.FloatTensor[n_text, n_emb] or [e_emb]
                            the embedding of the encoded texts
        """
        is_single = False
        if isinstance(texts, str):
            texts = [texts]
            is_single = True
        emb_texts = []
        if batch_size is None:
            if verbose:
                texts = tqdm(texts, total=len(texts), desc="Text Encoding")
            for text in texts:
                text       = self.preprocess_texts([text])
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch  = self.text_encoder(text)
                else:
                    emb_batch  = self.text_encoder(text)
                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_texts.append(emb_batch)
            emb_texts = torch.cat(emb_texts, 0)
        else:
            texts = TextLoader(texts, batch_size)
            if verbose:
                texts = tqdm(texts, total=len(texts), desc="Text Encoding")
            for batch in texts:
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch  = self.text_encoder(batch)
                else:
                    emb_batch  = self.text_encoder(batch)
                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_texts.append(emb_batch)
            emb_texts = torch.cat(emb_batch, 0)
        if is_single:
            return emb_texts[0] 
        else:
            return emb_texts


