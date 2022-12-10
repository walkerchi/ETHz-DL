import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List,Union,Optional
from PIL.ImageFile import ImageFile as PILImage
from PIL import Image 
from .clip import *

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

class TextLoader(DataLoader):
    def __init__(self, texts:List[str], batch_size:int, **kwargs):
        super().__init__(
            dataset    = texts,
            batch_size = batch_size,
            **kwargs
        )    
    def collate_fn(self, texts:List[str]):
        return tokenize(["a photo of " + text for text in texts])

class OpenAICLIP(nn.Module):
    def __init__(self, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.model, self.preprocessor = load(model_str, device="cpu", download_root=cache_dir)
        self.model_str  = model_str
        self.resolution = self.model.visual.input_resolution
        self.no_grad    = True
    
    def __str__(self):
        return f"OpenAICLIP<{self.model_str}>"

    def set_no_grad(self, state=True):
        self.no_grad = state
        return self

    @property
    def image_encoder_str(self):
        return f"OpenAICLIP<{self.model_str}>.ImageEncoder"
    
    @property
    def text_encoder_str(self):
        return f"OpenAICLIP<{self.model_str}>.TextEncoder"

    def preprocess_images(self, images:List[PILImage]):
        """
            Parameters
            ----------
                images:     List[PILImage]
            Returns
            -------
                torch.FloatTensor[n_batch, C, H, W]
        """
        with torch.no_grad():
            return torch.stack([self.preprocessor(image) for image in images],0)

    def preprocess_texts(self, texts:List[str]):
        """
            Parameters
            ----------
                texts:      List[str]
            Returns
            -------
                torch.LongTensor[n_batch, seq_len]
        """
        return tokenize(["a photo of " + text for text in texts])

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
                            the device should be in cpu, no matter what the device for the encoder
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
                        emb_batch  = self.model.encode_image(image)
                else:
                    emb_batch  = self.model.encode_image(image)
                if emb_batch.device != torch.device(device):
                    emb_batch  = emb_batch.to(device)
                emb_images.append(emb_batch)
            emb_images = torch.cat(emb_images, 0)
        else:
            images = ImageLoader(self.resolution, images, batch_size)
            if verbose:
                images = tqdm(images, total=len(images), desc="Image Encoding")
            for batch in images:
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch  = self.model.encode_image(batch)
                else:
                    emb_batch  = self.model.encode_image(batch)
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
                        emb_batch  = self.model.encode_text(text)
                else:
                    emb_batch  = self.model.encode_text(text)
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
                        emb_batch  = self.model.encode_text(batch)
                else:
                    emb_batch  = self.model.encode_text(batch)
                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_texts.append(emb_batch)
            emb_texts = torch.cat(emb_batch, 0)
        if is_single:
            return emb_texts[0] 
        else:
            return emb_texts

