import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from typing import List,Union,Optional
from PIL.ImageFile import ImageFile as PILImage
from PIL import Image 
from .clip import *
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

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
        with torch.no_grad():
            return tokenize(["a photo of " + text for text in texts])

    def encode_image(self, image:torch.Tensor):
        return self.model.encode_image(image)
    def encode_text(self, text:torch.Tensor):
        return self.model.encode_text(text)

    def encode_images(self, images:Union[List[PILImage], PILImage], batch_size:Optional[int]=None, device:str='cpu', verbose:bool=False, return_timing:bool=False)->torch.Tensor:
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
                return_timing:bool, default `False`,
                            if `True`, the return will be only a float time

            Returns
            -------
                if return_timing is `True`, will return a float number which is the process time
                else return the emb_images which is 
                emb_images: torch.FloatTensor[n_image, n_emb] or [e_emb]
                            the embedding of the encoded images
                            the device should be in cpu, no matter what the device for the encoder
        """
        assert len(images) > 0
        is_single = False
        if isinstance(images, PILImage):
            images = [images]
            is_single = True
        
        images = self.preprocess_images(images)

        if batch_size is not None:
            images = DataLoader(images, batch_size=batch_size)

        if verbose:
            images = tqdm(images, total=len(images), desc="Image Encoding")

        emb_images = []

        start_time = time.process_time()

        for image in images:
            if image.dim() == 3:
                image = image[None, ...]
            if self.no_grad:
                with torch.no_grad():
                    emb_batch  = self.encode_image(image)
            else:
                emb_batch  = self.encode_image(image)
            if emb_batch.device != torch.device(device):
                emb_batch  = emb_batch.to(device)
            emb_images.append(emb_batch)

        end_time = time.process_time()

        if return_timing:
            return end_time - start_time

        emb_images = torch.cat(emb_images, 0)

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
                if return_timing is `True`, will return a float number which is the process time
                else return the emb_images which is 
                emb_texts:  torch.FloatTensor[n_text, n_emb] or [e_emb]
                            the embedding of the encoded texts
        """
        is_single = False
        if isinstance(texts, str):
            texts = [texts]
            is_single = True
        
        texts = self.preprocess_texts(texts)

        if batch_size is not None:
            texts = TextLoader(texts, batch_size)
       
        if verbose:
            texts = tqdm(texts, total=len(texts), desc="Text Encoding")

        emb_texts = []

        for text in texts:
            if text.dim() == 1:
                text = text[None, ...]
            if self.no_grad:
                with torch.no_grad():
                    emb_batch  = self.encode_text(text)
            else:
                emb_batch  = self.encode_text(text)
            if emb_batch.device != torch.device(device):
                emb_batch = emb_batch.to(device)
            emb_texts.append(emb_batch)
            
        emb_texts = torch.cat(emb_texts, 0)

        if is_single:
            return emb_texts[0] 
        else:
            return emb_texts

