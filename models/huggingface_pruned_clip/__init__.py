import os
import torch 
import torch.nn as nn
from tqdm import tqdm
from importlib import import_module
from typing import List,Union,Optional
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader
from PIL.ImageFile import ImageFile as PILImage


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")


class ImageLoader(DataLoader):
    def __init__(self,  model_str:str, images:List[PILImage], batch_size:int, **kwargs):
        super().__init__(
            dataset     = images, 
            batch_size  = batch_size,
            **kwargs
        )
        self.processor = CLIPProcessor.from_pretrained(model_str)
    def collate_fn(self, images:List[PILImage]):
        return self.processor(images)

class TextLoader(DataLoader):
    def __init__(self, model_str:str, texts:List[str], batch_size:int, **kwargs):
        super().__init__(
            dataset    = texts,
            batch_size = batch_size,
            **kwargs
        )    
        self.tokenizer = CLIPTokenizer.from_pretrained(model_str)
    def collate_fn(self, texts:List[str]):
        return tokenize(["a photo of " + text for text in texts])



class PrunedHuggingFaceCLIP(nn.Module):
    def __init__(self, model_str:str, pruning_version:str='v0', cache_dir:str=CACHE_DIR):
        super().__init__()
        path = '.' + 'huggingface_pruned_clip' + '.' +  pruning_version
        pruned = import_module(path, 'models')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.tokenizer       = CLIPTokenizer.from_pretrained(model_str, cache_dir=cache_dir)
        self.processor       = CLIPProcessor.from_pretrained(model_str, cache_dir=cache_dir)
        self.model           = pruned.CLIPModel_pruned.from_pretrained(model_str, cache_dir=cache_dir)
        self.model_str       = model_str
        self.pruning_version = pruning_version
        self.no_grad         = True
    @property
    def image_encoder_str(self):
        return f"PrunnedHuggingFaceCLIP<{self.model_str},{self.pruning_version}>.ImageEncoder"
    @property
    def text_encoder_str(self):
        return f"HuggingFaceCLIP<{self.model_str}>.TextEncoder"
    def set_no_grad(self, state:bool=True):
        self.no_grad = state
        return self
    
    def preprocess_images(self, images:List[PILImage]):
        return self.processor(images=images, return_tensors="pt")
    
    def preprocess_texts(self, texts:List[str]):
        texts = ["a photo of" + text for text in texts]
        return self.tokenizer(texts, padding=True, return_tensors="pt")

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
                        emb_batch  = self.model.get_image_features(**image)
                else:
                    emb_batch = self.model.get_image_features(**image)
                if emb_batch.device != torch.device(device):
                    emb_batch  = emb_batch.to(device)
                emb_images.append(emb_batch)
            emb_images = torch.cat(emb_images, 0)
        else:
            images = ImageLoader(self.model_str, images, batch_size)
            if verbose:
                images = tqdm(images, total=len(images), desc="Image Encoding")
            for batch in images:
                if self.no_grad:
                    with torch.no_grad():
                        emb_batch  = self.model.get_image_features(**batch)
                else:
                    emb_batch = self.model.get_image_features(**batch)
                emb_batch  = self.encode(batch)
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
                        emb_batch  = self.model.get_text_features(**text)
                else:
                    emb_batch  = self.model.get_text_features(**text)

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
                        emb_batch  = self.model.get_text_features(**batch)
                else:
                    emb_batch  = self.model.get_text_features(**batch)

                if emb_batch.device != torch.device(device):
                    emb_batch = emb_batch.to(device)
                emb_texts.append(emb_batch)
            emb_texts = torch.cat(emb_batch, 0)
        if is_single:
            return emb_texts[0] 
        else:
            return emb_texts
 


