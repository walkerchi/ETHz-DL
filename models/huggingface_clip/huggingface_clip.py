import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Union, Optional
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader
from PIL.ImageFile import ImageFile as PILImage

CACHE_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".cache")


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class TextLoader(DataLoader):
    def __init__(self, model_str: str, texts: List[str], batch_size: int, **kwargs):
        super().__init__(
            dataset=texts,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_str)

    def collate_fn(self, texts: List[str]):
        return self.tokenizer(["a photo of " + text for text in texts], padding=True, return_tensors="pt")


class HuggingFaceImageEncoder(nn.Module):
    def __init__(self, model_str: str, cache_dir: str = CACHE_DIR):
        super().__init__()

        self.processor = CLIPProcessor.from_pretrained(
            model_str, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def preprocess(self, images: List[PILImage]):
        return self.processor(images=images, return_tensors="pt")['pixel_values']

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, pixel_values: torch.Tensor):
        x = self.vision_model(pixel_values=pixel_values.type(self.dtype).to(self.device))
        x = x[1]
        x = self.visual_projection(x)

        return x


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, model_str: str, cache_dir: str = CACHE_DIR):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_str, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        text = model.text_model
        self.embed = text.embeddings
        self.encoder = text.encoder
        self.post_layernorm = text.final_layer_norm
        self.projection = model.text_projection

    def preprocess(self, texts: List[str]):
        texts = ["a photo of" + text for text in texts]
        return self.tokenizer(texts, padding=True, return_tensors="pt")

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        n_batch, seq_len = input_ids.size()
        x = self.embed(input_ids=input_ids)
        attention_mask = _expand_mask(attention_mask, x.dtype)
        x = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask,
            causal_attention_mask=self._build_causal_attention_mask(n_batch, seq_len, x.dtype).to(x.device))
        x = x[0]
        x = self.post_layernorm(x)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            input_ids.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]

        x = self.projection(x)

        return x


class HuggingFaceCLIP(nn.Module):
    def __init__(self, model_str: str, cache_dir: str = CACHE_DIR):
        super().__init__()
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     model_str, cache_dir=cache_dir)
        # self.processor = CLIPProcessor.from_pretrained(
        #     model_str, cache_dir=cache_dir)
        # self.model = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        self.device = 'cpu'
        self.image_encoder = HuggingFaceImageEncoder(model_str, cache_dir)
        self.text_encoder = HuggingFaceTextEncoder(model_str, cache_dir)
        self.model_str = model_str
        self.no_grad = True

    def to(self, device):
        self.image_encoder.to(device)
        # self.model.to(device)
        self.device = device
        return self

    @property
    def image_encoder_str(self):
        return f"HuggingFaceCLIP<{self.model_str}>.ImageEncoder"

    @property
    def text_encoder_str(self):
        return f"HuggingFaceCLIP<{self.model_str}>.TextEncoder"

    def set_no_grad(self, state: bool = True):
        self.no_grad = state
        return self

    def preprocess_images(self, images: List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts: List[str]):
        return self.text_encoder.preprocess(texts)
        
    def encode_image(self, image: torch.Tensor):
        # return self.model.get_image_features(pixel_values=image.to(device=self.device))
        return self.image_encoder(image)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.TensorType):
        # return self.model.get_text_features(input_ids=input_ids.to(device=self.device), attention_mask=attention_mask.to(device=self.device))
        return self.text_encoder(input_ids, attention_mask)

    def encode_images(self, images: Union[List[PILImage], PILImage], batch_size: Optional[int] = None,
                      device: str = 'cpu', verbose: bool = False, return_timing: bool = False) -> torch.Tensor:
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
                return_timing:bool, default `False`
                            if `True` return the cpu time rather than the result

            Returns
            -------
                if return_timing is `True`, will return a float number which is the process time
                else return the emb_images which is
                emb_images: torch.FloatTensor[n_image, n_emb] or [e_emb]
                            the embedding of the encoded images
                            the device should be in cpu, no matter what the device for the encoder
        """
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
                    emb_batch = self.encode_image(image)
            else:
                emb_batch = self.encode_image(image)
            if emb_batch.device != torch.device(device):
                emb_batch = emb_batch.to(device)
            emb_images.append(emb_batch)

        end_time = time.process_time()

        if return_timing:
            return end_time - start_time

        emb_images = torch.cat(emb_images, 0)

        if is_single:
            return emb_images[0]
        else:
            return emb_images

    def encode_texts(self, texts: Union[List[str], str], batch_size: Optional[int] = None, device: str = 'cpu',
                     verbose: bool = False, return_timing: bool = False) -> torch.Tensor:
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

        if batch_size is not None:
            texts = TextLoader(texts, batch_size)
        else:
            texts = [self.preprocess_texts([text]) for text in texts]

        if verbose:
            texts = tqdm(texts, total=len(texts), desc="Text Encoding")

        emb_texts = []

        for text in texts:
            if self.no_grad:
                with torch.no_grad():
                    emb_batch = self.encode_text(**text)
            else:
                emb_batch = self.encode_text(**text)
            if emb_batch.device != torch.device(device):
                emb_batch = emb_batch.to(device)
            emb_texts.append(emb_batch)

        emb_texts = torch.cat(emb_texts, 0)

        if is_single:
            return emb_texts[0]
        else:
            return emb_texts
