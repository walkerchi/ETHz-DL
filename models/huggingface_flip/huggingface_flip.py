import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional
from PIL.ImageFile import ImageFile as PILImage
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from ..huggingface_clip import HuggingFaceCLIP, HuggingFaceTextEncoder

CACHE_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".cache")


class MAEImageEncoder(nn.Module):
    def __init__(self, p: float, model_str: str, cache_dir: str = CACHE_DIR):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(
            model_str, cache_dir=cache_dir)
        model = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        vision = model.vision_model
        embed = vision.embeddings
        self.class_embed = embed.class_embedding
        self.patch_embed = embed.patch_embedding
        self.position_embed = embed.position_embedding
        self.num_patches = embed.num_patches
        self.pre_layernorm = vision.pre_layrnorm
        self.encoder = vision.encoder
        self.post_layernorm = vision.post_layernorm
        self.projection = model.visual_projection
        self.p = 1 - p
        self.model_str = model_str

    def preprocess(self, images: List[PILImage]):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, x: torch.Tensor):
        """
            x:      torch.FloatTensor[B,C,H,W]
        """
        B = x.shape[0]
        x = x.type(self.dtype).to(self.device)
        x = self.patch_embed(x)  # [n_batch, n_hid, n_grid, n_grid]
        x = x.flatten(2)  # [n_batch, n_hid, n_grid**2]
        x = x.transpose(1, 2)  # [n_batch, n_grid**2, n_hid]
        m = torch.randperm(x.shape[1]) < int(self.p*x.shape[1])
        x = x[:, m]
        x = torch.cat([self.class_embed.expand(B, 1, -1), x], dim=1)
        x += self.position_embed(torch.cat([torch.where(m)[0], torch.tensor([self.num_patches])])[None, :].to(self.device))

        x = self.pre_layernorm(x)
        x = self.encoder(inputs_embeds=x)
        x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)

        x = self.projection(x)

        return x


class HuggingFaceFLIP(HuggingFaceCLIP):
    def __init__(self, p: float, model_str: str, cache_dir: str = CACHE_DIR):
        super(HuggingFaceCLIP, self).__init__()
        self.image_encoder = MAEImageEncoder(p, model_str, cache_dir)
        self.text_encoder = HuggingFaceTextEncoder(model_str, cache_dir)
        self.model_str = model_str
        self.p = p

    @property
    def image_encoder_str(self):
        return f"HuggingFaceFLIP<{self.model_str},{self.p}>.ImageEncoder"

    def preprocess_images(self, images: List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts: List[str]):
        return self.text_encoder.preprocess(texts)

    def encode_image(self, image: torch.Tensor):
        return self.image_encoder(image)

    def encode_text(self, input_ids: torch.tensor, attention_mask: torch.Tensor):
        return self.text_encoder(input_ids, attention_mask)
