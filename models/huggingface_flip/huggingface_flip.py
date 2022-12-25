import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List,Optional
from PIL.ImageFile import ImageFile as PILImage
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from ..huggingface_clip import HuggingFaceCLIP

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),".cache")


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class MAEImageEncoder(nn.Module):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        self.processor       = CLIPProcessor.from_pretrained(model_str, cache_dir=cache_dir)
        model                = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        vision               = model.vision_model
        embed                = vision.embeddings
        self.class_embed     = embed.class_embedding
        self.patch_embed     = embed.patch_embedding
        self.position_embed  = embed.position_embedding
        self.num_patches     = embed.num_patches
        self.pre_layernorm   = vision.pre_layrnorm
        self.encoder         = vision.encoder 
        self.post_layernorm  = vision.post_layernorm
        self.projection      = model.visual_projection
        self.p               = 1 - p 
        self.model_str       = model_str
    def preprocess(self, images:List[PILImage]):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]
    def forward(self, x:torch.Tensor):
        """
            x:      torch.FloatTensor[B,C,H,W]
        """
        B = x.shape[0]
        x = self.patch_embed(x) #[n_batch, n_hid, n_grid, n_grid]
        x = x.flatten(2)   #[n_batch, n_hid, n_grid**2]
        x = x.transpose(1,2) #[n_batch, n_grid**2, n_hid]
        m = torch.randperm(x.shape[1]) < int(self.p*x.shape[1])
        x = x[:, m]
        x = torch.cat([self.class_embed.expand(B, 1, -1), x], dim=1)
        x += self.position_embed(torch.cat([torch.where(m)[0], torch.tensor([self.num_patches])])[None,:])

        x = self.pre_layernorm(x)
        x = self.encoder(inputs_embeds=x)
        x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)

        x = self.projection(x)

        return x
        

class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, model_str:str, cache_dir:str=CACHE_DIR):
        super().__init__()
        self.tokenizer       = CLIPProcessor.from_pretrained(model_str, cache_dir=cache_dir)
        model                = CLIPModel.from_pretrained(model_str, cache_dir=cache_dir)
        text                 = model.text_model
        self.embed           = text.embeddings
        self.encoder         = text.encoder
        self.post_layernorm  = text.final_layer_norm
        self.projection      = model.text_projection
    def preprocess(self, texts:List[str]):
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

    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor):
        n_batch, seq_len = input_ids.size()
        x = self.embed(input_ids = input_ids)
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

        return  x

class HuggingFaceFLIP(HuggingFaceCLIP):
    def __init__(self, p:float, model_str:str, cache_dir:str=CACHE_DIR):
        super(HuggingFaceCLIP, self).__init__()
        self.image_encoder = MAEImageEncoder(p, model_str, cache_dir)
        self.text_encoder  = HuggingFaceTextEncoder(model_str, cache_dir)
        self.model_str = model_str 
        self.p         = p

    @property
    def image_encoder_str(self):
        return f"HuggingFaceFLIP<{self.model_str},{self.p}>.ImageEncoder"

    def preprocess_images(self, images:List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts:List[str]):
        return self.text_encoder.preprocess(texts)

    def encode_image(self, image:torch.Tensor):
        if image.dim() == 3:
            image = image[None, ...]
        return self.image_encoder(image)

    def encode_text(self, input_ids:torch.tensor, attention_mask:torch.Tensor):
        return self.text_encoder(input_ids, attention_mask)