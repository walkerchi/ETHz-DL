import os
import time
import logging
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional
from transformers import CLIPProcessor
from PIL.ImageFile import ImageFile as PILImage
from torch.utils.data import DataLoader
from ..huggingface_clip import HuggingFaceCLIP
from ..huggingface_flip import HuggingFaceTextEncoder

CACHE_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".cache")
DISTILL_DIR = os.path.join(CACHE_DIR, "distill")


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


class DistillImageEncoder(nn.Module):
    def __init__(self, model_str: str,
                 net: str = "mobilenet_v3_small",
                 n_layers: int = 1,
                 n_hidden: int = 512,
                 cache_dir: str = CACHE_DIR,
                 epoch: Optional[int] = None,
                 lr: Optional[float] = None
                 ):
        super().__init__()
        self.model_str = model_str
        self.net_name = net
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # save old environment
        torch_home = os.environ["TORCH_HOME"] if "TORCH_HOME" is os.environ else None
        # change the downloading path
        os.environ["TORCH_HOME"] = cache_dir

        if net == "mobilenet_v3_small":
            net = torchvision.models.mobilenet_v3_small(pretrained=True)
            self.net = nn.Sequential(net.features, net.avgpool, Squeeze())
            n_dim = 576
        elif net == "mobilenet_v3_large":
            net = torchvision.models.mobilenet_v3_large(pretrained=True)
            self.net = nn.Sequential(net.features, net.avgpool, Squeeze())
            n_dim = 960
        elif net == "resnet_18":
            net = torchvision.models.resnet18(pretrained=True)
            self.net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                     net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool, Squeeze())
            n_dim = 512
        else:
            raise Exception(f"{net} is currently not supported")

        if torch_home is None:           # restore the environment
            os.environ.pop("TORCH_HOME")
        else:
            os.environ["TORCH_HOME"] = torch_home

        self.net.eval()
        if n_layers == 1:
            self.projection = nn.Linear(n_dim, 512)
        else:
            assert n_layers >= 2
            projection = [nn.Linear(n_dim, n_hidden)]
            for _ in range(n_layers - 2):
                projection.extend([nn.ReLU(), nn.Linear(n_hidden, n_hidden)])
            projection.extend([nn.ReLU(), nn.Linear(n_hidden, 512)])
            self.projection = nn.Sequential(*projection)

        self.processor = CLIPProcessor.from_pretrained(
            model_str, cache_dir=cache_dir)

        if epoch is None and lr is None:
            self.load()
            print(
                "\x1b[36mweight file loaded, you can do either distilling again or do inference on it.\x1b[37m")
        else:
            print(
                "\x1b[33mweight file not found, make sure you are distilling now.\x1b[37m")
            assert epoch is not None and lr is not None
            self.epoch = epoch
            self.lr = lr

    @property
    def name(self):
        return f"{self.model_str},{self.net_name},{self.n_hidden},{self.n_layers}"

    @property
    def filename(self):
        return f"{self.model_str.replace('/','-')}_{self.net_name}_{self.n_hidden}_{self.n_layers}.pt"

    def preprocess(self, images: List[PILImage]):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def forward(self, x: torch.Tensor):
        """
            x:      torch.FloatTensor[B,C,H,W]
        """
        with torch.no_grad():
            x = self.net(x)
        x = self.projection(x)
        return x

    def save(self, path: str = DISTILL_DIR):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.projection.state_dict(),
                   os.path.join(path, self.filename))

    def load(self, path: str = DISTILL_DIR):
        path = os.path.join(path, self.filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Weight File not exist, You should distill it first to have the weight")
        self.projection.load_state_dict(torch.load(path))
        return self

    def distill(self, teacher: nn.Module, images: List[PILImage], batch_size: Optional[int], device: str = "cpu", verbose: bool = True, logger=logging):

        if batch_size is not None:
            images = DataLoader(self.preprocess(images), batch_size=batch_size)

        teacher.eval()
        self.train()

        optimizer = torch.optim.Adam(self.projection.parameters(), lr=self.lr)

        for ep in range(self.epoch):
            if verbose:
                loader = tqdm(images, total=len(images),
                              desc=f"Epoch[{ep:2}/{self.epoch}]")
            else:
                loader = images

            l_loss = []
            start = time.process_time()

            for x in loader:
                if x.device != torch.device(device):
                    x = x.to(device)

                with torch.no_grad():
                    y_tr = teacher(x)
                y_st = self(x)

                loss = F.cosine_embedding_loss(
                    y_tr, y_st, torch.ones(len(y_tr)).to(device))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                l_loss.append(loss.item())

            end = time.process_time()
            l_loss = np.array(l_loss)
            logger.info(
                f"Epoch[{ep}/{self.epoch}] loss:{l_loss.mean():5.3f}({l_loss.std():5.2f}) time:{end-start:7.3f}s")

        self.save()


class HuggingFaceDistillCLIP(HuggingFaceCLIP):
    def __init__(self, model_str: str, cache_dir: str = CACHE_DIR, net: str = "mobilenet_v3_small", **kwargs):
        super(HuggingFaceCLIP, self).__init__()
        self.image_encoder = DistillImageEncoder(
            model_str, net=net, cache_dir=cache_dir, **kwargs)
        self.text_encoder = HuggingFaceTextEncoder(model_str, cache_dir)
        self.model_str = model_str

    @property
    def image_encoder_str(self):
        return f"HuggingFaceDistill<{self.image_encoder.name}>.ImageEncoder"

    def preprocess_images(self, images: List[PILImage]):
        return self.image_encoder.preprocess(images)

    def preprocess_texts(self, texts: List[str]):
        return self.text_encoder.preprocess(texts)

    def encode_image(self, image: torch.Tensor):
        return self.image_encoder(image)

    def encode_text(self, input_ids: torch.tensor, attention_mask: torch.Tensor):
        return self.text_encoder(input_ids, attention_mask)
