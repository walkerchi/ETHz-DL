import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision  
from PIL import Image
from tqdm import tqdm 
import numpy as np
import transformers
from transformers import CLIPProcessor, CLIPModel 
from sklearn import metrics
import argparse
import os
import logging

class Cifar100Loader(torch.utils.data.DataLoader):
    def __init__(self, 
                batch_size       = 16,
                cifar100_path       = "./.dataset/cifar100",
                transformer_path = "./.transformer/clip-vit-base-patch32"
                ):
        self.cifar100  = torchvision.datasets.CIFAR100(root=cifar100_path,train=False,download=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.texts     = self.processor(text=self.cifar100.classes, return_tensors="pt", padding=True)
        super().__init__(self.cifar100, batch_size=batch_size, collate_fn=self.cifar100_collate_fn)
    def cifar100_collate_fn(self, batch):
        x, y = [], []
        for item in batch:
            x.append(item[0])
            y.append(item[1])
        x = self.processor(images=x,return_tensors="pt")
        x = dict(x, **self.texts)
        return x, torch.tensor(y).long()


class MobileCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = None 
        self.text_projection = None
        self.visual_model = None 
        self.visual_projection = None
    @staticmethod
    def from_assemble(num_layers=1, 
                    load_path="./.mobilenet/project.pt",
                    transformer_path="./.transformer/clip-vit-base-patch32",
                    mobilenet_path="./.mobilenet"
                    ):
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        class MobileNet(nn.Module):
            def __init__(self, mobilenet_path):
                super().__init__()
                net = torchvision.models.mobilenet_v3_small(pretrained=True, model_dir=mobilenet_path)
                self.features = net.features
                self.avgpool  = net.avgpool
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = x[:, :, 0, 0]
                return None,x
        class MobileNetProjection(nn.Module):
            def __init__(self, num_layers, load_path):
                super().__init__()
                project = [nn.Linear(576, 512)]
                for i in range(num_layers-1):
                    project.append(nn.ReLU())
                    project.append(nn.Linear(512,512))
                self.project  = nn.Sequential(*project)
                self.project.load_state_dict(torch.load(load_path))
            def forward(self, x):
                x = self.project(x)
                return x

        mobileclip = MobileCLIP()
        mobileclip.text_model = clip.text_model
        mobileclip.text_projection = clip.text_projection
        mobileclip.visual_model = MobileNet(mobilenet_path)
        mobileclip.visual_projection = MobileNetProjection(num_layers,load_path)
        return mobileclip
    @staticmethod
    def from_pretrained(self, path="./.mobileclip/mobileclip.pt"):
        return torch.load(path)
    def save(self, path="./.mobileclip/mobileclip.pt"):
        torch.save(self, path)
        return self
    def visual_encode(self, x, is_norm=True):
        x = self.visual_model(x)
        x = x[1]
        x = self.visual_projection(x)
        if is_norm:
            x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x
    def text_encode(self, x, m, is_norm=True):
        x = self.text_model(x, m)
        x = x[1]
        x = self.text_projection(x)
        if is_norm:
            x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x
    def forward(self, x):
        x_vision = self.visual_encode(x['pixel_values'])
        x_text   = self.text_encode(x['input_ids'], x['attention_mask'])
        return x_vision @ x_text.T


def main(config):
    mobileclip = MobileCLIP.from_assemble(config.n_layers, config.load_path, config.transformer_path, config.mobilenet_path)
    dataloader = Cifar100Loader(config.batch_size, config.cifar100_path, config.transformer_path)

    def to_cuda(x):
        if isinstance(x, (torch.Tensor,torch.nn.Module)):
            return x.cuda()
        elif isinstance(x, (dict,transformers.tokenization_utils_base.BatchEncoding)):
            for k, v in x.items():
                x[k] = to_cuda(v)
            return x 
        elif isinstance(x, (list, tuple)):
            return [to_cuda(i) for i in x]
        else:
            print(type(x))
            print(x)
            raise NotImplementedError()
    
    
    if config.cuda:
        mobileclip.cuda()

    with torch.no_grad():
        l_metrics = [[] for _ in config.topk]
        for batch_x, batch_y in tqdm(dataloader):
            if config.cuda:
                batch_x = to_cuda(batch_x)
            logits  = mobileclip(batch_x)
            for i in range(len(config.topk)):
                metric  = metrics.top_k_accuracy_score(batch_y.cpu().numpy(),logits.cpu().numpy(),k=config.topk[i],labels=np.arange(100))
                l_metrics[i].append(metric)
        l_metrics = np.array(l_metrics)
        for i, k in enumerate(config.topk):
            logging.info(f"top{k}:{l_metrics[i].mean()}({l_metrics[i].std()})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        action="store_true",
                        help="whether use cuda to verify")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="batch size of data loader")
    parser.add_argument("--topk",
                        type=str,
                        default="1,3,5,10,15,20",
                        help="the k of topk metrics")
    parser.add_argument("--load_path",
                        type=str,
                        default="./.mobilenet/project_2.pt",
                        help="pt weight file of projection")
    parser.add_argument("--n_layers",
                        type=int,
                        default=1,
                        help="number of layers of projection")
    parser.add_argument("--cifar100_path",
                        type=str,
                        default="./.dataset/cifar100")
    parser.add_argument("--transformer_path",
                        type=str,
                        default="./.transformer/clip-vit-base-patch32")
    parser.add_argument("--mobilenet_path",
                        type=str,
                        default="./.mobilenet")
    parser.add_argument("--log_path",
                        type=str,
                        default="./.log/somelog.log",
                        help="log path")
    config = parser.parse_args()
    config.topk = [int(i) for i in config.topk.split(',')]
    if config.log_path:
        if not os.path.exists(os.path.dirname(config.log_path)):
            os.mkdir(os.path.dirname(config.log_path))
        logging.basicConfig(
            level = logging.DEBUG,
            filename=config.log_path,
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logging.info(config)
    if not os.path.exists(config.mobilenet_path):
        os.mkdir(config.mobilenet_path)
    if not os.path.exists(config.transformer_path):
        os.mkdir(config.transformer_path)
    if not os.path.exists(config.cifar100_path):
        os.mkdir(config.cifar100_path)
    main(config)