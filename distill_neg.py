import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision  
from PIL import Image
from tqdm import tqdm 
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel 
import argparse


class CocoLoader(torch.utils.data.DataLoader):
    def __init__(self,
        batch_size:int=2, 
        n_neg=4, 
        image_path = "./.coco/val2017",
        ann_path   = "./.coco/annotations/captions_val2017.json",
        transformer_path = "./.transformer/clip-vit-base-patch32",
        ):
        self.coco      = torchvision.datasets.CocoCaptions(root=image_path,annFile=ann_path)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.n_neg     = n_neg
        super().__init__(
            self.coco,
            batch_size  =batch_size,
            shuffle     =True,
            drop_last   =False,
            collate_fn  =self.coco_collate_fn
        )
    def coco_collate_fn(self, items):
        images =[ ]
        for item in items:
            image = item[0]
            images.append(image)
            for _ in range(self.n_neg):
                images.append(self.coco[np.random.randint(0, len(self.coco))][0])
        index = np.arange(len(images)).astype(int)
        np.random.shuffle(index)
        batch = np.stack(self.processor(images=images)['pixel_values'])
        label = np.tile(np.array([1]+[0]*self.n_neg), len(items))
        batch = batch[index]
        label = label[index]
        batch = torch.tensor(batch)
        label = torch.tensor(label)
        return batch,label

class StudentNet(nn.Module):
    def __init__(self,
        mobilenet_path="./.mobilenet",
        num_layers=1):
        super().__init__()
        self.net = torchvision.models.mobilenet_v3_small(pretrained=True, model_dir=mobilenet_path)
        self.net.eval()
        project = [nn.Linear(576,512)]
        for _ in range(num_layers-1):
            project.append(nn.ReLU())
            project.append(nn.Linear(512,512))
        self.project = nn.Sequential(*project)
    def parameters(self):
        return self.project.parameters()
    def state_dict(self):
        return self.project.state_dict()
    def load_state_dict(self,state_dict):
        self.project.load_state_dict(state_dict)
    def save(self, path="./.mobilenet", prefix="",  affix=""): 
        torch.save(self.state_dict(),os.path.join(path, f"{prefix}project{affix}.pt"))
        return self
    @staticmethod
    def load(path="./.mobilenet", prefix="", affix=""):
        net = StudentNet()
        net.load_state_dict(torch.load(os.path.join(path,f"{prefix}project{affix}.pt")))
        return net
    def __call__(self, x):
        """
            Parameters
            ----------
                x       torch.Tensor[B, C, H, W] float32
            
            Returns
            -------
                torch.Tensor[B, 768] float32
        """
        with torch.no_grad():
            x = self.net.features(x)
            x = self.net.avgpool(x)
            x = x[:, :, 0, 0]
        x = self.project(x)
        return x 

class TeacherNet(nn.Module):
    def __init__(self,
        transformer_path="./.transformer/clip-vit-base-patch32"
        ):
        super().__init__()
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.model.eval()
    def __call__(self, x):
        """
            Parameters
            ----------
                x       torch.Tensor[B, C, H, W] float32
            
            Returns
            -------
                torch.Tensor[B, 768] float32
        """
        with torch.no_grad():
            x = self.model.vision_model(
                pixel_values = x,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False)[1]
            x = self.model.visual_projection(x)
            return x


def main(config):
    """
        Parameters
        ----------
            config:  configuration
                    contains

                    cuda:       bool,
                                whether use cuda 
                    n_neg:      int
                                number of negative sampling per positive sampling
                    batch_size: int,
                                batch size of dataloader
                    lr          float,
                                learning rate
                    epoch:      int,
                                epochs to train 
                    from_ckpt:  Union[str,None]
                                checkpoint path of training, if None, train from scratch
                    to_ckpt:    Union[str,None]
                                checkpoint path of training to save, if None, will not save 
                    save_path:  str
                                where to save the projection weight of the studentnet
                    ann_path:   str
                                path to the coco annotation json file
                    image_path: str
                                path to the directory of coco image 
                    transformer_path:str
                                path to the cache directory of transformer
                    mobilenet_path:str
                                path to the cache directory of mobilnet
    """
    studentnet = StudentNet(config.mobilenet_path,
                            config.n_layers
                            )
    teachernet = TeacherNet(config.transformer_path)
    dataloader = CocoLoader(config.batch_size,
                            config.n_neg,
                            config.image_path,
                            config.ann_path,
                            config.transformer_path
                            )
    optimizer  = torch.optim.Adam(studentnet.parameters(),lr=config.lr)
    loss_fn    = nn.CosineEmbeddingLoss()

    if config.from_ckpt is not None:
        state_dict = torch.load(config.from_ckpt)
        studentnet.load_state_dict(state_dict["studentnet"])
        optimizer.load_state_dict(state_dict["optimizer"])

    if config.cuda:
        studentnet.cuda()
        teachernet.cuda()

    with tqdm(total=config.epoch*len(dataloader), ncols=100) as _tqdm:
        for ep in range(config.epoch):
            _tqdm.set_description(f'epoch:{ep}/{config.epoch}')
            for batch,label in dataloader:
                if config.cuda:
                    batch = batch.cuda()
                    label = label.cuda()
                y_st = studentnet(batch)
                y_tr = teachernet(batch)
                loss = loss_fn(y_st, y_tr, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                _tqdm.set_postfix(loss=f'{loss.item():7.5f}')
                _tqdm.update(1)
    if config.to_ckpt:
        state_dict = {
            "studentnet":studentnet.state_dict(),
            "optimizer":optimizer.state_dict()
        }
        torch.save(state_dict, config.to_ckpt)
    studentnet.save(config.save_path,
                    affix=f"_{config.n_layers}_{config.n_neg}_{config.epoch}_{config.lr}")
    
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        action="store_true",
                        help="whether use cuda to train")
    parser.add_argument("--n_neg",
                        default=2,
                        help="number of negative sampling per pos sample")
    parser.add_argument("--n_layers",
                        default=1,
                        help="number of layers of the projection")
    parser.add_argument("--batch_size", 
                        default=2,          
                        help="batch size of train loader")
    parser.add_argument("--lr",
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--epoch",
                        default=10,
                        help="how many epochs to train")
    parser.add_argument("--from_ckpt",
                        default=None,
                        help="where to restore the ckpt, if None, just train from scratch")
    parser.add_argument("--to_ckpt",
                        default="./.ckpt/distll_neg_ckpt.pt",
                        help="where to save the ckpt, if None, just do not save ckpt")
    parser.add_argument("--save_path",
                        default="./.mobilenet/project_distill_neg.pt",
                        help="where to save the projection layer weight")
    parser.add_argument("--ann_path", 
                        default="./.coco/annotations/captions_val2017.json",
                        help="where the coco annotation json file")
    parser.add_argument("--image_path",
                        default="./.coco/val2017",
                        help="where the coco images folder")
    parser.add_argument("--transformer_path", 
                        default= "./.transformer/clip-vit-base-patch32",
                        help="where to cache transformer weight")
    parser.add_argument("--mobilenet_path",
                        default="./.mobilenet",
                        help="where to cache mobilenet weight")
    config = parser.parse_args()
    main(config)