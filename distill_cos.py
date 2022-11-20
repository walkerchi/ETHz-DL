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
import logging


class CocoLoader(torch.utils.data.DataLoader):
    def __init__(self,
        batch_size:int=2, 
        image_path = "./.coco/val2017",
        ann_path   = "./.coco/annotations/captions_val2017.json",
        transformer_path = "./.transformer/clip-vit-base-patch32",
        ):
        self.coco      = torchvision.datasets.CocoCaptions(root=image_path,annFile=ann_path)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
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
        batch = np.stack(self.processor(images=images)['pixel_values'])
        batch = torch.tensor(batch)
        return batch

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
    def save(self, path="./.mobilenet/project.pt"): 
        torch.save(self.state_dict(),path)
        return self
    @staticmethod
    def load(path="./.mobilenet/project.pt"):
        net = StudentNet()
        net.load_state_dict(torch.load(path))
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
                            config.image_path,
                            config.ann_path,
                            config.transformer_path
                            )
    optimizer  = torch.optim.Adam(studentnet.parameters(),lr=config.lr)
    loss_fn    = lambda x,y: F.cosine_embedding_loss(x, y, torch.ones(len(x)).to(x.device))

    if config.cuda:
        studentnet.cuda()
        teachernet.cuda()


    if config.from_ckpt is not None:
        state_dict = torch.load(config.from_ckpt)
        studentnet.load_state_dict(state_dict["studentnet"])
        optimizer.load_state_dict(state_dict["optimizer"])


    with tqdm(total=config.epoch*len(dataloader), ncols=100) as _tqdm:
        for ep in range(config.epoch):
            _tqdm.set_description(f'epoch:{ep}/{config.epoch}')
            l_loss = []
            for batch in dataloader:
                if config.cuda:
                    batch = batch.cuda()
                y_st = studentnet(batch)
                y_tr = teachernet(batch)
                loss = loss_fn(y_st, y_tr)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                l_loss.append(loss.item())
                _tqdm.set_postfix(loss=f'{loss.item():7.5f}')
                _tqdm.update(1)
            if config.log_path:
                l_loss = np.array(l_loss)
                logging.info(f"[Epoch{ep}/{config.epoch}] loss:{l_loss.mean():7.5f}({l_loss.std():7.5f})")
    if config.to_ckpt:
        state_dict = {
            "studentnet":studentnet.state_dict(),
            "optimizer":optimizer.state_dict()
        }
        torch.save(state_dict, config.to_ckpt)
    studentnet.save(config.save_path)
    
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        action="store_true",
                        help="whether use cuda to train")
    parser.add_argument("--n_layers",
                        type=int,
                        default=1,
                        help="number of layers of the projection")
    parser.add_argument("--batch_size", 
                        type=int,
                        default=2,          
                        help="batch size of train loader")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="learning rate")
    parser.add_argument("--epoch",
                        type=int,
                        default=10,
                        help="how many epochs to train")
    parser.add_argument("--from_ckpt",
                        type=str,
                        default=None,
                        help="where to restore the ckpt, if None, just train from scratch")
    parser.add_argument("--to_ckpt",
                        type=str,
                        default="./.ckpt/distll_neg_ckpt.pt",
                        help="where to save the ckpt, if None, just do not save ckpt")
    parser.add_argument("--save_path",
                        type=str,
                        default="./.mobilenet/project_ly2_ep1.pt",
                        help="where to save the projection layer weight")
    parser.add_argument("--ann_path", 
                        type=str,
                        default="./.coco/annotations/captions_val2017.json",
                        help="where the coco annotation json file")
    parser.add_argument("--image_path",
                        type=str,
                        default="./.coco/val2017",
                        help="where the coco images folder")
    parser.add_argument("--transformer_path", 
                        type=str,
                        default= "./.transformer/clip-vit-base-patch32",
                        help="where to cache transformer weight")
    parser.add_argument("--mobilenet_path",
                        type=str,
                        default="./.mobilenet",
                        help="where to cache mobilenet weight")
    parser.add_argument("--log_path",
                        type=str,
                        default=None,
                        help="where to set log")
    config = parser.parse_args()
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
    if config.to_ckpt:
        if not os.path.exists(os.path.dirname(config.to_ckpt)):
            os.mkdir(os.dirname(config.to_ckpt))
    main(config)