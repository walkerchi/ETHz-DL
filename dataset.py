import torch
import torchvision
import json
import glob 
from typing import List
from PIL import Image,ImageFile
import os
import numpy as np

PILImage = ImageFile.ImageFile

class ImageDataset:
    @classmethod
    def from_dir(cls, path:str):
        return cls(glob.glob(f"{path}/*"))
    def __init__(self, paths:List[str]):
        self.paths = paths
    def __getitem__(self, index:int)->PILImage:
        img =  Image.open(self.paths[index])
        if img.mode == "L":
            img = img.convert("RGB")
        return img
    def __len__(self):
        return  len(self.paths)

class ImageLoader(torch.utils.data.DataLoader):
    @classmethod
    def from_dir(cls, path:str, batch_size:int=4):
        dataset = ImageDataset.from_dir(path)
        return cls(dataset, batch_size)
    def __init__(self, dataset:ImageDataset, batch_size:int=4):
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            drop_last  = False
        )
        self.n_batches = int((len(self.dataset) + self.batch_size - 1)//self.batch_size)
    def subloader(self, index:torch.Tensor):
        paths = [self.dataset.paths[i] for i in index]
        dataset = ImageDataset(paths)
        return ImageLoader(
            dataset,
            batch_size=self.batch_size,
        )
    def __iter__(self):
        self.counter = 0 
        return self 
    def __len__(self):
        return self.n_batches
    def __next__(self):
        if self.counter >= len(self):
            raise StopIteration()
        else:
            imgs = []
            for i in range(self.batch_size):
                i = self.counter*self.batch_size+i
                if i >= len(self.dataset):
                    break
                imgs.append(self.dataset[i])
            self.counter += 1
            return imgs


class CocoImage:
    def __init__(self,
        image_path = "./.coco/val2017",
        ann_path   = "./.coco/annotations/captions_val2017.json",
        ):
        ann = json.load(open(ann_path,"r"))
        imageids = sorted([x['id'] for x in ann['images']])
        imageid2index = {imageids[i]:i for i in range(len(imageids))}
        paths    = [None for _ in range(len(imageids))]
        for item in ann['images']:
            paths[imageid2index[item['id']]] = os.path.join(image_path, item['file_name'])
        self._paths = paths
    def __getitem__(self, index:int):
        img = Image.open(self._paths[index])
        if img.mode == "L":
            img = img.convert("RGB")
        return img
    def __len__(self):
        return len(self._paths)
    @property
    def paths(self):
        return self._paths 

class CocoText:
    def __init__(self, 
        ann_path="./.coco/annotations/captions_val2017.json"
        ):
        ann = json.load(open(ann_path,"r"))
        imageids = sorted([x['id'] for x in ann['images']])
        imageid2index = {imageids[i]:i for i in range(len(imageids))}
        texts    = [[] for _ in range(len(imageids))]
        for item in ann['annotations']:
            texts[imageid2index[item['image_id']]].append(item['caption'])
        for i, t in enumerate(texts):
            texts[i] = tuple(t)
        self._texts = texts
        flat_mask  = []
        for i,texts in enumerate(self._texts):
            flat_mask.extend([i for _ in range(len(texts))])
        self.mask = flat_mask
    def __getitem__(self, index:int):
        return self._texts[index]
    def first(self):
        first_texts = [i[0] for i in self._texts]
        return first_texts
    def flatten(self):
        flat_texts = []
        for texts in self._texts:
            flat_texts.extend(texts)
        return flat_texts
        
if __name__ == '__main__':
    # from tqdm import tqdm
    # from model import CLIP
    # coco_text = CocoText()
    # texts, mask = coco_text.flatten()
    # emb = CLIP.text_encode(texts)
    pass