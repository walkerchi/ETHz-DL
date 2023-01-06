import csv 
import os
import re
import pandas as pd
import pathlib 
from typing import Optional
from PIL import Image

CAPS_PATH = str(pathlib.Path(__file__).parent.resolve() / "results.csv")
IMGS_DIR  = str(pathlib.Path(__file__).parent.resolve() / "flickr30k_images")


class Flickr30:
    def __init__(self, 
                n_samples:Optional[int]=None,
                layout:str = "image-caption[0]",
                image_path:str=IMGS_DIR, 
                cap_path:str=CAPS_PATH):
        """
            there are 31783 images and 5 captions for each image.
            Parameters
            ----------
                n_samples:  Optional[int], default `None`
                            if None, no truncate for the images and captions
                            if not None, it will do a lazy loading to limit the memory cost and loading time
                layout:     str, default `image-caption[0]`
                            `caption`:  the image is copied for each corresponding caption
                                        which means the total length will be 20,000 ~ 30,000       
                            `image-caption[i]`: get the i-th caption for each image,
                                        which means the total length will be 5000,
                                        the `image-caption[0]` is getting the first caption for each image.
                image_path: str
                            the path to flickr30 image folder 
                cap_path:   str
                            the path to caption csv file 
        """
        assert os.path.isdir(image_path), f"The image_path should be directory"
        assert cap_path.endswith('.csv'), f"The cap_path should ends with `.csv`"
        assert layout == "caption" or re.sub("-?\d","",layout) == "image-caption[]"
        df = pd.read_csv(cap_path, sep="\|\s")

        if layout == "caption":
            if n_samples is None:
                images   = df['image_name'].tolist()   
                captions = df['comment'].tolist()
            else:
                images   = df['image_name'][:n_samples].tolist()
                captions = df['comment'][:n_samples].tolist() 
        else:
            caption_index = int(re.findall('-?\d',layout)[0])
            images   = []
            captions = []
            for image, group in df.groupby('image_name'):
                images.append(image)
                captions.append(group['comment'].iloc[caption_index])
                if n_samples is not None and len(images) >= n_samples:
                    break
        self._images   = [os.path.join(image_path, image) for image in images]
        self._captions = captions

    def __len__(self):
        return len(self._captions)
    @property
    def images(self):
        imgs = []
        for image in self._images:
            img = Image.open(image)
            img.load()
            imgs.append(img)
        return imgs
    @property
    def captions(self):
        return self._captions

