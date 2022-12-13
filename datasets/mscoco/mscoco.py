import json
import pathlib
import json
import re
import os
import pandas as pd
from typing import List, Union, Optional
from PIL import Image

CAPS_PATH = str(pathlib.Path(__file__).parent.resolve() / "annotations_trainval2017" / \
    "annotations" / "captions_val2017.json")
IMGS_DIR = str(pathlib.Path(__file__).parent.resolve() / "val2017")


class MSCOCO:
    def __init__(self, 
                n_samples:Optional[int]=None,
                layout:str =  "image-caption[0]",
                image_path:str=IMGS_DIR, 
                ann_path:str=CAPS_PATH):
        """
            there are 5000 images and 4~6 captions for each image. (25014 captions)
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
                            the path to coco image folder 
                ann_path:   str
                            the path to annotation json file 
        """
        assert os.path.isdir(image_path), f"The image_path should be directory"
        assert ann_path.endswith('.json'), f"The ann_path should ends with `.json`"
        assert layout == "caption" or re.sub("-?\d","",layout) == "image-caption[]"
        with open(ann_path,  'r') as f:
            ann = json.load(f)

        imageid2filename = {item['id']:item['file_name'] for item in ann['images']}
        
        if layout == "caption":
            if n_samples is not None:
                assert n_samples <= len(ann['annotations']), f"the number of samples should not be greater than the total length"
            else:
                n_samples = len(ann['annotations'])
            images   = []
            captions = []
            for i in range(n_samples):
                item = ann['annotations'][i]
                images.append(imageid2filename[item['image_id']])
                captions.append(item['caption'])
            
        else:
            caption_index = int(re.findall('-?\d',layout)[0])
            if n_samples is not None:
                assert n_samples <= len(ann['annotations']), f"the number of samples should not be greater than the total length"
            else:
                n_samples = len(ann['annotations'])
            df = pd.DataFrame.from_dict(ann['annotations'])
            images   = []
            captions = []
            for image_id, group in df.groupby('image_id'):
                images.append(imageid2filename[image_id])
                captions.append(group['caption'].iloc[caption_index])
                if len(images) >= n_samples:
                    break


        self._images   = [os.path.join(image_path, image) for image in images]
        self._captions = captions

    def __len__(self):
        return len(self._captions)
    @property
    def images(self):
        return [Image.open(image) for image in self._images]
    @property
    def captions(self):
        return self._captions


