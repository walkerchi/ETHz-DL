import json
import pathlib
from PIL import Image
import torch
import torchvision
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset


CAPS_PATH = pathlib.Path(__file__).parent.resolve() / "annotations_trainval2017" / \
    "annotations" / "captions_val2017.json"
IMGS_DIR = pathlib.Path(__file__).parent.resolve() / "val2017"

class MSCOCO(Dataset):
    def __init__(self, n_samples, name):
        self.n_samples = n_samples
        with open(CAPS_PATH) as f:
            j = json.load(f)
        self.model_og = CLIPModel.from_pretrained(name)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.samples = [(x["image_id"], x["caption"]) for x in j['annotations']]
        self.samples = self.samples[:self.n_samples]
        self.captions = [x[1] for x in self.samples]
        self.images = [self._get_image(x[0]) for x in self.samples]

    def __getitem__(self, index):
        processed = self.processor(images=self.images[index], return_tensors='pt')
        return  processed, self.model_og.get_image_features(**processed)


    def _get_image(self, image_id):
        img = Image.open(IMGS_DIR / (str(image_id).zfill(12)+".jpg"))
        img.load()
        ret_img = img.copy()
        img.close()
        return ret_img

    def __len__(self):
        return len(self.samples)

