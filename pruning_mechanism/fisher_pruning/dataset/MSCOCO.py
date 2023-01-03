import json
import pathlib
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from torch.utils.data import Dataset


CAPS_PATH = pathlib.Path(__file__).parent.resolve() / \
    "annotations_trainval2017" / \
    "annotations" / "captions_train2017.json"
IMGS_DIR = pathlib.Path(__file__).parent.resolve() / "train2017"


class MSCOCO(Dataset):
    """Pytorch Dataset implementation of mscoco."""

    def __init__(self, n_samples, name, offset=0):
        """
        :param n_samples (int): The number of samples contained by the dataset
        :param name (string): Name of the HuggingFace CLIP model
        :param offset (int): Starting index of used samples in mscoco
        """
        self.n_samples = n_samples
        with open(CAPS_PATH) as f:
            j = json.load(f)
        self.model_og = CLIPModel.from_pretrained(name)
        self.processor = CLIPProcessor.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.samples = [(x["image_id"], x["caption"]) for x in j['annotations']]
        self.samples = self.samples[offset:offset+self.n_samples]
        self.captions = [x[1] for x in self.samples]
        self.images = [self._get_image(x[0]) for x in self.samples]
        self.teacher = False

    def __getitem__(self, index):
        processed_img = self.processor(images=self.images[index],
                                       return_tensors='pt')
        processed_txt = self.tokenizer(self.captions[index], padding=True,
                                       return_tensors="pt")
        if self.teacher:
            return processed_img, self.model_og.get_image_features(**processed_img)
        return processed_img, self.model_og.get_text_features(**processed_txt)

    def _get_image(self, image_id):
        img = Image.open(IMGS_DIR / (str(image_id).zfill(12)+".jpg"))
        img.load()
        ret_img = img.copy()
        img.close()
        return ret_img

    def __len__(self):
        return len(self.samples)
