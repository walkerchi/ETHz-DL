import json
import pathlib
from PIL import Image

CAPS_PATH = pathlib.Path(__file__).parent.resolve() / "annotations_trainval2017" / \
    "annotations" / "captions_val2017.json"
IMGS_DIR = pathlib.Path(__file__).parent.resolve() / "val2017"

class MSCOCO():
    def __init__(self, n_samples):
        self.n_samples = n_samples
        with open(CAPS_PATH) as f:
            j = json.load(f)
        self.samples = [(x["image_id"], x["caption"]) for x in j['annotations']]
        self.samples = self.samples[:self.n_samples]
        self.captions = [x[1] for x in self.samples]
        self.images = [self._get_image(x[0]) for x in self.samples]

    def _get_image(self, image_id):
        img = Image.open(IMGS_DIR / (str(image_id).zfill(12)+".jpg"))
        img.load()
        ret_img = img.copy()
        img.close()
        return ret_img