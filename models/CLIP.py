from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class CLIP():
    def __init__(self, size):
        if size == 'big':
            model_str = "openai/clip-vit-large-patch14"
        elif size == 'small':
            model_str = "openai/clip-vit-base-patch32"
        else:
            raise Exception(f"Unkown size '{size}'.")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_str)
        self.model = CLIPModel.from_pretrained(model_str)
        self.processor = CLIPProcessor.from_pretrained(model_str)

    def img_vecs(self, imgs):
        img_vecs = []
        for img in tqdm(imgs):
            inputs = self.processor(images=img, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            img_vecs.append(image_features.detach().squeeze().numpy())
        return img_vecs

    def cap_vecs(self, caps):
        cap_vecs = []
        for cap in tqdm(caps):
            cap = "a photo of " + cap
            inputs = self.tokenizer(cap, padding=True, return_tensors="pt")
            text_features = self.model.get_text_features(**inputs)
            cap_vecs.append(text_features.detach().squeeze().numpy())
        return cap_vecs
