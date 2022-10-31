from tqdm import tqdm
import clip

class CLIPOpenAI():
    def __init__(self, model_str):
        self.model, self.preprocess = clip.load(model_str, device="cpu")

    def img_vecs(self, imgs):
        img_vecs = []
        for img in tqdm(imgs):
            image = self.preprocess(img).unsqueeze(0)
            image_features = self.model.encode_image(image)
            img_vecs.append(image_features.detach().squeeze().numpy())
        return img_vecs

    def cap_vecs(self, caps):
        cap_vecs = []
        for cap in tqdm(caps):
            cap = "a photo of " + cap
            text = clip.tokenize([cap])
            text_features = self.model.encode_text(text)
            cap_vecs.append(text_features.detach().squeeze().numpy())
        return cap_vecs