from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from tqdm import tqdm
import torch
print(__package__)
from .main import

from .. import CLIPVisionTransformer as VisionTransformer
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
vision_model = model.vision_model
# torch.save(vision_model.state_dict(), 'vision_model.pt')
# model = VisionTransformer()
# model.load_state_dict(torch.load(PATH))
# model.eval()


