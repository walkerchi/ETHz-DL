import torch 
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
from typing import List,Union
from transformers import CLIPProcessor, CLIPModel 

from dataset import PILImage,ImageLoader


class MobileNetImageEncoder(nn.Module):
    def __init__(self, n_layers:int, mobilenet_path:str="./.mobilenet"):
        super().__init__()
        net = torchvision.models.mobilenet_v3_small(pretrained=True, model_dir=mobilenet_path)
        self.features = net.features 
        self.avgpool  = net.avgpool
        project = [nn.Linear(576, 512)]
        for _ in range(n_layers-1):
            project.append(nn.ReLU())
            project.append(nn.Linear(512,512))
        self.project  = nn.Sequential(*project)
    def load(self, path:str):
        self.project.load_state_dict(torch.load(path))
    def save(self, path:str):
        torch.save(self.project.state_dict(),path)
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x = self.features(x)
                x = self.avgpool(x)
                x = x[:, :, 0, 0]
            x = self.project(x)
        else:
            with torch.no_grad():
                x = self.features(x)
                x = self.avgpool(x)
                x = x[:, :, 0, 0]
                # print(x.shape)
                x = self.project(x)
        return x
        
class CLIPImageEncoder(nn.Module):
    def __init__(self, transformer_path="./.transformer/clip-vit-base-patch32"):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
    def forward(self, x, norm:bool=True):
        with torch.no_grad():
            x = self.vision_model(x)
            x = x[1]
            x = self.visual_projection(x)
            if norm:
                x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

class CLIPTextEncoder(nn.Module):
    def __init__(self, transformer_path:str="./.transformer/clip-vit-base-patch32"):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection 
    def forward(self, x, m, norm:bool=True):
        with torch.no_grad():
            x = self.text_model(x)
            x = x[1]
            x = self.text_projection(x)
            if norm:
                x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x



class CLIPBase(nn.Module):
    def __init__(self, transformer_path:str="./.transformer/clip-vit-base-patch32"):
        super().__init__()
        self.processor     = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=transformer_path)
        self.text_encoder  = CLIPTextEncoder(transformer_path)
    @property
    def device(self):
        return next(iter(self.parameters())).device

    def preprocess_image(self, images:List[PILImage])->torch.Tensor:
        try:
            x = self.processor(images=images, return_tensors="pt")['pixel_values']
        except Exception as e:
            for image in images:
                print(image.size)
                print(np.array(image).shape)
            print(self.processor(images=images, return_tensors="pt"))
            exit()
        return x.to(self.device)
    def preprocess_text(self, texts:List[str])->torch.Tensor:
        x = self.processor(text=texts, return_tensors="pt",padding=True)
        return x['input_ids'].to(self.device), \
                x['attention_mask'].to(self.device)
    def indices_to_images(self, indices:torch.Tensor, images:Union[torch.utils.data.DataLoader,List[PILImage]]):
        n_images, n_texts = indices.shape
        results = [[] for _ in range(n_texts)]
        for i in range(n_images):
            for j in range(n_texts):
                if isinstance(images, torch.utils.data.DataLoader):
                    results[j].append(images.dataset[indices[i,j]])
                else:
                    results[j].append(images[indices[i,j]])
        return results
    def text_encode(self, texts:Union[torch.utils.data.DataLoader,List[str]], verbose:bool=False):
        if isinstance(texts, torch.utils.data.DataLoader):
            texts_emb = []
            if verbose:
                texts = tqdm(texts, desc="Text Encode", total=len(texts))
            for batch in texts:
                texts_th, mask = self.preprocess_text(batch)
                texts_emb.append(self.text_encoder(texts_th, mask).cpu())
            texts_emb = torch.cat(texts_emb, 0)
        else:
            texts_th, mask = self.preprocess_text(texts)
            texts_emb = self.text_encoder(texts_th, mask).cpu()
        return texts_emb

class CLIP(CLIPBase):
    def __init__(self, transformer_path:str="./.transformer/clip-vit-base-patch32"):
        super().__init__(transformer_path)
        self.image_encoder = CLIPImageEncoder(transformer_path)
    def image_encode(self, images:Union[torch.utils.data.DataLoader,List[PILImage]],verbose:bool=False):
        if isinstance(images, torch.utils.data.DataLoader):
            l_batch_images_emb = []
            if verbose:
                images = tqdm(images,desc="Image Encoding", total=len(images))
            for batch_images in images:
                batch_images_th = self.preprocess_image(batch_images)
                batch_images_emb= self.image_encoder(batch_images_th).cpu()
                l_batch_images_emb.append(batch_images_emb)
            images_emb = torch.cat(l_batch_images_emb, 0)
        else:
            images_th  = self.preprocess_image(images)
            images_emb = self.image_encoder(images_th).cpu()
        return images_emb
    def topk_images(self,
                    images:Union[torch.utils.data.DataLoader,List[PILImage]], 
                    texts:Union[torch.utils.data.DataLoader,List[str]], 
                    topk:int=1,
                    return_index:bool=False,
                    return_text_emb:bool=False,
                    verbose:bool=False)->Union[List[List[PILImage]],torch.LongTensor]:
        """
            Parameters
            ----------
                image:      torch.utils.data.DataLoader->List[PIL.Image] or List[PIL.Image]
                texts:      torch.utils.data.DataLoader->List[str] or List[str]
                topk:       int
                return_index:bool       default:False
                            whether return the index of the image 
                            or the PIL.Image of the indexed Image
                return_text_emb:bool    default:False
                            whether return the embedding of the text
            Returns
            -------
                List[List[PIL.Image]]   [n_text,[topk,[PIL.Image]]]
        """
        if verbose:
            print("Image Encoding...")
        images_emb          = self.image_encode(images,verbose)
        if verbose:
            print("Text Encoding")
        texts_emb           = self.text_encode(texts, verbose)
        
        image_per_text      = images_emb @ texts_emb.T
        topk_indices        = image_per_text.topk(topk, dim=0).indices
        if return_index:
            if return_text_emb:
                return topk_indices.T, texts_emb
            else:
                return topk_indices.T
        else:
            if return_text_emb:
                return self.indices_to_images(topk_indices, images), texts_emb
            else:
                return self.indices_to_images(topk_indices, images)



class CascadeCLIP(CLIPBase):
    def __init__(self, n_layers:int = 2, 
                    mobilenet_projection_path:str = "./.mobilenet/project_coco_train_ep1_ly2.pt",
                    mobilenet_path:str="./.mobilenet",
                    transformer_path:str="./.transformer/clip-vit-base-patch32"):
        super().__init__(transformer_path)
        self.small_image_encoder = MobileNetImageEncoder(n_layers, mobilenet_path)
        self.small_image_encoder.load(mobilenet_projection_path)
        self.large_image_encoder  = CLIPImageEncoder(transformer_path)
    
    def small_image_encode(self, images:Union[torch.utils.data.DataLoader,List[PILImage]],verbose:bool=False):
        if isinstance(images, torch.utils.data.DataLoader):
            l_batch_images_emb = []
            if verbose:
                images = tqdm(images, desc="Small Image Encoder:")
            for batch_images in images:
                batch_images_th = self.preprocess_image(batch_images)
                batch_images_emb= self.small_image_encoder(batch_images_th).cpu()
                l_batch_images_emb.append(batch_images_emb)
            images_emb = torch.cat(l_batch_images_emb, 0)
        else:
            images_th  = self.preprocess_image(images)
            images_emb = self.small_image_encoder(images_th)
        return images_emb
    def large_image_encode(self, images:Union[torch.utils.data.DataLoader,List[PILImage]],verbose:bool=False):
        if isinstance(images, torch.utils.data.DataLoader):
            l_batch_images_emb = []
            if verbose:
                images = tqdm(images, desc="Large Image Encoder")
            for batch_images in images:
                batch_images_th = self.preprocess_image(batch_images)
                batch_images_emb= self.large_image_encoder(batch_images_th).cpu()
                l_batch_images_emb.append(batch_images_emb)
            images_emb = torch.cat(l_batch_images_emb, 0)
        else:
            images_th  = self.preprocess_image(images)
            images_emb = self.large_image_encoder(images_th).cpu()
        return images_emb

    def filter_images(self, 
                        indices:torch.Tensor, 
                        images:Union[torch.utils.data.DataLoader,List[PILImage]]):
        index = indices.unique().tolist()
        
        if isinstance(images,torch.utils.data.DataLoader):
            return images.subloader(index)
        else:
            new_images = []
            for i in index:
                new_images.append(images[i])
            return new_images
    def topk_images(self, 
                    images:Union[torch.utils.data.DataLoader,List[PILImage]], 
                    texts:Union[torch.utils.data.DataLoader, List[str]], 
                    topk:int=1, 
                    topm:int=20,
                    return_index:bool=False,
                    return_text_emb:bool=False,
                    verbose:bool=False)->Union[List[List[PILImage]],torch.LongTensor]:
        """
            Parameters
            ----------
                image:      torch.utils.data.DataLoader->List[PIL.Image] or List[PIL.Image]
                texts:      torch.utils.data.DataLoader->List[str] or List[str]
                topk:       int
                topm:       int topm>topk
                return_index:bool
                            whether return the index of the image 
                            or the PIL.Image of the indexed Image
                return_text_emb:bool    default:False
                            whether return the embedding of the text
                verbose:    bool
            Returns
            -------
                torch.LongTensor[topk, n_text]
        """
        if verbose:
            print("Text Encoding...")
        texts_emb       = self.text_encode(texts, verbose)
        if verbose:
            print("Small Image Encoding...")
        images_emb      = self.small_image_encode(images,verbose)
        image_per_text  = images_emb @ texts_emb.T 
        topm_indices    = image_per_text.topk(topm, dim=0).indices
        if verbose:
            print("Filter Image...")
        subimages       = self.filter_images(topm_indices, images)
        if verbose:
            print("Large Image Encoding...")
        images_emb      = self.large_image_encode(subimages,verbose)
        image_per_text  = images_emb @ texts_emb.T 
        topk_indices    = image_per_text.topk(topk, dim=0).indices
        topk_indices    = topm_indices.unique()[topk_indices]
        if return_index:
            if return_text_emb:
                return topk_indices.T, texts_emb
            else:
                return topk_indices.T
        else:
            if return_text_emb:
                return self.indices_to_images(topk_indices, images), texts_emb
            else:
                return self.indices_to_images(topk_indices, images)


if __name__ == '__main__':
    from dataset import ImageLoader
    loader = ImageLoader.from_dir("./.coco/val2017",batch_size=4)
    # model  = CLIP()
    model = CascadeCLIP(
        2,"./.mobilenet/project_coco_train_ep1_ly2.pt"
    )
    model.cuda()
    model.eval()
    top2 = model.topk_images(loader, ["an image of banana"], topk=2,verbose=True)[0]
    print(top2)