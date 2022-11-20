import torch
import glob 
from typing import List
from PIL import Image,ImageFile

PILImage = ImageFile.ImageFile

class ImageFolder:
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

class ImageFolderLoader:
    @classmethod
    def from_dir(cls, path:str, batch_size:int=4):
        dataset = ImageFolder.from_dir(path)
        return cls(dataset, batch_size)
    def __init__(self, dataset, batch_size:int=4):
        self.dataset    = dataset
        self.batch_size = batch_size 
        self.n_batch = int((len(self.dataset) + self.batch_size-1) / self.batch_size)
    def subloader(self, index):
        paths = [self.dataset.paths[i] for i in index]
        dataset = ImageFolder(paths)
        return ImageFolderLoader(
            dataset,
            batch_size=self.batch_size
        )
    def __iter__(self):
        self.counter = 0
        return self
    
    def __len__(self):
        return self.n_batch

    def __next__(self):
        if self.counter >= self.n_batch:
            raise StopIteration()
        else:
            batch = []
            for i in range(self.counter*self.batch_size, (self.counter+1)*self.batch_size):
                if i < len(self.dataset):
                    batch.append(self.dataset[i])
            self.counter += 1
            return batch
        


if __name__ == '__main__':
    loader = ImageFolderLoader.from_dir("./.coco/val2017")
    batch = next(iter(loader))
    print(batch)
    