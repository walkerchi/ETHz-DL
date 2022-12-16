import numpy as np
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel
from modeling_pruned_clip import CLIPModel as CLIPModel_p
from fisher_pruning.dataset.MSCOCO import MSCOCO
from tqdm import tqdm

def main():
    # load pruned and unpruned model
    seed = 7
    model_name = f'vitB32_pruned_{seed}.pt'
    pruned_model = torch.load(f'pruned_models/{model_name}')
    unpruned_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    dataset = MSCOCO(2048, 'openai/clip-vit-base-patch32', offset=2000)
    dataloader = DataLoader(dataset, batch_size=32)
    # measure times for both models
    t_pruned = 0
    t_unpruned = 0
    for batch in tqdm(dataloader):
        batch[0]['pixel_values'] = torch.squeeze(batch[0]['pixel_values'])
        t0 = time.perf_counter()
        output = pruned_model.get_image_features(**batch[0])
        t1 = time.perf_counter()
        output = unpruned_model.get_image_features(**batch[0])
        t2 = time.perf_counter()
        t_pruned += t1 - t0
        t_unpruned += t2 - t1

    print(f'Pruned Model Time: {t_pruned}\nUnpruned: {t_unpruned}')



if __name__ == '__main__':
    main()
