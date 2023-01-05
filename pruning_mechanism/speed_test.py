"""
This Script runs both the pruned and unpruned models alternately on a dataset
to measure the time they take without being susceptible to external influences.
"""
import time
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel
from modeling_pruned_clip import CLIPModel as CLIPModel_p
from fisher_pruning.dataset.MSCOCO import MSCOCO
from tqdm import tqdm


def main():
    # load pruned and unpruned model
    device = 'cpu'
    seed = 2
    restriction = 0.75
    model_name = f'vitB16_pruned_{restriction}_{seed}.pt'
    pruned_model = torch.load(f'pruned_models/{model_name}',
                              map_location=torch.device(device))
    unpruned_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    dataset = MSCOCO(1024, 'openai/clip-vit-base-patch32', offset=2000)
    dataloader = DataLoader(dataset, batch_size=32)
    # measure times for both models
    t_pruned = 0
    t_unpruned = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch[0]['pixel_values'] = torch.squeeze(batch[0]['pixel_values']).to(device)
            t0 = time.perf_counter()
            _ = pruned_model.get_image_features(**batch[0])
            t1 = time.perf_counter()
            _ = unpruned_model.get_image_features(**batch[0])
            t2 = time.perf_counter()
            t_pruned += t1 - t0
            t_unpruned += t2 - t1

    print(f'Pruned Model Time: {t_pruned}\nUnpruned: {t_unpruned}')
    print(f'{100*t_pruned/t_unpruned:.2f}%')


if __name__ == '__main__':
    main()
