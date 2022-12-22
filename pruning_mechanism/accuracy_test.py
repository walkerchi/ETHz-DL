import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    set_seed,
    CLIPProcessor,
    CLIPModel,
    CLIPTokenizer
)
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, 'fisher_pruning')
sys.path.append(module_dir)
from fisher_pruning.modeling_clip import CLIPModel as CLIPModel_pruned
from fisher_pruning.dataset.MSCOCO import MSCOCO
from fisher_pruning.evaluate.eval import test_model, get_losses


if __name__ == "__main__":
    # load model
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel_pruned.from_pretrained(model_name)
    tokenizer = CLIPProcessor.from_pretrained(model_name)
    config = model.config.vision_config
    # load masks
    base_folder = 'fisher_pruning/outputs/openai/clip-vit-base-patch32/mscoco/'
    restriction = '0.6'
    seed = 603
    head_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/head_mask.pt', map_location=torch.device('cpu'))
    neuron_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/neuron_mask.pt', map_location=torch.device('cpu'))
    breakpoint()
    model.cpu()

    # load dataset
    test_size = 256
    dataset = MSCOCO(1024, model_name, offset=4100)
    test_dataset = Subset(
        dataset,
        np.random.choice(len(dataset), test_size).tolist(),
    )
    test_batch_size = 32
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    losses = test_model(model, head_mask, neuron_mask, test_dataloader, torch.device('cpu'))
    print('Head mask only losses:', *['{:.3e}'.format(l.item()) for l in losses[0]])
    print('Neuron mask only losses:', *['{:.3e}'.format(l.item()) for l in losses[1]])
    print('Both masks losses:', *['{:.3e}'.format(l.item()) for l in losses[2]])
    print('Both binary mask losses:', *['{:.3e}'.format(l.item()) for l in losses[3]])
    print('Average loss for both mask:', sum(losses[2])/len(losses[2]))
    head_mask = head_mask != 0
    neuron_mask = neuron_mask != 0
    losses_b = test_model(model, head_mask != 0, neuron_mask != 0, test_dataloader, torch.device('cpu'))
    print('Now for binary masks: ')
    print('Head mask only losses:', *['{:.3e}'.format(l.item()) for l in losses_b[0]])
    print('Neuron mask only losses:', *['{:.3e}'.format(l.item()) for l in losses_b[1]])
    print('Both masks losses:', *['{:.3e}'.format(l.item()) for l in losses_b[2]])
    print('Both binary mask losses:', *['{:.3e}'.format(l.item()) for l in losses_b[3]])
    print('Average loss for both mask:', sum(losses_b[2])/len(losses_b[2]))
    breakpoint()


