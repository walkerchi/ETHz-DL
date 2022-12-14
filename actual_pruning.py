import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    set_seed,
    CLIPProcessor,
    CLIPModel,
    CLIPTokenizer
)
from fisher_pruning.modeling_clip import CLIPModel as CLIPModel_pruned
from fisher_pruning.dataset.MSCOCO import MSCOCO

def main():
    # Set parameters and load model
    model_name = 'openai/clip-vit-base-patch32'
    base_folder = 'fisher_pruning/outputs/openai/clip-vit-base-patch32/mscoco/mac/'
    restriction = '0.5'
    seed = 7
    head_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/head_mask.pt')
    neuron_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/neuron_mask.pt')
    model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPProcessor.from_pretrained(model_name)

    for idx, layer in enumerate(model.vision_model.encoder.layers):
        prune_neurons(layer.mlp, neuron_mask[idx])
        prune_heads(layer.self_attn, head_mask[idx])


def prune_neurons(mlp, mask):
    hidden_size = mlp.config.hidden_size
    nonzero_neurons = mask.nonzero().flatten()
    num_nonzero_neurons = nonzero_neurons.shape[0]

    w1 = mlp.fc1.weight
    b1 = mlp.fc1.bias
    w2 = mlp.fc2.weight

    fc1_new = torch.nn.Linear(hidden_size, num_nonzero_neurons)
    fc2_new = torch.nn.Linear(num_nonzero_neurons, hidden_size)

    fc1_new.bias.data = b1[nonzero_neurons]
    fc1_new.weight.data = w1[nonzero_neurons]

    w2 = (mask * w2).t()

    fc2_new.weight.data = w2[nonzero_neurons].t()

    mlp.fc1 = fc1_new
    mlp.fc2 = fc2_new


def prune_heads(mha, mask):
    pass

if __name__ == '__main__':
    main()
    breakpoint()
