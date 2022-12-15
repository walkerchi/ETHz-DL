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

def prune(model):

    for idx, layer in enumerate(model.vision_model.encoder.layers):
        prune_neurons(layer.mlp, neuron_mask[idx])
        prune_heads(layer.self_attn, head_mask[idx])

    return model


def prune_neurons(mlp, mask):
    hidden_size = mlp.config.hidden_size
    mlp.fc1 = prune_lin_layer_after(mlp.fc1, mask, hidden_size)
    mlp.fc2 = prune_lin_layer_before(mlp.fc2, mask, hidden_size, rescale = True)


def prune_heads(mha, mask):
    num_nonzero = mask.count_nonzero()
    mha.num_heads = num_nonzero
    hd = mha.head_dim
    hidden_size = mha.embed_dim
    reduced_size = hd*num_nonzero
    expanded_mask = mask.repeat_interleave(hd)
    mha.k_proj = prune_lin_layer_after(mha.k_proj, expanded_mask, hidden_size)
    mha.v_proj = prune_lin_layer_after(mha.v_proj, expanded_mask, hidden_size)
    mha.q_proj = prune_lin_layer_after(mha.q_proj, expanded_mask, hidden_size)
    mha.out_proj = prune_lin_layer_before(mha.out_proj, expanded_mask, hidden_size, rescale=True)



def prune_lin_layer_after(fc, mask, hidden_size):
    nonzero_neurons = mask.nonzero().flatten()
    w = fc.weight
    fc_new = torch.nn.Linear(hidden_size, mask.count_nonzero())
    fc_new.bias.data = fc.bias[nonzero_neurons]
    fc_new.weight.data = w[nonzero_neurons]

    return fc_new

def prune_lin_layer_before(fc, mask, hidden_size, rescale=False):
    nonzero_neurons = mask.nonzero().flatten()
    w = fc.weight
    fc_new = torch.nn.Linear(mask.count_nonzero(), hidden_size)
    if rescale:
        w = mask*w
    w = w.t()
    fc_new.weight.data = w[nonzero_neurons].t()
    fc_new.bias.data = fc.bias
    return fc_new


if __name__ == '__main__':
    # Set parameters and load model
    model_name = 'openai/clip-vit-base-patch32'
    base_folder = 'fisher_pruning/outputs/openai/clip-vit-base-patch32/mscoco/mac/'
    restriction = '0.7'
    seed = 7
    head_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/head_mask.pt')
    neuron_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/neuron_mask.pt')
    pruned_model = CLIPModel.from_pretrained(model_name)
    unpruned_model = CLIPModel_pruned.from_pretrained(model_name)
    tokenizer = CLIPProcessor.from_pretrained(model_name)

    # prune the model
    pruned_model = prune(pruned_model)

    # verify wether it is the same


    breakpoint()
