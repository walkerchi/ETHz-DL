import time
import torch
from torch.utils.data import DataLoader
from transformers import (
    CLIPProcessor,
)
from fisher_pruning.modeling_clip import CLIPModel as CLIPModel_unpruned
from modeling_pruned_clip import CLIPModel as CLIPModel_p
from fisher_pruning.dataset.MSCOCO import MSCOCO
from fisher_pruning.utils.arch import apply_neuron_mask


def prune(model, head_mask, neuron_mask):

    for idx, layer in enumerate(model.vision_model.encoder.layers):
        prune_neurons(layer.mlp, neuron_mask[idx])
        prune_heads(layer.self_attn, head_mask[idx])

    return model


def prune_neurons(mlp, mask):
    hidden_size = mlp.config.hidden_size
    mlp.fc1 = prune_lin_layer_after(mlp.fc1, mask, hidden_size)
    mlp.fc2 = prune_lin_layer_before(mlp.fc2, mask, hidden_size, rescale=True)


def prune_heads(mha, mask):
    num_nonzero = mask.count_nonzero()
    mha.num_heads = num_nonzero
    hd = mha.head_dim
    hidden_size = mha.embed_dim
    expanded_mask = mask.repeat_interleave(hd)
    mha.k_proj = prune_lin_layer_after(mha.k_proj, expanded_mask, hidden_size)
    mha.v_proj = prune_lin_layer_after(mha.v_proj, expanded_mask, hidden_size)
    mha.q_proj = prune_lin_layer_after(mha.q_proj, expanded_mask, hidden_size)
    mha.out_proj = prune_lin_layer_before(mha.out_proj, expanded_mask,
                                          hidden_size, rescale=True)


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
    model_name = 'openai/clip-vit-base-patch16'
    base_folder = 'fisher_pruning/outputs/openai/clip-vit-base-patch16/mscoco/'
    restriction = '0.4'
    seed = 1234
    save_path = f'pruned_models/vitB16_pruned_{restriction}_{seed}.pt'
    head_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/head_mask.pt',
                           map_location=torch.device('cpu'))
    neuron_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/neuron_mask.pt',
                             map_location=torch.device('cpu'))
    pruned_model = CLIPModel_p.from_pretrained(model_name)
    tokenizer = CLIPProcessor.from_pretrained(model_name)

    # prune the model
    pruned_model = prune(pruned_model, head_mask, neuron_mask)
    torch.save(pruned_model, save_path)

    # verify wether the outputs are the same (besides rounding error)
    unpruned_model = CLIPModel_unpruned.from_pretrained(model_name)
    handles = apply_neuron_mask(unpruned_model.vision_model, neuron_mask)
    ds = MSCOCO(32, model_name)
    dl = DataLoader(ds, 32)
    for batch in dl:
        batch[0]['pixel_values'] = torch.squeeze(batch[0]['pixel_values'])
        t0 = time.perf_counter()
        outputs_pruned = pruned_model.get_image_features(**batch[0])
        t1 = time.perf_counter()
        outputs_unpruned = unpruned_model.get_image_features(**batch[0], head_mask=head_mask)
        t2 = time.perf_counter()
    for handle in handles:
        handle.remove()
    print(f"pruned time: {t1-t0:.4f} unpruned time: {t2-t1:.4f}")
    diff = (outputs_pruned - outputs_pruned).abs()
    if diff.count_nonzero() == 0:
        print("SUCCESS: The models are equivalent.")
        torch.save(pruned_model, save_path)
        print("Saved successfully.")
    else:
        print("Something has gone wrong.")
