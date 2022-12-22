import torch
import numpy as np
from dataset.MSCOCO import MSCOCO
from utils.arch import apply_neuron_mask
from tqdm import tqdm


def random_mask_like(mask):
    ind = torch.rand(mask.numel()).topk(mask.count_nonzero()).indices
    ind = np.array(np.unravel_index(ind.numpy(), mask.shape)).T
    rand_mask = torch.zeros(mask.shape, dtype=torch.bool)
    for i in ind: rand_mask[tuple(i)] = True

    return rand_mask

def get_losses(model, head_mask, neuron_mask, dataloader, device):
    # return unpruned moldel image encoding as target
    dataloader.dataset.dataset.teacher = True # twice .dataset, because dataset is a subset, subset.dataset gives the mscoco
    losses = []
    if neuron_mask is not None:
        handles = apply_neuron_mask(model.vision_model, neuron_mask)
    for batch in tqdm(dataloader):
        batch[0]['pixel_values'] = torch.squeeze(batch[0]['pixel_values']).to(device)
        outputs = model.get_image_features(**batch[0], head_mask=head_mask)

        l = model.get_loss(outputs, batch[1].squeeze().to(device))
        losses.append(l)

    if neuron_mask is not None:
        for handle in handles:
            handle.remove()
    dataloader.dataset.dataset.teacher = False

    return losses


@torch.no_grad()
def test_model(model, head_mask, neuron_mask, dataloader, device):
    # create random masks
    head_rand_mask = random_mask_like(head_mask).to(device)
    neuron_rand_mask = random_mask_like(neuron_mask).to(device)
    losses = [[], [], [], []] # in order head loss, neuron loss, both loss, rand loss
    h_masks = [head_mask, None, head_mask, head_rand_mask]
    n_masks = [None, neuron_mask, neuron_mask, neuron_rand_mask]
    for i in tqdm(range(len(losses))):
        losses[i] = get_losses(model, h_masks[i], n_masks[i], dataloader, device)
    return losses


