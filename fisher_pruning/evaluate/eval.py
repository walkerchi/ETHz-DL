import torch
import numpy as np
from dataset.MSCOCO import MSCOCO
from utils.arch import apply_neuron_mask


def random_mask_like(mask):
    ind = torch.rand(mask.numel()).topk(mask.count_nonzero()).indices
    ind = np.array(np.unravel_index(ind.numpy(), mask.shape)).T
    rand_mask = torch.zeros(mask.shape, dtype=torch.bool)
    for i in ind: rand_mask[tuple(i)] = True

    return rand_mask

@torch.no_grad()
def test_model(model, head_mask, neuron_mask, dataloader):
    # create random masks
    head_rand_mask = random_mask_like(head_mask)
    neuron_rand_mask = random_mask_like(neuron_mask)
    # return unpruned moldel image encoding as target
    dataloader.dataset.dataset.teacher = True # twice .dataset, because dataset is a subset, subset.dataset gives the mscoco
    losses = [[], [], [], []] # in order head loss, neuron loss, both loss, rand loss
    h_masks = [head_mask, None, head_mask, None]#head_rand_mask]
    n_masks = [None, neuron_mask, neuron_mask, neuron_rand_mask]
    # only head mask
    for i in range(len(losses)):
        if n_masks[i] is not None:
            handles = apply_neuron_mask(model.vision_model, n_masks[i])
        for batch in dataloader:
            batch[0]['pixel_values'] = torch.squeeze(batch[0]['pixel_values'])
            outputs = model.get_image_features(**batch[0], head_mask=h_masks[i])

            l = model.get_loss(outputs, batch[1].squeeze())
            losses[i].append(l)

        if n_masks[i] is not None:
            for handle in handles:
                handle.remove()
    dataloader.dataset.dataset.teacher = False
    return losses


