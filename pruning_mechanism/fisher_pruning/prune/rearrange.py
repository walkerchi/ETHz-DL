import torch


@torch.no_grad()
def greedy_rearrange(mask, grads):
    """ Rearranges one layer with block diagonal approximation
    of the Fisher Information Matrix. """
    num_unpruned = int(mask.sum())
    num_pruned = mask.shape[0] - num_unpruned
    if num_unpruned == 0 or num_pruned == 0:
        return mask
    grads = grads.permute(1, 0).contiguous()
    # shape: [#heads/neurons, #mini_batches]
    grads_sq = grads.pow(2).sum(dim=1)
    _, indicies = grads_sq.sort(descending=False)
    indicies = indicies.tolist()

    # Greedy search
    masked_indicies = indicies[:num_pruned]
    for index in indicies[num_pruned:]:
        masked_indicies.append(index)
        grad_vectors = grads[masked_indicies]
        grad_sum = grad_vectors.sum(dim=0)

        complement = grad_sum - grad_vectors
        grad_sum_length = complement.pow(2).sum(dim=1)

        removed = grad_sum_length.argmin()
        del masked_indicies[removed]

    new_mask = torch.ones_like(mask)
    new_mask[masked_indicies] = 0
    return new_mask


def rearrange_mask(mask, grads):
    """Rearrange the mask that has been calculated with a
    diagonal Approximation of the Fisher Information Matrix to account for
    interactions between different mask variables.
    Here a block diagonal approximation is used to improve the mask.

    :param mask: Head or neuron mask calculated with
                 diagonal Approximation of the Fisher Information Matrix.
    :param grads: The gradients of given mask
    """
    # NOTE: temporarily convert to CPU tensors as the comp. intensity is low
    device = mask.device
    mask = mask.cpu()
    grads = grads.cpu()

    num_hidden_layers = mask.shape[0]
    for i in range(num_hidden_layers):
        mask[i] = greedy_rearrange(mask[i], grads[:, i, :])

    mask = mask.to(device)
    return mask
