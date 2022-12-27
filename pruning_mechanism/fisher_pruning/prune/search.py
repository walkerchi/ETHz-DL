import torch

from prune.fisher import compute_fisher_info
from efficiency.mac import compute_mac, mac_per_head, mac_per_neuron


@torch.no_grad()
def search_mac(
    config,
    head_grads,
    neuron_grads,
    seq_len,
    mac_constraint,
    device
):
    assert mac_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    max_mac = mac_constraint * original_mac
    head_importance = compute_fisher_info(head_grads)
    neuron_importance = compute_fisher_info(neuron_grads)
    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)

    max_importance = 0
    for num_heads in range(1, num_hidden_layers * num_attention_heads + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_mask = torch.zeros(num_hidden_layers * num_attention_heads).to(device)
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).to(device)
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    return head_mask, neuron_mask


