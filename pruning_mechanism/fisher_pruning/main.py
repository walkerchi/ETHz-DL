import argparse
import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    set_seed
)
from modeling_clip import CLIPModel as CLIPModel_pruned
from dataset.MSCOCO import MSCOCO
from efficiency.mac import compute_mask_mac
from prune.fisher import collect_mask_grads
from prune.search import search_mac
from prune.rearrange import rearrange_mask
from prune.rescale import rescale_mask
from evaluate.eval import test_model


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default='openai/clip-vit-base-patch16')
parser.add_argument("--task_name", type=str, default='mscoco', choices=[
    "mscoco",
])
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=1)

parser.add_argument("--constraint", type=float, default=0.65,
                    help="MAC constraint relative to the original model")
parser.add_argument("--num_samples", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)


def main():
    args = parser.parse_args()
    seq_len = 197
    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            str(args.constraint),
            f"seed_{args.seed}",
        )
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')
    if args.gpu != 0:
        if torch.cuda.is_available():
            device = torch.device('cuda')

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    # Set the experiment seed
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned model and the corresponding tokenizer
    model = CLIPModel_pruned.from_pretrained(args.model_name)
    config = model.config.vision_config

    # Load the training dataset
    n_load = args.num_samples if args.num_samples > 128 else 128
    if args.task_name == 'mscoco':
        training_dataset = MSCOCO(n_load, args.model_name)
    else:
        raise NotImplementedError

    # Sample the examples to be used for search
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist())

    sample_batch_size = 32
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # initialize the masks
    full_head_mask = torch.ones(config.num_hidden_layers,
                                config.num_attention_heads).to(device)
    full_neuron_mask = torch.ones(config.num_hidden_layers,
                                  config.intermediate_size).to(device)

    start = time.time()
    # Get the gradients for the masks, while the model is frozen
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
        device
    )
    # Find the head and neuron masks with the highest fisher value
    # that satisfy the FLOP constraint
    head_mask, neuron_mask = search_mac(
        config,
        head_grads,
        neuron_grads,
        seq_len,
        args.constraint,
        device
    )
    # Log the calculated FLOP reduction
    pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask,
                                            seq_len, config.hidden_size)
    logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")

    # Rearrange the mask
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)

    # Rescale the mask by solving a least squares problem
    head_mask, neuron_mask = rescale_mask(
        model,
        config,
        full_head_mask,
        full_neuron_mask,
        head_mask,
        neuron_mask,
        sample_dataloader,
        device,
        classification_task=False,
    )

    # Print the pruning time
    end = time.time()
    logger.info(f"{args.task_name} Pruning time (s): {end - start}")

    # Save the masks
    torch.save(head_mask, os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.output_dir, "neuron_mask.pt"))

    # Evaluate the results with a small test
    test_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), 64).tolist(),
    )
    test_batch_size = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    losses = test_model(model, head_mask, neuron_mask, test_dataloader, device)
    logger.info(f"Losses for head mask only: {losses[0]}")
    logger.info(f"Losses for neuron mask only: {losses[1]}")
    logger.info(f"Losses for both masks: {losses[2]}")
    logger.info("Losses for random binary masks with same number of zeros: "
                f" {losses[3]}")


if __name__ == "__main__":
    main()
