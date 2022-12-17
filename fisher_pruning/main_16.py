import argparse
import logging
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    set_seed,
    CLIPProcessor,
    CLIPModel,
    CLIPTokenizer
)
from modeling_clip import CLIPModel as CLIPModel_pruned
from dataset.MSCOCO import MSCOCO
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.search import search_mac, search_latency, prune_all_but_one
from prune.rearrange import rearrange_mask
from prune.rescale import rescale_mask
from evaluate.eval import test_model
from utils.schedule import get_pruning_schedule


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='openai/clip-vit-base-patch32')
parser.add_argument("--task_name", type=str, default='mscoco', choices=[
    "mscoco",
])
parser.add_argument("--ckpt_dir", type=str, default='ckpt')
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--metric", type=str, choices=[
    "mac",
    "latency",
], default="mac")
parser.add_argument("--constraint", type=float, default=0.5,
    help="MAC/latency constraint relative to the original model",
)
parser.add_argument("--mha_lut", type=str, default=None)
parser.add_argument("--ffn_lut", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=128)
parser.add_argument("--seed", type=int, default=8)


def main():
    args = parser.parse_args()
    seq_len = 50
    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            args.metric,
            str(args.constraint),
            f"seed_{args.seed}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Set a GPU and the experiment seed
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned model and the corresponding tokenizer
    model = CLIPModel_pruned.from_pretrained(args.model_name)
    config = model.config.vision_config

    # Load the training dataset
    if args.task_name == 'mscoco':
        training_dataset = MSCOCO(2048, args.model_name)
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
        #collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cpu() #cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads)#cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size)#cuda()

    start = time.time()
    # Search the optimal mask
    head_grads, _ = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    end = time.time()
    logger.info(f"{args.task_name} Pruning time (s): {end - start}")

    for i in range(config.num_hidden_layers):

        head_mask = prune_all_but_one(
            config,
            head_grads,
            i,
        )

        neuron_mask = torch.ones((config.num_hidden_layers, config.intermediate_size))
        pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
        logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")


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

        # Evaluate the accuracy
        losses = test_model(model, head_mask, neuron_mask, test_dataloader)
        logger.info(f"Losses for head mask only: {losses[0]}")
        logger.info(f"Losses for neuron mask only: {losses[1]}")
        logger.info(f"Losses for both masks: {losses[2]}")
        logger.info(f"Losses for random binary masks with same number of zeros: {losses[3]}")
        # Save the masks
        torch.save(head_mask, os.path.join(args.output_dir, "head_mask_{}.pt".format(i)))


if __name__ == "__main__":
    main()
