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
from dataset.MSCOCO import MSCOCO, test_speed
from evaluate.eval import test_model


if __name__ == "__main__":
    # load model
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel_pruned.from_pretrained(model_name)
    tokenizer = CLIPProcessor.from_pretrained(model_name)
    config = model.config.vision_config
    # load masks
    base_folder = 'outputs/openai/clip-vit-base-patch32/mscoco/mac/'
    restriction = '0.5'
    seed = 3
    head_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/head_mask.pt')
    neuron_mask = torch.load(f'{base_folder}{restriction}/seed_{seed}/neuron_mask.pt')
    head_mask = head_mask.cuda()
    model.cuda()
    test_speed(model, 256, 16, None)#head_mask)
    test_speed(model, 256, 16, None)
    test_speed(model, 256, 16, head_mask)
    test_speed(model, 256, 16, None)
    breakpoint()
    # load dataset
    dataset = MSCOCO(1000, model_name, offset=3000)
    test_dataset = Subset(
        dataset,
        np.random.choice(len(dataset), 128).tolist(),
    )
    test_batch_size = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    losses = test_model(model, head_mask, neuron_mask, test_dataloader)
    print('Head mask only losses:', l*[round(l.item(),2) for l in losses[0]])
    print('Neuron mask only losses:', *[round(l.item(),2) for l in losses[1]])
    print('Both masks losses:', *[round(l.item(),2) for l in losses[2]])
    print('Random mask losses:', *[round(l.item(),2) for l in losses[3]])
    print('Average loss for binary head mask:', sum(losses[0])/len(losses[0]), 'Average loss for random binary head mask:', sum(losses[3])/len(losses[3]))
    breakpoint()


