import numpy as np
import argparse
import logging
import toml
import time
import os
from time import time
from tqdm import tqdm
from config import Config

def main():
    parser = argparse.ArgumentParser(description=
"""
Evaluate an experiment
""")
    parser.add_argument( 
        "--id", 
        type    =str,
        default =None,
        help    =f"configure file id of the experiment, Example: `1`->`0001.toml`",
    )
    parser.add_argument(
        "--name",
        type    =str,
        default =None,
        help    =f"configure filename of the experiment, Example: `test`->`test.toml`")
    args = parser.parse_args()
    
    if args.name is None:
        assert args.id is not None, "either `--id` or `--name` should be assigned"
        filename = args.id.zfill(4) 
    else:
        assert args.name is not None, "either `--id` or `--name` should be assigned"
        filename = args.name
    file_path = os.path.join("experiments",f"{filename}.toml")
    assert os.path.exists(file_path), f"The file path you assign is not exists:{file_path}"
    config = toml.load(file_path)
    config['filename'] = filename
    config = Config(config)
    
    
    dataset     = config.dataset()
    model       = config.models()

    logging.info(f"Building index...")
    start       = time()
    model.build(dataset.images, batch_size=config.batch_size, verbose=True)
    end         = time()
    logging.info(f"Building cost {end-start:7.3f}s")

    topk_score  = np.array([0.0 for i in range(len(config.topk))])
    times       = []

    for i, text in tqdm(enumerate(dataset.captions),total=len(dataset.captions), desc="Query Caption"):

        start   = time()
        indices = model.query(text, topk=max(config.topk), topm=config.topm, batch_size=config.batch_size)
        end     = time()
        times.append(end-start)

        for j,k in enumerate(config.topk):
            if i in indices[:k]:
                topk_score[j] += 1.0

    topk_score /= len(dataset)

    logging.info("-------topk score-------")
    for k, score in zip(config.topk, topk_score):
        logging.info(f"top{k}:{score:7.5f}")
    logging.info("-----time each text-----")
    for i, t in enumerate(times):
        logging.info(f"[{i}]:{t:5.3f}s")




if __name__ == "__main__":
    main()
