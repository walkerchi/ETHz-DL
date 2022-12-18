import numpy as np
import argparse
import logging
import toml
import time
import os
from time import time
from tqdm import tqdm
from config import Config, ModelsConfig

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

    if config.experiment == "topk":
        logging.info(f"Building index...")
        build_start       = time()
        model.build(dataset.images, batch_size=config.batch_size, verbose=True)
        build_end         = time()
        logging.info(f"Building cost:{build_end-build_start:7.3f}s")

        topk_score  = np.array([0.0 for i in range(len(config.topk))])
        times       = []

        index       = np.arange(len(dataset.images))
        np.random.shuffle(index)
        if config.query_rate is not None:
            index   = index[:int(config.query_rate * len(index))]
        labels      = index
        inputs      = [dataset.captions[i]  for i in index]

        logging.info(f"Querying...")
        for label, text in tqdm(zip(labels,inputs),total=len(index), desc="Query Caption"):

            start   = time()
            indices = model.query(text, topk=max(config.topk), topm=config.topm, batch_size=config.batch_size)
            end     = time()
            times.append(end-start)

            for j,k in enumerate(config.topk):
                if label in indices[:k]:
                    topk_score[j] += 1.0
        times = np.array(times)
        query_end = time()
        topk_score /= len(index)
        logging.info(f"Query cost:{sum(times):7.3f}s, time each query:{np.mean(times):5.3f}({np.std(times):5.3f})s, max a query:{np.max(times):5.3f}s, min a query:{np.min(times):5.3f}s\n\n")
        logging.info(f"Build and Query time:{query_end - build_start:7.3f}s\n\n")

        model.log_cache()

        logging.info("\n-------topk score-------")
        for k, score in zip(config.topk, topk_score):
            logging.info(f"top{k}:{score:7.5f}")
        logging.info("\n-----time each text-----")
        for i, t in enumerate(times):
            logging.info(f"[{i}]:{t:5.3f}s")

    elif config.experiment == "speedup":
        base_model = config.models[-1]()
        
        logging.info("Build Base Model...")
        base_build_start = time()
        base_model.build(dataset.images, batch_size=config.batch_size, verbose=True)
        base_build_end   = time()
        large_time = (base_build_end - base_build_start) / len(dataset.images)
        logging.info(f"Base Model Building cost:{base_build_end - base_build_start:7.3f}s average:{large_time:7.3f}s\n\n")
        del base_model # save memory

        logging.info("Build Model...")
        build_start = time()
        model.build(dataset.images, batch_size=config.batch_size, verbose=True)
        build_end   = time()
        small_time = (build_end - build_start) / len(dataset.images)
        logging.info(f"Model Building cost:{build_end - build_start:7.3f}s average:{small_time:7.3f}s\n\n")
        del model

        speedup = large_time / (small_time + large_time * config.query_rate)        
        logging.info(f"speed up:{speedup:5.3f}")




if __name__ == "__main__":
    main()
