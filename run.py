import numpy as np
import argparse
import logging
import toml
import time
import os
from PIL import Image
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
    logging.info("Full configuration: " + str(config.to_dict()))

    dataset     = config.dataset()

    if config.experiment == "topk":
        model       = config.models()
        logging.info(f"Building index...")
        build_start       = time()
        model.build(dataset.images, batch_size=config.batch_size, verbose=True)
        build_end         = time()
        logging.info(f"Building cost:{build_end-build_start:7.3f}s")

        topk_score  = np.array([0.0 for i in range(len(config.topk))])
        times       = []

        index       = np.arange(len(dataset.images))
        np.random.shuffle(index)
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

    elif config.experiment == "speedup":
        def time_model(model, name):
            img_times = []
            # Measure image encoding time
            for _ in range(config.n_reps):
                time = model.encode_images(dataset.images, batch_size=config.batch_size, verbose=True, return_timing=True)
                img_times.append(time / len(dataset.images))
            img_times = np.array(img_times)
            logging.info(f"{name} img encoding time mean, min, max: {img_times.mean():7.3f}, {img_times.min():7.3f}, {img_times.max():7.3f}\n\n")
            return img_times.mean()

        def time_models_config(models_config, name):
            total_time = 0
            for idx in range(len(models_config)):
                # If multiple models are instantiated at the same time, timing is skewed.
                # Therefore, only instatiate a single model at a time.
                model = models_config[idx]().models[0]
                img_time = time_model(model, f"{name}[{idx}]")
                fraction = 1 if idx == 0 else config.f
                total_time += img_time * fraction
            return total_time

        base_time = time_models_config(config.base_model, "base model")
        cascaded_time = time_models_config(config.models, "cascaded model")

        speedup = base_time / cascaded_time
        logging.info(f"speed up mean: {speedup:5.3f}")


if __name__ == "__main__":
    main()
