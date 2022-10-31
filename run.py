
import argparse
import toml
from evaluate import evaluate

from utils import EXPERIMENTS_PATH, configure_logger, fill_config_defaults, prep_experiment_dir, set_seed

def main() -> None:
    parser = argparse.ArgumentParser(description=
"""
Evaluate an experiment
""")
    parser.add_argument( "experiment", type=str,
        help=f"ID of the experiment that you want to evaluate.",
    )
    args = parser.parse_args()
    config = toml.load(EXPERIMENTS_PATH / f"{args.experiment}.toml")
    config = fill_config_defaults(config)
    set_seed(config["seed"])
    experiment_dir = prep_experiment_dir(args.experiment)
    config["experiment_dir"] = experiment_dir
    configure_logger(experiment_dir, config["logging_level"])

    if config["type"] == "evaluate":
        evaluate(config)
    elif config["type"] == "shrink":
        # TODO
        raise NotImplementedError()
    else:
        raise Exception("Unknown experiment type '{}'".format(config["type"]))

if __name__ == "__main__":
    main()
