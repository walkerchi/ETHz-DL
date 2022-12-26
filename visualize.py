import os
import re
import numpy as np
import logging
import toml
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Union
from config import Config
from run import Task

MARKERS = [".","v","1","s","p","*","+","x","d"]

def load_speedup_log(config):
    """It will try to match all the file under the config.filename logging folder,
    and match configuration, if matched, it will return a success=True, (speedup mean, speedup std)
    else, it will return success=False, None
        Parameters
        ----------
            tag:        str
                        the unique tag for the logging
        Returns
        -------
            bool, Optional[Tuple[int, int]]
            success, (speedup mean, speedup std)
    """
    path = os.path.join(".log", config.filename)
    if not os.path.exists(path):
        return False, None
    if len(os.listdir(path)) == 0:
        return False, None
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            content = f.read()
            results = re.findall("^.*base\smodel",content,flags=re.DOTALL) # find the toml configuration in the log file, which will endswith `base model`
            if len(results) != 1:
                continue
            configuration = results[0].replace("base model", "")
            configuration = toml.loads(configuration)
            if configuration != toml.loads(toml.dumps(config.to_dict())): # toml.loads(toml.dumps()) to eliminate the `None`
                continue
            results = re.findall("speed up:\d+\.\d+\(\d+.\d+\)", content) # it looks like `speed up:1.22
            if len(results) != 1:
                continue
            results = re.findall("\d+\.\d+",results[0])
            mean, std = results
            mean, std = float(mean), float(std)
            return True, (mean, std)
    return False, None

def load_topk_log(config:Config):
    """It will try to match all the file under the config.filename logging folder,
    and match configuration, if matched, it will return a success=True, topk_scores
    else, it will return success=False, None
        Parameters
        ----------
            config:     Config
        Returns
        -------
            bool, Optional[List[float]]
            success, topk_scores
    """
    path = os.path.join(".log", config.filename)
    if not os.path.exists(path):
        return False, None
    if len(os.listdir(path)) == 0:
        return False, None
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            content = f.read()
            results = re.findall("^.*Building\sindex", content,flags=re.DOTALL) # find the toml configuration in the log file, which will endswith `Building index`
            if len(results) != 1:
                continue
            configuration = results[0].replace("Building index", "")
            configuration = toml.loads(configuration)
            if configuration != toml.loads(toml.dumps(config.to_dict())): # toml.loads(toml.dumps()) to eliminate the `None`
                continue

            topk_scores = []
            for k in config.topk:
                results = re.findall(f"top{k}:\d+\.\d+", content)  # it looks like `top3:0.02`
                if len(results) != 1:
                    break 
                score = float(re.findall("\d+\.\d+", results[0])[0])
                topk_scores.append(score)
            if len(topk_scores) == len(config.topk):
                return True, topk_scores
            else:
                print("topk score length not match")
    return False, None
                

def plot_flip():
    """Plot the FLIP model 
        left y axis is topk
        right y axis is seepup
        xaxis is p
    
        It will do speedup and topk experiment for every topk,
        and read the result from the log. 
        So remember to delete the log if 
    """
    
    name = "visualization_huggingface_flip" # the unique name for this function used for logging
    
    """
        Configuration Here,
        TODO: add argparse later
    """
    ps   = np.arange(0.1, 1.0, 0.1).round(1)  # you must do round, or it will cause something like 3.00000000000001, because of the IEEE754 float
    topk = [1, 3, 5]
    topm = [50]
    f    = 0.1
    dataset   = "MSCOCO"
    n_samples = 100
    layout    = "image-caption[0]"
    model_str = "openai/clip-vit-base-patch32"


    l_speedup    = []
    l_topk_score = []
    for i,p in enumerate(ps):
        """
            Speedup Experiment
            try:
                load from log file
            except:
                run the speedup experiment
        """
        print(f"\n\np:{p} ({i+1}/{len(ps)})")
        config = Config({
                "dataset":  {
                    "name":dataset,
                    "kwargs":{
                        "n_samples":n_samples,
                        "layout":layout
                    }
                },
                "models"    :{
                    "name":["HuggingFaceFLIP","HuggingFaceCLIP"],
                    "0":{"kwargs":{
                            "p":p,
                            "model_str":model_str
                        }},
                    "1":{"kwargs":{
                            "model_str":model_str
                        }}
                },
                "topm"      : topm,
                "filename"  : f"{name}_{p}_speedup",
                "experiment":"speedup",
                "n_reps"    :3,
                "f"         :f,
                "device"    :"cpu",
                "base_model":{
                    "name":["HuggingFaceCLIP"],
                    "0":{"kwargs":{
                        "model_str":model_str
                    }}
                }
        }, init_logging=False)
        success, speedup = load_speedup_log(config)
        print("\nCalculate Speed Up")
        if not success:
            config.init_logging()
            speedup = Task.speedup(config)
            speedup = (speedup.mean(), speedup.std())
        else:
            print("Load Directly")
        l_speedup.append(speedup)
        
        """
            Topk Experiment
            try:
                load from log file
            except:
                run the topk experiment
        """
        print("\nCalculate Topk...")
        config = Config({
                "dataset":  {
                    "name":dataset,
                    "kwargs":{
                        "n_samples":n_samples,
                        "layout":layout
                    }
                },
                "models"    :{
                    "name":["HuggingFaceFLIP","HuggingFaceCLIP"],
                    "0":{"kwargs":{
                            "p":p,
                            "model_str":model_str
                        }},
                    "1":{"kwargs":{
                            "model_str":model_str
                        }}
                },
                "topm"      : [50],
                "topk"      : topk,
                "filename"  : f"{name}_{p}_topk",
                "experiment":"topk",
                "device"    :"cpu"
            },init_logging=False)
        success, topk_score = load_topk_log(config)
        if not success:
            config.init_logging()
            topk_score = Task.topk(config)
        else:
            print("Load Directly")
        l_topk_score.append(topk_score)
    speedup = np.array(l_speedup) # [len(fs), 2]
    topk_score    = np.stack(l_topk_score)  # [len(fs), len(topk)]

    """
        Calculate the baseline for topk
    """
    config = Config({
                    "dataset":  {
                        "name":dataset,
                        "kwargs":{
                            "n_samples":n_samples,
                            "layout":layout
                        }
                    },
                    "models"    :{
                        "name":["HuggingFaceCLIP"],
                        "0":{"kwargs":{
                                "model_str":model_str
                            }}
                    },
                    "topk"      : topk,
                    "filename"  : f"{name}_base_topk",
                    "experiment":"topk",
                    "device"    :"cpu"
                },init_logging=False)
    success, base_topk_score = load_topk_log(config)
    if not success:
        config.init_logging()
        base_topk_score = Task.topk(config)

    """
        Do the plotting
    """
    fig, ax = plt.subplots(figsize=(12,8))  # change the figure size and dpi here
    cmap    = plt.cm.Accent                 # change the color map here
    for i, k in enumerate(topk):            # plot topk
        ax.plot(ps, topk_score[:, i], color=cmap(i), label=f"top{k}")
        ax.scatter(ps, topk_score[:, i], c=[cmap(i) for _ in ps], marker=MARKERS[i])
        ax.axhline(y=base_topk_score[i], color=cmap(i), label=f"baseline(top{k})", linestyle=":")
    ax_ = ax.twinx()
    ax_.plot(ps, speedup[:, 0], color=cmap(len(topk)+1), label="speedup")  # plot speedup
    ax_.fill_between(ps, speedup[:,0]-speedup[:,1], speedup[:,0]+speedup[:,1], alpha=0.3, color=plt.cm.Accent(len(topk)+1)) # this is the uncertainty for speedup
    ax.set_xlabel("p")
    ax.set_ylabel("topk")
    ax_.set_ylabel("speedup(1x)")
    ax_.axhline(y=1.0, color="purple", label="baseline(speedup)", linestyle="-.")
    fig.legend(loc="upper left")
    ax.spines['top'].set_visible(False)
    ax_.spines['top'].set_visible(False)
    ax.set_title("FLIP speed up and topk to p")
    path = "./visualization"
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(os.path.join(path, f"{name}.pdf")) # pdf for the overleaf
    fig.savefig(os.path.join(path, f"{name}.png")) # png for the preview
    plt.show()

if __name__ == "__main__":
    plot_flip()
