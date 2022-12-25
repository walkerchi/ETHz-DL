import os
import re
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Union
from config import Config
from run import Task

MARKERS = [".","v","1","s","p","*","+","x","d"]

def load_speedup_log(tag:str):
    """
        Parameters
        ----------
            tag:        str
                        the unique tag for the logging
        Returns
        -------
            bool, Optional[Tuple[int, int]]
            success, (speedup mean, speedup std)
    """
    path = os.path.join(".log", tag)
    if not os.path.exists(path):
        return False, None
    if len(os.listdir(path)) == 0:
        return False, None
    path = os.path.join(path,os.listdir(path)[-1])
    content = open(path, "r").read()
    results = re.findall("speed up:\d+\.\d+\(\d+.\d+\)", content)
    if len(results) == 1:
        result  = results[0]
        results = re.findall("\d+\.\d+", result)
        mean, std = results
        mean, std = float(mean), float(std)
        return True, (mean, std)
    else:
        return False, None

def load_topk_log(tag:str, topk:List[int]):
    """
        Parameters
        ----------
            tag:        str 
                        the unique tag for the logging
            top:        int
                        the topk should be specified
        Returns
        -------
            bool, Optional[List[float]]
            success, topk_scores
    """
    path = os.path.join(".log", tag)
    if not os.path.exists(path):
        return False, None
    if len(os.listdir(path)) == 0:
        return False, None
    path = os.path.join(path,os.listdir(path)[-1])
    content = open(path, "r").read()
    topk_scores = []
    for k in topk:
        results = re.findall(f"top{k}:\d+\.\d+",content)
        if len(results) == 1:
            result = results[0]
            score  = float(re.findall("\d+\.\d+", result)[0])
            topk_scores.append(score)
        else:
            return False, None
    return True, topk_scores

def plot_flip():
    """Plot the FLIP model 
        left y axis is topk
        right y axis is seepup
        xaxis is p
    """
    
    name = "visualization_huggingface_flip"
    
    ps   = np.arange(0.1, 1.0, 0.1).round(1)
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
        print(f"\n\np:{p} ({i+1}/{len(ps)})")
        success, speedup = load_speedup_log(f"{name}_{p}_speedup")
        print("\nCalculate Speed Up")
        if not success:
            speedup = Task.speedup(Config({
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

            }))
            speedup = (speedup.mean(), speedup.std())
        else:
            print("Load Directly")
        l_speedup.append(speedup)
        print("\nCalculate Topk...")
        success, topk_score = load_topk_log(f"{name}_{p}_topk", topk)
        if not success:
            topk_score = Task.topk(Config({
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
            }))
        else:
            print("Load Directly")
        l_topk_score.append(topk_score)
    speedup = np.array(l_speedup) # [len(fs), 2]
    topk_score    = np.stack(l_topk_score)  # [len(fs), len(topk)]

    success, base_topk_score = load_topk_log(f"{name}_base_topk", topk)
    if not success:
        base_topk_score = Task.topk(Config({
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
                }))

    fig, ax = plt.subplots(figsize=(12,8))
    cmap    = plt.cm.Accent
    for i, k in enumerate(topk):
        ax.plot(ps, topk_score[:, i], color=cmap(i), label=f"top{k}")
        ax.scatter(ps, topk_score[:, i], c=[cmap(i) for _ in ps], marker=MARKERS[i])
        ax.axhline(y=base_topk_score[i], color=cmap(i), label=f"baseline(top{k})", linestyle=":")
    ax_ = ax.twinx()
    ax_.plot(ps, speedup[:, 0], color=cmap(len(topk)+1), label="speedup")
    ax_.fill_between(ps, speedup[:,0]-speedup[:,1], speedup[:,0]+speedup[:,1], alpha=0.3, color=plt.cm.Accent(len(topk)+1))
    ax.set_xlabel("p")
    ax.set_ylabel("topk")
    ax_.set_ylabel("speedup(1x)")
    ax_.axhline(y=1.0, color="purple", label="baseline(speedup)", linestyle="-.")
    fig.legend(loc="upper left")
    ax.spines['top'].set_visible(False)
    ax_.spines['top'].set_visible(False)
    # ax.grid(linestyle="--", color="")
    # ax_.grid(linestyle=":")
    ax.set_title("FLIP speed up and topk to p")
    path = "./visualization"
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(os.path.join(path, f"{name}.pdf"))
    fig.savefig(os.path.join(path, f"{name}.png"))
    plt.show()

if __name__ == "__main__":
    plot_flip()
