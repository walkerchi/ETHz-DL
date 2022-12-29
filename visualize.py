import os
import re
import numpy as np
import logging
import toml
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Optional, Union
from config import Config
from run import Task

MARKERS = [".","v","1","s","p","*","+","x","d"]

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'font.serif': ['Palatino'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.rcParams['font.size'] = '9'
plt.rcParams['figure.figsize'] = (3.5, 2.5)

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
    ps   = np.arange(0, 1.1, 0.1).round(1)  # you must do round, or it will cause something like 3.00000000000001, because of the IEEE754 float
    topk = [1, 5, 10]
    topm = [50]
    f    = 0.1
    dataset   = "MSCOCO"
    layout    = "image-caption[0]"
    model_str = "openai/clip-vit-base-patch16"


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
                "batch_size": 32,
                "dataset":  {
                    "name":dataset,
                    "kwargs":{
                        "n_samples":128,
                        "layout":layout
                    }
                },
                "models"    :{
                    "name":["HuggingFaceFLIP","HuggingFaceFLIP"],
                    "0":{"kwargs":{
                            "p":p,
                            "model_str":model_str
                        }},
                    "1":{"kwargs":{
                            "p":1-p,
                            "model_str":model_str
                        }}
                },
                "topm"      : topm,
                "filename"  : f"{name}_{p}_speedup",
                "experiment":"speedup",
                "n_reps"    :3,
                "do_warmup_rep"    : True,
                "f"         :f,
                "device"    :"cpu",
                "base_model":{
                    "name":["HuggingFaceCLIP"],
                    "0":{"kwargs":{
                        "model_str":model_str
                    }},
                    "topm": []
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

    for i,p in enumerate(ps):
        """
            Topk Experiment
            try:
                load from log file
            except:
                run the topk experiment
        """
        print("\nCalculate Topk...")
        config = Config({
                "batch_size": 32,
                "dataset":  {
                    "name":dataset,
                    "kwargs":{
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
                    "batch_size": 32,
                    "dataset":  {
                        "name":dataset,
                        "kwargs":{
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
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({"Masked input in \%": ps * 100,
                    "Top-1": topk_score[:, 0] * 100, 
                    "Top-5": topk_score[:, 1] * 100,
                    "Top-10": topk_score[:, 2] * 100})
    df = df.melt('Masked input in \%', var_name=' ', value_name='Accuracy in \%')
    sns.set_style("darkgrid")
    sns.lineplot(data=df, x="Masked input in \%", y="Accuracy in \%", hue=' ', marker='o', markersize=2, legend=False)
    sns.despine()
    
    ax = plt.gca()
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel("Accuracy in \%", loc="top", rotation="horizontal")
    ax.yaxis.set_label_coords(0.30,1.02)
    yts = [0,10,20,30,40,50,60,70]
    ax.set_yticks(yts)
    ax.set_yticklabels([str(yt) for yt in yts])
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-1)
    ax_ = plt.twinx()
    ax_.plot(ps* 100, speedup[:, 0], color='brown', label="Speedup", marker='o', markersize=2, markeredgecolor=(1, 1, 1, 1))  # plot speedup
    ax_.fill_between(ps*100, speedup[:,0]-speedup[:,1], speedup[:,0]+speedup[:,1], alpha=0.3, color='brown') # this is the uncertainty for speedup
    ax_.set_ylabel("Speedup", loc="top", rotation="horizontal")
    ax_.yaxis.set_label_coords(1,1.10)
    ax_.set_yscale('log', base=2)
    ax_.set_ylim(bottom=1, top=2**3.5)
    ax_.set_yticks([1,2,4,8,])
    ax_.set_yticklabels(['1','2','4','8'])
    ax_.grid(None)
    ax_.spines['bottom'].set_visible(True)
    ax_.spines['bottom'].set_linewidth(1.5)
    ax_.spines['bottom'].set_color('black')
    ax.tick_params(bottom="on")
    ax.tick_params(axis='x', direction='out')
    ax.xaxis.tick_bottom()
    ax.text(10,10,'Speedup')
    ax.text(10,32,'Top-1')
    ax.text(10,49,'Top-5')
    ax.text(10,60,'Top-10')
    path = "./visualization"
    if not os.path.exists(path):
        os.mkdir(path)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{name}.pgf")) # pdf for the overleaf

if __name__ == "__main__":
    plot_flip()
