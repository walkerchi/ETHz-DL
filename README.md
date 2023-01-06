# Model Cascades for Efficient Image Search

This is the repository for the 2022 Deep Learning project *Model Cascades
for Efficient Image Search*.

* `Prerequisites` describes how to set up your environment to run our experiments.

* `Usage` explains how to run our experiments.

* `Reproduction` explains how to reproduce individual results in our paper.

* `Layout` presents the content and organization of this repository.

* `Contribute` explains how to add new models.

## Prerequisites

### System

We tested this code on a workstation in the ETH Euler cluster with 32 cores of an AMD EPYC 7742 CPU and 128 Gigabytes of RAM.

### Software

0. Create a new Python 3.9.7 environment.

1. Install all packages in `requirements.txt` by running
   ```
   pip install -r requirements.txt
   ```
### Datasets

#### MSCOCO

1. Download http://images.cocodataset.org/zips/val2017.zip and extract it into
   `datasets/mscoco`.
   The images are in `datasets/mscoco/val2017/\d+.jpg`

2. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `datasets/mscoco`.

3. (If you want to do the distilling) Download http://images.cocodataset.org/zips/train2017.zip and extract it into `datasets/mscoco`.

   The images are in `datasets/mscoco/train2017/\d+.jpg`

#### Flickr30k

1. Download https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset and extract it into `dataset/flickr30`.

   The captions should be in  `datasets/flickr30/results.csv`

   The images are in `datasets/flickr30/flickr30k_images/\d+.jpg`

### Models

#### Distillation

The following instructions create all distilled models
necessary for our experiments.

- MobileNetV3-S

  ```bash
  python run.py --name distilling_mobilenetv3-S
  ```

- MobileNetV3-L

  ```bash
  python run.py --name distilling_mobilenetv3-L
  ```

- ResNet34

  ```bash
  python run.py --name distilling_resnet34
  ```

- ResNet50

  ```bash
  pyton run.py --name distilling_resnet50
  ```

#### Fisher Pruning

To create the pruned models necessary for our experiments, follow the instructions in [ pruning_mechanism/README.md](pruning_mechanism/README.md).


## Usage

The directory `experiments` stores the specification of each experiment in a `.toml` config file.

To run experiment `experiments/<EXPERIMENT>.toml`, execute

```sh
python run.py --name <EXPERIMENT>
```

The previous command logs results to `.log/<EXPERIMENT>/<time-of-execution>.log`.

*Tip: To accelerate experiments that measure model accuracy, set `device = 'cuda:0'`  and increase `batch_size` in the <EXPERIMENT.toml.*


## Reproduction

First, make sure that you satisfy all prerequisites in Section *Prerequisites*.

### Table 1
| Method | Top-k  < EXPERIMENT > .toml | Speedup < EXPERIMENT > .toml |
|---|---|---|
| No Cascade | 2lvl_noscacade_topk | 2lvl_noscacade_speedup |
| Model Retraining | 2lvl_retraining_speedup | 2lvl_retraining_topk |
| Fisher Pruning (p=0.6) | 2lvl_fisher_topk_0.6 | 2lvl_fisher_speedup_0.6 |
| Fisher Pruning (p=0.25) | 2lvl_fisher_topk_0.2 | 2lvl_fisher_speedup_0.2 |
| Sensitivity Pruning (p=0.17) | 2lvl_sensitivity_topk_22 | 2lvl_sensitivity_speedup_22 |
| Sensitivity Pruning (p=1) | 2lvl_sensitivity_speedup_132 | 2lvl_sensitivity_speedup_132 |
| Distillation (MobileNet) | 2lvl_distill_topk_mobilenetv3-S | 2lvl_distill_speedup_mobilenetv3-S |
| Distillation (ResNet) | 2lvl_distill_topk_resnet50 | 2lvl_distill_speedup_resnet50 |
| Masking (p=0.5) | 2lvl_masking_topk_2x | 2lvl_masking_speedup_2x |
| Masking (p=0.74) | 2lvl_masking_topk_3x | 2lvl_masking_speedup_3x |

### Table 2

| Method | Top-k <EXPERIMENT>.toml | Speedup <EXPERIMENT>.toml |
|---|---|---|
| No Cascade | 2lvl_nocascade_topk_flickr30k | / |
| Model Retraining | 2lvl_retraining_topk_flickr30k | 2lvl_retraining_speedup_flickr30k |
| Input Masking $p_1=0.5$ | 2lvl_masking_topk_2x_flickr30k | 2lvl_masking_speedup_2x_flickr30k |
| Input Masking $p_1=0.74$ | 2lvl_masking_topk_3x_flickr30k | 2lvl_masking_speedup_3x_flickr30k |

### Table 3

| Method | Top-k <EXPERIMENT>.toml | Speedup <EXPERIMENT>.toml |
|---|---|---|
| 2-level Masking | 2lvl_flip_topk_2x | 2lvl_flip_speedup_2x |
| 3-level Masking | 3lvl_flip_topk_2x | 3lvl_flip_speedup_2x |
  
### Figure 3

Run the following command:

````bash
python visualize.py
````

This will write the plot for Figure 3 into the directory
`visualization`.


## Layout

```
├── config.py         
├── datasets          # Stores MSCOCO Flickr30k.
├── experiments       # Stores experiment configuration files.
├── LICENCSE
├── models            # Implements baseline models, FLIP and Distillation.
├── pruning_mechanism # Implements Pruning.
├── README.md
├── requirements.txt
├── run.py            # Runs experiments. Main interface.
├── visualization     # Stores output of visualize.py.
└── visualize.py      # Creates Figure 3.
```