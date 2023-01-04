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

We tested this code on a machine with the following specs:

| Operating System | Ubuntu 20.04 |
|---|---|
| CPU | Intel Core i7-11800H @ 2.30GHz |
| Memory | 32 GB |
| GPU | None |

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

## Usage

The directory `experiments` stores the specification of each experiment in a `.toml` config file.

To run experiment `experiments/<EXPERIMENT>.toml`, execute

```sh
python run.py --name <EXPERIMENT>
```

The previous command logs results to `.log/<EXPERIMENT>/<time-of-execution>.log`.


## Reproduction

First, make sure that you satisfy all prerequisites in Section *Prerequisites*.

The pruned models can be downloaded at https://polybox.ethz.ch/index.php/s/1N8tNyRPRlRZDAq .

### Table 1

*you could also change the device setting to `cuda:0` and increase the `batch_size` in the corresponding .toml file to accelerate the evalutation*

The result will be printed in the console and also in the corresponding log file.

- Distillation(MobileNetV3-S)

  ```bash
  python run.py --name 2lvl_distill_topk_mobilenetv3-S # for the topk experiment
  python run.py --name 2lvl_distill_speedup_mobilenetv3-S # for the speedup experiment
  ```

- Distillation(MobilenetV3-L)

  ```bash
  python run.py --name 2lvl_distill_topk_mobilenetv3-L # for the topk experiment
  python run.py --name 2lvl_distill_speedup_mobilenetv3-L # for the speedup experimennt
  ```

- Distillation(ResNet34)

  ```bash
  python run.py --name 2lvl_distill_topk_resnet34 # for the topk experiment
  python run.py --name 2lvl_distill_speedup_resnet34 # for the speedup experiment
  ```

- Distillation(ResNet50)

  ```bash
  python run.py --name 2lvl_distill_topk_resnet50 # for the topk experiment
  python run.py --name 2lvl_distill_speedup_resnet50 # for the speedup experiment
  ```

- Masking(p1 = 0.5)

  ```bash
  python run.py --name 2lvl_flip_topk_2x # for the topk experiment
  python run.py --name 2lvl_flip_speedup_2x # for the speedup experiment
  ```

- Masking(p1 = 0.74)

  ```bash
  python run.py --name 2lvl_flip_topk_3x # for the topk experiment
  python run.py --name 2lvl_flip_speedup_3x # for the speedup experiment
  ```
  
- Sensitivity(22 heads pruned)

  ```bash
  python run.py --name 2lvl_sensitivity_topk_22 # for the topk experiment
  python run.py --name 2lvl_sensitivity_speedup_22 # for the speedup experiment
  ```
  
- Sensitivity(77 heads pruned)

  ```bash
  python run.py --name 2lvl_sensitivity_topk_77 # for the topk experiment
  python run.py --name 2lvl_sensitivity_speedup_77 # for the speedup experiment
  ```
  
- Sensitivity(132 heads pruned)

  ```bash
  python run.py --name 2lvl_sensitivity_topk_132 # for the topk experiment
  python run.py --name 2lvl_sensitivity_speedup_132 # for the speedup experiment
  ```
  
- Fisher(mac_constraint = 0.65)

  ```bash
  cd pruning_mechanism
  python actual_pruning.py
  cd ..
  python run.py --name 2lvl_fisher_topk # for the topk experiment
  python run.py --name 2lvl_fisher_speedup # for the speedup experiment
  ```
  
### Figure 3

````bash
python visualize.py
````

The output image will be the [`visualization/visualization_huggingface_flip.(pgf/png)`](visualization/visualization_huggingface_flip.png)

### Customize Distillation

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

### Custumize Fisher Pruning

Follow the [instructions](pruning_mechanism/README.md) from pruning_mechanism/README.md.

## Layout
 <details><summary><strong>datasets</strong>: all dataset</summary>
  <p>
  <ul>
  <li><em>mscoco.py</em> : MSCOCO dataset</li>
  <li><em>flickr30.py</em> ：Flickr30 dataset</li>
  </ul>
  </p>
  </details>

<details><summary><strong>models</strong>: all CLIP models</summary>
  <p>
  <ul>
  <li> <em> huggingface_clip.py</em>: HuggingFace version of CLIP </li>
  <li> <em>openai_clip.py</em>: OpenAI version of CLIP </li>
  <li> <em>openai_flip.py</em>: OpenAI version of FLIP </li>
  <li> <em>huggingface_flip.py</em> : HuggingFace version of FLIP </li>
  <li> <em>huggingface_pruned_clip.py</em>: HuggingFace version of PrunedCLIP </li>
  </ul>
  </p>
</details>  

**pruning_mechanism**: codes about pruning

*run.py*: main execution file

*config.py*: Basic Configuration for experiement, models and dataset

*visualize.py* : Visualization for the casCLIP(Masking) performance

