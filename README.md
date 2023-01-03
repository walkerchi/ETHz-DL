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
  python pruning_mechanism/actual_pruning.py
  python run.py --name 2lvl_fisher_topk # for the topk experiment
  python run.py --name 2lvl_fisher_speedup # for the speedup experiment
  ```
  
### Figure 3

````bash
python visualize.py
````

The output image will be the `visualization/visualization_huggingface_flip.(pgf/png)`

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

### Costumize Fisher Pruning

Follow the instructions from pruning_mechanism/README.md.

## Layout

TODO: Make prettier

- **datasets**: all dataset
  - mscoco
  - flickr30
  - ...
- **models**:all CLIP models
  - **huggingface_clip**:HuggingFace version of CLIP
  - **openai_clip**:OpenAI version of CLIP
  - **openai_flip**:OpenAI version of FLIP (Mingyuan Chi)
  - **huggingface_pruned_clip**: HuggingFace version of PrunedCLIP (Lars Graf)
  - ...
- **run.py**: main file
- **config.py**:Basic Configuration for experiement, models and dataset

## Contribute

### Style Guide

[Numpy Python Code Style](https://peps.python.org/pep-0008/) which is the PEP-484 style for python

### New Model
If you add new model.
1. you could **add your folder** under `models`.

2. And **import** your model class in `models\__init__.py`

3. Your model class should provide functions
   - `image_encoder_str`
    
      property, a string identify what kind of image encoder it is, (used for cache)
   
   - `text_encoder_str`
    
      property, a string identify what kind of text encoder it is, (used for cache)
      
   - `set_no_grad`
   
      **Parameters**
      
      - state: bool, default:True

         if state is True, the encoding process should be called inside `torch.with_nograd()` to minize the memory cost.


   - `encode_images`
   
      **Parameters**

      - images:   
        
         Union[List[PILImage], PILImage]

         could input a list of PIL.Image or a single.

         If input a single, the output shape will be [n_emb]
         
         else the output will be [n_image, n_emb]

      - batch_size: 
      
         Optional[int]

         if batch_size is `None`, it will visit the image iteratively,

         else it will grab them in a dataloader and do it in batch

      - device: 
        
         str

         The output device for the embedding

         As the embeding is so big, so sometimes we should store them in cpu rather than gpu

         Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`

      - verbose:    
      
         bool

         if verbose, the tqdm progress bar will be showed 

         else, the encoding process will keep silent

      **Returns**

      - emb_images: 
      
         torch.FloatTensor[n_image, n_emb] or [e_emb]
      
         the embedding of the encoded images

   - `encode_texts`
   
      **Parameters**

      - texts:      
      
         Union[List[str], str]

         could input a list of str or a single.

         If input a single, the output shape will be [n_emb]

         else the output will be [n_text, n_emb]

      - batch_size: 
      
         Optional[int]

         if batch_size is `None`, it will visit the text iteratively,

         else it will grab them in a dataloader and do it in batch

      - device:     
       
         str

         The output device for the embedding

         As the embeding is so big, so sometimes we should store them in cpu rather than gpu

         Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`

      - verbose:    
        
         bool

         if verbose, the tqdm progress bar will be showed 

         else, the encoding process will keep silent

      **Returns**
      
      - emb_texts:  
      
         torch.FloatTensor[n_text, n_emb] or [e_emb]
      
         the embedding of the encoded texts
