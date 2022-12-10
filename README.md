# ETHz-DeepLearning-Project

## 1.Setup

### 1.1Python environment

1. Create a new Python 3.9.7 environment.
1. Install all packages in `requirements.txt`.

### 1.2Datasets

#### MSCOCO

1. Download http://images.cocodataset.org/zips/val2017.zip and extract it into
   `datasets/mscoco`.
   The images are in `datasets/mscoco/val2017/\d+.jpg`
2. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `datasets/mscoco`.
   The annotation file is in `datasets/mscoco/annotations_trainval2017/annotations/captions_val2017.json`

#### Filckr30

1. Download https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset and extract it into `dataset/flickr30`.
   The images are in `datasets/flickr30/flickr30k_images/\d+.jpg`.
   The caption file is in `datasets/filckr30/results.csv`.

## 2.Usage

1. Run `python run.py --id 1` to run experiment configure file `experiments/0001.toml`.

   Run `python run.py --name example` to run experiment configure file `experiments/example.toml`

2. The previous command logs results to `.log/<experiment-name>/<time>.log`.

## 3.Contribution

### 3.1 Style Guide

[Numpy Python Code Style](https://peps.python.org/pep-0008/) which is the PEP-484 style for python

### 3.2 Structure
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
  
### 3.3 New Model
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
         - images:     Union[List[PILImage], PILImage]
  
                        could input a list of PIL.Image or a single.

                        If input a single, the output shape will be [n_emb]
                        
                        else the output will be [n_image, n_emb]

         - batch_size: Optional[int]
  
                        if batch_size is `None`, it will visit the image iteratively,

                        else it will grab them in a dataloader and do it in batch

         - device:     str
        
                        The output device for the embedding

                        As the embeding is so big, so sometimes we should store them in cpu rather than gpu

                        Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`

         - verbose:    bool
          
                        if verbose, the tqdm progress bar will be showed 

                        else, the encoding process will keep silent

      **Returns**
         - emb_images: torch.FloatTensor[n_image, n_emb] or [e_emb]
         
                        the embedding of the encoded images

   - `encode_texts`
      **Parameters**
         - texts:      Union[List[str], str]
        
                        could input a list of str or a single.

                        If input a single, the output shape will be [n_emb]

                        else the output will be [n_text, n_emb]

         - batch_size: Optional[int]
         
                        if batch_size is `None`, it will visit the text iteratively,

                        else it will grab them in a dataloader and do it in batch

         - device:     str
        
                        The output device for the embedding

                        As the embeding is so big, so sometimes we should store them in cpu rather than gpu

                        Of course, the runtime device is different from output device which you can set through `.cpu()`  or `.cuda()`

         - verbose:    bool
         
                        if verbose, the tqdm progress bar will be showed 

                        else, the encoding process will keep silent

      **Returns**
         - emb_texts:  torch.FloatTensor[n_text, n_emb] or [e_emb]
        
                        the embedding of the encoded texts
