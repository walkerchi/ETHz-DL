# Fisher-based Pruning
This part of the project is based on [retraining-free-pruning GitHub](https://github.com/WoosukKwon/retraining-free-pruning).
Its aim is to prune a transformer without having to retrain the whole network.
The poject has been adapted in order to work for the Vision Transformer of [CLIP](http://proceedings.mlr.press/v139/radford21a) instead of language transformers.
Additionally it runs on CPU and GPU instead of only running on GPU and for the FLOP constrained problem formulation only.
The file fisher_pruning/modeling_clip.py has been taken from the [HuggingFace GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py).
It has been modified such that it can apply a head mask as explained in the retraining-free-pruning Paper: [A Fast Post-Training Pruning Framework for
Transformers](https://arxiv.org/pdf/2204.09656.pdf).
The file modeling_pruned_clip.py has been taken from the same repo as before, but only one small change had been made in oreder to run
the already pruned model that has varying dimensions.

## 1. Setup
### 1.1 Python Requirements
The Python requirements from ../requirements.txt are sufficient to run the code from this branch as well.
### 1.2 Datasets 
#### MSCOCO

1. Download http://images.cocodataset.org/zips/train2017.zip and extract it into
   `fisher_pruning/dataset`.
   The images are in `fisher_pruning/dataset/train2017/\d+.jpg`
2. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `fisher_pruning/dataset`.
   The annotation file is in `fisher_pruning/dataset/annotations_trainval2017/annotations/captions_train2017.json`

## 2. Running the Algorithm 

To run the pruning mechanism on the Model 'openai/clip-vit-base-patch16' and the mscoco task run:

### 2.1 Generating the masks
```bash
cd fisher_pruning
python3 main.py --gpu 1 --num_samples 8192 --constraint 0.75 --seed 1
```

This will use 8192 samples from mscoco and reduces the total number of FLOPs to 75% and is equivalent
to p=0.25 from the paper.
To run this a GPU with 16 GB of memory is required. If no GPU is available, the --gpu option can be set to 0. 
In that case it will take a long time to run the pruning mechanism.
The resulting head and neuron masks will be stored in fisher_pruning/outputs/openai/clip-vit-base-patch16/mscoco/0.65/seed_0/.

### 2.2 Create the pruned model 

The method from above only generated the pruning masks. 
To apply the mask and remove all unnecessary weights from the model:
1. open actual_pruning.py.
2. Change the variables "restriction" and "seed" on line 67 and 68 according to the masks you want to use.
3. run the file
This generates the pruned model and saves it to the folder pruned_models/ with name vitB16_pruned_"restriction"_"seed".pt.
The pruned model can be used as part of an experiment in the main project,
if the pruning name in the experiment toml file is adjusted to the file name of the model.

## 3. Reproducing the results from the paper of this main project

To reproduce the results, follow the instructions above, with the exact same settings (8192 samples, constraints 0.60 and 0.20 and seed 1).
Note: Running this code will overwrite the original output files.

# Sensitivity-based Pruning


## 1. Setup
### 1.1 Python Requirements
The Python requirements from ../requirements.txt are sufficient to run the code from this branch as well.
### 1.2 Datasets 
#### MSCOCO

1. Download http://images.cocodataset.org/zips/train2017.zip and extract it into
   `fisher_pruning/dataset`.
   The images are in `fisher_pruning/dataset/train2017/\d+.jpg`
2. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `fisher_pruning/dataset`.
   The annotation file is in `fisher_pruning/dataset/annotations_trainval2017/annotations/captions_train2017.json`

## 2. Running the Algorithm 

To run the pruning mechanism on the Model 'openai/clip-vit-base-patch16' and the mscoco task run:

### 2.1 Generating the masks
```bash
cd fisher_pruning
python3 main_16.py --gpu 1 --output_dir output
```
This will iteratively prune the layers and produce 12 different head masks at the dir output.
If no GPU is available, the --gpu option can be set to 0. 


### 2.2 Create the pruned model 

The method from above only generated the pruning masks. 
To apply the mask and remove all unnecessary weights from the model:
1. open actual_pruning.py.
2. Change the variables "restriction" and "seed" on line 67 and 68 according to the masks you want to use.
3. run the file
This generates the pruned model and saves it to the folder pruned_models/ with name vitB16_pruned_"restriction"_"seed".pt.
The pruned model can be used as part of an experiment in the main project,
if the pruning name in the experiment toml file is adjusted to the file name of the model.

## 3. Reproducing the results from the paper of this main project

To reproduce the results, follow the instructions above and move the pruned model to the dir pruned_models
