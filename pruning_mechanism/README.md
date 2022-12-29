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

To run the pruning mechanism on the Model 'openai/clip-vit-base-patch32' and the mscoco task run:

### 2.1 Generating the masks
```bash
cd fisher_pruning
python3 main.py --num_samples 8192 --constraint 0.65 --seed 123
```

This will use 2048 samples from mscoco and try to reduce the FLOP constraint to 70%.
The calculations are based on a constant number of patches.
The resulting head and neuron masks will be stored in fisher_pruning/outputs/openai/clip-vit-base-patch32/mscoco/mac/0.7/seed_5/.
Note: The constraint is only a broad estimate, because the number of image patches is variable in the MSCOCO dataset.

### 2.2  

The method from above only generated a pruning mask. 
To apply the mask and remove all unnecessary weights from the model:
1. open actual_pruning.py.
2. Change the variables "restriction" and "seed" on line 67 and 68 accordingly.
3. run the file
This generates the pruned model and saves it to the folder pruned_models/ with name vitB32_pruned_"restriction"_"seed".pt.
The pruned model can then be copied to the folder ../models/huggingface_pruned_clip/pruned_models/.
There it can be used as part of an experiment in the main project.

## 3. Reproducing the results from the paper of this main project

To reproduce the results, follow the instructions above, with the exact same settings (8192 samples, constraint 0.65 and seed 123).
Note: Running this code will overwrite the original output.


 **reference**

> 1.Andrew,  H, Mark S, Searching for MobileNetV3,, ICCV 2019
>
> 2.Alec Radford, Jong Wook K, Learning Transferable Visual Models From Natural Language Supervision, RMLR2021
