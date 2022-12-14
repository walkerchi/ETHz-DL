# Fisher-based Pruning
This project is based on [retraining-free-pruning GitHub](https://github.com/WoosukKwon/retraining-free-pruning).
Named poject has been heavily modified in order to work for the Vision Transformer of [CLIP](http://proceedings.mlr.press/v139/radford21a) instead of language transformers.
Additionally it runs on CPU instead of GPU and for the FLOP constrained problem formulation only.
The file fisher_pruning/modeling_clip.py has been taken from the [HuggingFace GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py).
It has been modified such that it can apply a head mask as explained in the retraining-free-pruning Paper: [A Fast Post-Training Pruning Framework for
Transformers](https://arxiv.org/pdf/2204.09656.pdf).

## 1. Simple Start

The Python requirements from the main branch are sufficient to run the code from this branch.
To run the pruning mechanism on the Model 'openai/clip-vit-base-patch32' and the mscoco task run:
$$
\begin{aligned}
cd fisher_pruning
python3 main.py --num_samples 2048 --constraint 0.5 --seed 5
\end{aligned}
$$
This will use 2048 samples from mscoco and try to reduce the FLOP constraint to 50%.
The calculations are based on a constant number of patches.
The resulting head and neuron masks will be stored in fisher_pruning/outputs/openai/clip-vit-base-patch32/mscoco/mac/0.5/seed_5/.
Note: The constraint is only a broad estimate, because the number of image patches is variable in the MSCOCO dataset.

## 2. Introduction

 **reference**

> 1.Andrew,  H, Mark S, Searching for MobileNetV3,, ICCV 2019
>
> 2.Alec Radford, Jong Wook K, Learning Transferable Visual Models From Natural Language Supervision, RMLR2021
