# Cascade CLIP

The project is based on the [MobilenetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) and [CLIP-base-patch32](http://proceedings.mlr.press/v139/radford21a)

## 1. Simple Start

the `reference.ipynb` is the final example of this project.

You can just open `reference.ipynb` with jupyter notebook and click the run button to see the result

## 2. Introduction

By using the `distill_cos.py`, one can distill the large ViT model into a small MobiletnetV3 which is small, faster and easy  to deploy. 

And the `verify.py` can test the distilling result on `cifar100` dataset. 

The `model.py` is the final CLIP and CascadeCLIP for final usage. You can find the network structure there.

Finally, just see the `reference.ipynb` to see how to use the CascadeCLIP to booster the inference of traditional CLIP.

By the way, the `script` directory is some command line script to run the file for either distilling or verifing

## 3.Experiment

### 3.1 Verifying for Distilling on Cifar100

Comparing distilling MobiletNetV3 on COCO train dataset for one epoch and ViT 

| ViT top-1 | mobilenet top-1 | mobilenet top-3 | mobiletnet top-5 | mobiletnet top-10 | mobiletnet top-20 |
| --------- | --------------- | --------------- | ---------------- | ----------------- | ----------------- |
| 0.5515    | $0.107(0.078)$  | $0.222(0.109)$  | $0.303(0.115)$   | $0.443(0.122)$    | $0.603(0.118)$    |

which means that the top20 score for MobileNetV3 is better than ViT top1. It's the basic logic of our project.

### 3.2 Inference Speed

GPU:MX450(2GB)

CPU:i5-11300H(16GB)

batch_size:4

topk:3

topm:100

text: an image of banana

| CLIP  | Image Encoding(CLIP) | Cascade CLIP | Small Image Encoding(Cas CLIP) | Large Image Encoding(Cas CLIP) |
| ----- | -------------------- | ------------ | ------------------------------ | ------------------------------ |
| 3m23s | 3m8s                 | 1m40s        | 1m22s                          | 4s                             |

#### 3.2.1 Top3 Result for CLIP

top1

![img](./.images/clip-top1.png)

top2

![img](./.images/clip-top2.png)

top3

![img](./.images/clip-top3.png)

### 3.2.2 Top3 Result for Cas CLIP

top1

![img](./.images/cas-clip-top1.png)

top2

![img](./.images/cas-clip-top2.png)

top3

![img](./.images/cas-clip-top3.png)



 **reference**

> 1.Andrew,  H, Mark S, Searching for MobileNetV3,, ICCV 2019
>
> 2.Alec Radford, Jong Wook K, Learning Transferable Visual Models From Natural Language Supervision, RMLR2021