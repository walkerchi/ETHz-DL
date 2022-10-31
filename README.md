# ETHz-DeepLearning-Project

## Setup

### Python environment

1. Create a new Python 3.9.7 environment.
1. Install all packages in `requirements.txt`.

### Datasets

#### MSCOCO

1. Download http://images.cocodataset.org/zips/val2017.zip and extract it into
   `datasets/mscoco`
1. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `datasets/mscoco`.

## Usage

1. Run `python run.py 0123` to run experiment 0123. Experiments are stored
   in `experiments`.

2. The previous command logs results to `experiments/0123/<RUN>/output.log`.
   The end of `output.log` shows a dictionary similar to this one:
   ```
   {
      'processing times': {
         'images': [407.3408706188202, 3423.6847863197327],
         'captions': [104.48231744766235, 152.48748326301575]
      },
      'top_k_accs': [
         {1: 0.0782, 5: 0.4088, 10: 0.528},
         {1: 0.091, 5: 0.4666, 10: 0.5818},
         {1: 0.0908, 5: 0.4714, 10: 0.5818}
      ]
   }
   ```
   `images` and `captions` store total processing times in seconds of the
   image and captioning submodels of the small and large models in that order.
   `top_k_accs` stores the top-k accuracies of the small model, the large model,
   and their combination in that order.