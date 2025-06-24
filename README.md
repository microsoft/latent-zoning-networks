# Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification

**[[paper (arXiv)](#)]**
**[[code](https://github.com/microsoft/latent-zoning-networks)]**


**Authors:** [Zinan Lin](https://zinanlin.me), [Enshu Liu](https://scholar.google.com/citations?user=0LUhWzoAAAAJ), [Xuefei Ning](https://nics-effalg.com/ningxuefei/), [Junyi Zhu](https://junyizhu-ai.github.io/), [Wenyu Wang](#), [Sergey Yekhanin](https://www.microsoft.com/en-us/research/people/yekhanin/)

**Correspondence to:** [Zinan Lin](https://zinanlin.me) (zinanlin AT microsoft DOT com)

**Abstract:** Generative modeling, representation learning, and classification are three core problems in machine learning (ML), yet their state-of-the-art (SoTA) solutions remain largely disjoint. In this paper, we ask: Can a unified principle address all three? Such unification could simplify ML pipelines and foster greater synergy across tasks. We introduce Latent Zoning Network (LZN) as a step toward this goal. At its core, LZN creates a shared Gaussian latent space that encodes information across all tasks. Each data type (e.g., images, text, labels) is equipped with an encoder that maps samples to disjoint latent zones, and a decoder that maps latents back to data. ML tasks are expressed as compositions of these encoders and decoders: for example, label-conditional image generation uses a label encoder and image decoder; image embedding uses an image encoder; classification uses an image encoder and label decoder. We demonstrate the promise of LZN in three increasingly complex scenarios: (1) LZN can enhance existing models (image generation): When combined with the SoTA Rectified Flow model, LZN improves FID on CIFAR10 from 2.76 to 2.59—without modifying the training objective. (2) LZN can solve tasks independently (representation learning): LZN can implement unsupervised representation learning without auxiliary loss functions, outperforming the seminal MoCo method by 5.4% on downstream linear classification on ImageNet. (3) LZN can solve multiple tasks simultaneously (joint generation and classification): With image and label encoders/decoders, LZN performs both tasks jointly by design, improving FID and achieving SoTA classification accuracy on CIFAR10. Code and models will be released.

## News
* `X/XX/XXXX`: 🚀 The models and training/inference code for **image generation on AFHQ-Cat** and **image embedding trained on ImageNet** have been released! The rest of the models and code are undergoing internal review process and will be released at a later date. Stay tuned!
  * Code: https://github.com/microsoft/latent-zoning-networks
  * Models: https://huggingface.co/microsoft/latent-zoning-networks
* `X/XX/XXXX`: The paper is released [here](#).

## Environment Setup

We provide a docker file with the necessary dependencies in [`docker/Dockerfile`](docker/Dockerfile). Alternatively, you can install PyTorch as well as the libraries in [`docker/requirements.txt`](docker/requirements.txt) to set up the environment.

## Unconditional Generative Modeling (Case Study 1)

### AFHQ-Cat

#### Data Preparation

Download the [AFHQ-Cat dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and put the images in the following folder structure:

```
📦/tmp/data/AFHQ
 ┣ 📂train
 ┃ ┣ 📂cat
 ┃ ┃ ┣ 📜flickr_cat_000002.jpg
 ┃ ┃ ┗  ... (more images)
 ┗ 📂val
 ┃ ┣ 📂cat
 ┃ ┃ ┣ 📜flickr_cat_000008.jpg
 ┃ ┃ ┗  ... (more images)
```
Note that the root folder ``/tmp/data/AFHQ`` can be moved to any locaton, as long as it matches `config.data.params.afhq_root` in the configuration file [configs/lzn1/case_study_1_afhqcat.py](configs/lzn1/case_study_1_afhqcat.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_1_afhqcat.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_afhqcat`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_1_afhqcat/ema-rk45-random_samples_array/`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_1_afhqcat.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_1_afhqcat/000003000-000060000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_1_afhqcat/000003000-000060000.pt).

Simiarly to model training, the results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_afhqcat`. 

## Unsupervised Representation Learning300 (Case Study 2)

### ImageNet

#### Data Preparation

Please download the [ImageNet dataset](http://www.image-net.org/) and place `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar`, and `ILSVRC2012_img_val.tar` in `/tmp/data/ImageNet`.

Note that the root folder ``/tmp/data/ImageNet`` can be other locatons, as long as it matches `config.data.params.root` in the configuration file [configs/lzn1/case_study_2.py](configs/lzn1/case_study_2.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_2.py
```

The results (checkpoints, logs, linear classification accuries, etc.) will be saved in the folder `results/case_study_2`. ema-rk45-random_samples_array/`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_2.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn2/case_study_2/XXXXX.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_2/XXXXX.pt).

Simiarly to model training, the results (checkpoints, logs, linear classification accuries, etc.) will be saved in the folder `results/case_study_2`. 
