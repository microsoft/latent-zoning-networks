# Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification

**[[paper (NeurIPS 2025)](https://openreview.net/forum?id=8KuKSKLott)]**
**[[paper (arXiv)](https://arxiv.org/abs/2509.15591)]**
**[[website](https://zinanlin.me/blogs/latent_zoning_networks.html#post)]**
**[[code](https://github.com/microsoft/latent-zoning-networks)]**
**[[models](https://huggingface.co/microsoft/latent-zoning-networks)]**


**Authors:** [Zinan Lin](https://zinanlin.me), [Enshu Liu](https://scholar.google.com/citations?user=0LUhWzoAAAAJ), [Xuefei Ning](https://nics-effalg.com/ningxuefei/), [Junyi Zhu](https://junyizhu-ai.github.io/), [Wenyu Wang](#), [Sergey Yekhanin](https://www.microsoft.com/en-us/research/people/yekhanin/)

**Correspondence to:** [Zinan Lin](https://zinanlin.me) (zinanlin AT microsoft DOT com)

**Abstract:** **Generative modeling, representation learning, and classification** are three core problems in machine learning (ML), yet their state-of-the-art (SoTA) solutions remain largely disjoint. In this paper, we ask: **Can a unified principle address all three?** Such unification could simplify ML pipelines and foster greater synergy across tasks. We introduce Latent Zoning Network (LZN) as a step toward this goal. 

At its core, LZN creates a shared Gaussian latent space that encodes information across all tasks. Each data type (e.g., images, text, labels) is equipped with an encoder that maps samples to disjoint latent zones, and a decoder that maps latents back to data. ML tasks are expressed as compositions of these encoders and decoders: for example, label-conditional image generation uses a label encoder and image decoder; image embedding uses an image encoder; classification uses an image encoder and label decoder. 

We demonstrate the promise of LZN in three increasingly complex scenarios: **(1) LZN can enhance existing models (image generation)**: When combined with the SoTA Rectified Flow model, LZN improves FID on CIFAR10 from 2.76 to 2.59—without modifying the training objective. **(2) LZN can solve tasks independently (representation learning)**: LZN can implement unsupervised representation learning without auxiliary loss functions, outperforming the seminal MoCo and SimCLR methods by 9.3% and 0.2%, respectively, on downstream linear classification on ImageNet. **(3) LZN can solve multiple tasks simultaneously (joint generation and classification)**: With image and label encoders/decoders, LZN performs both tasks jointly by design, improving FID and achieving SoTA classification accuracy on CIFAR10.

## News
* `2/20/2026`: The models and code for all datasets and tasks are released!
* `9/21/2025`: 🚀 The models and training/inference code for **image generation on AFHQ-Cat** and **image embedding trained on ImageNet** have been released! Due to the sensitive nature of the datasets, the remaining models and code are undergoing an internal review process and will be released at a later date. Stay tuned!
  * Code: https://github.com/microsoft/latent-zoning-networks
  * Models: https://huggingface.co/microsoft/latent-zoning-networks
* `9/21/2025`: The paper is released [here](https://arxiv.org/abs/2509.15591).

## Environment Setup

We provide a docker file with the necessary dependencies in [`docker/Dockerfile`](docker/Dockerfile). Alternatively, you can install PyTorch as well as the libraries in [`docker/requirements.txt`](docker/requirements.txt) to set up the environment.

## Distributed Training Configuration

The training code uses Distributed Data Parallelism (DDP) for multi-GPU and multi-node training. In the configuration files, specify the number of GPUs per node using `config.distributed.num_gpus_per_node`.

For **single-node training**, this is all you need—the training script will automatically launch processes on each GPU.

For **multi-node training**, the training command must be executed once on each node, and the following environment variables need to be set:

- `MASTER_ADDR`: The IP address of the master node.
- `MASTER_PORT`: The port number on the master node for inter-node communication.
- `WORLD_SIZE`: The total number of nodes.
- `NODE_RANK`: The rank of the current node, starting from 0 on the master node.


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
Note that the root folder ``/tmp/data/AFHQ`` can be moved to any locations, as long as it matches `config.data.params.afhq_root` in the configuration file [configs/lzn1/case_study_1_afhqcat.py](configs/lzn1/case_study_1_afhqcat.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_1_afhqcat.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_afhqcat`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_1_afhqcat/ema-rk45-random_samples_array/` and `results/case_study_1_afhqcat/ema-rk45-random_samples_more_array`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_1_afhqcat.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_1_afhqcat/000003000-000060000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_1_afhqcat/000003000-000060000.pt).

Simiarly to model training, the results (logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_afhqcat`. 




### Celeba-HQ

#### Data Preparation

Download and preprocess the [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans) and put the images in the following folder structure:

```
📦/tmp/data/CelebA-HQ-1024
 ┣ 📜img00000000.png
 ┣ 📜img00000001.png
 ┗  ... (more images)
```
Note that the root folder ``/tmp/data/CelebA-HQ-1024`` can be moved to any locations, as long as it matches `config.data.params.root` in the configuration file [configs/lzn1/case_study_1_celebahq.py](configs/lzn1/case_study_1_celebahq.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_1_celebahq.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_celebahq`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_1_celebahq/ema-rk45-random_samples_array/` and `results/case_study_1_celebahq/ema-rk45-random_samples_more_array`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_1_celebahq.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_1_celebahq/000002991-000350000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_1_celebahq/000002991-000350000.pt).

Simiarly to model training, the results (logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_celebahq`. 



### LSUN Bedroom

#### Data Preparation

Download and preprocess the [LSUN Bedroom dataset](https://github.com/tkarras/progressive_growing_of_gans) and put the images in the following folder structure:

```
📦/tmp/data/LSUN
 ┣ 📂bedroom_train
 ┃ ┣ 📜img00000000.png
 ┃ ┣ 📜img00000001.png
 ┃ ┗  ... (more images)
 ┗ 📂bedroom_val
 ┃ ┣ 📜img00000000.png
 ┃ ┣ 📜img00000001.png
 ┃ ┗  ... (more images)
```
Note that the root folder ``/tmp/data/LSUN`` can be moved to any locations, as long as it matches `config.data.params.lsun_root` in the configuration file [configs/lzn1/case_study_1_lsunbedroom.py](configs/lzn1/case_study_1_lsunbedroom.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_1_lsunbedroom.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_lsunbedroom`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_1_lsunbedroom/ema-rk45-random_samples_array/` and `results/case_study_1_lsunbedroom/ema-rk45-random_samples_more_array`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_1_lsunbedroom.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_1_lsunbedroom/000000173-002050000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_1_lsunbedroom/000000173-002050000.pt).

Simiarly to model training, the results (logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_lsunbedroom`. 



### CIFAR10

#### Data Preparation

The data will be downloaded automatically by the scripts, so no manual data preparation is needed for CIFAR10.

#### Model Training

```
python train.py --config=configs/lzn1/case_study_1_cifar10.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_cifar10`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_1_cifar10/ema-rk45-random_samples_array/` and `results/case_study_1_cifar10/ema-rk45-random_samples_more_array`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_1_cifar10.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_1_cifar10/000002000-000050000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_1_cifar10/000002000-000050000.pt).

Simiarly to model training, the results (logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_1_cifar10`. 






## Unsupervised Representation Learning (Case Study 2)

### ImageNet

#### Data Preparation

Please download the [ImageNet dataset](http://www.image-net.org/) and place `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar`, and `ILSVRC2012_img_val.tar` in `/tmp/data/ImageNet`.

Note that the root folder ``/tmp/data/ImageNet`` can be other locations, as long as it matches `config.data.params.root` in the configuration file [configs/lzn1/case_study_2.py](configs/lzn1/case_study_2.py).

#### Model Training

```
python train.py --config=configs/lzn1/case_study_2.py
```

The results (checkpoints, logs, image representations, linear classification accuracies, etc.) will be saved in the folder `results/case_study_2`. For examples, the image representations of the validation set can be found in `results/cast_study_2/ema-representation_no_head_representations/`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_2.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_2/000032051-005000000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_2/000032051-005000000.pt).

Simiarly to model training, the results (logs, image representations, linear classification accuracies, etc.) will be saved in the folder `results/case_study_2`. 






## Conditional Generative Modeling and Classification (Case Study 3)


### CIFAR10

#### Data Preparation

The data will be downloaded automatically by the scripts, so no manual data preparation is needed for CIFAR10.

#### Model Training

```
python train.py --config=configs/lzn1/case_study_3.py
```

The results (checkpoints, logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_3`. For examples, the generated images using the RK45 sampler in the Numpy format can be found in `results/case_study_3/ema-rk45-random_samples_array/` and `results/case_study_3/ema-rk45-random_samples_more_array`.

#### Model Inference

```
python evaluate.py --config=configs/lzn1/case_study_3.py --config.checkpoint.load_checkpoint=manual --config.checkpoint.path="<path to the checkpoint>"
```
where the checkpoint `lzn1/case_study_3/000008000-000200000.pt` can be downloaded from [here](https://huggingface.co/microsoft/latent-zoning-networks/resolve/main/lzn1/case_study_3/000008000-000200000.pt).

Simiarly to model training, the results (logs, metrics, generated images, etc.) will be saved in the folder `results/case_study_3`. 



## Citation

Please cite the paper if you use this code:
```
@article{lin2025latent,
  title={Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification},
  author={Lin, Zinan and Liu, Enshu and Ning, Xuefei and Zhu, Junyi and Wang, Wenyu and Yekhanin, Sergey},
  journal={arXiv preprint arXiv:2509.15591},
  year={2025}
}
```

## Privacy

See [Microsoft Privacy Statements](https://go.microsoft.com/fwlink/?LinkId=521839).

##  Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
