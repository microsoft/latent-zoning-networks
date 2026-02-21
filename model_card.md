# Model card for Latent Zoning Networks

## Model details

### Model description

Generative modeling, representation learning, and classification are three core problems in machine learning (ML), yet their state-of-the-art (SoTA) solutions remain largely disjoint. In this paper, we ask: Can a unified principle address all three? Such unification could simplify ML pipelines and foster greater synergy across tasks. We introduce Latent Zoning Network (LZN) as a step toward this goal. At its core, LZN creates a shared Gaussian latent space that encodes information across all tasks. Each data type (e.g., images, text, labels) is equipped with an encoder that maps samples to disjoint latent zones, and a decoder that maps latents back to data. ML tasks are expressed as compositions of these encoders and decoders: for example, label-conditional image generation uses a label encoder and image decoder; image embedding uses an image encoder; classification uses an image encoder and label decoder. We demonstrate the promise of LZN in three increasingly complex scenarios: (1) LZN can enhance existing models (image generation): When combined with the SoTA Rectified Flow model, LZN improves FID on CIFAR10 from 2.76 to 2.59—without modifying the training objective. (2) LZN can solve tasks independently (representation learning): LZN can implement unsupervised representation learning without auxiliary loss functions, outperforming the seminal MoCo and SimCLR methods by 9.3% and 0.2%, respectively, on downstream linear classification on ImageNet. (3) LZN can solve multiple tasks simultaneously (joint generation and classification): With image and label encoders/decoders, LZN performs both tasks jointly by design, improving FID and achieving SoTA classification accuracy on CIFAR10.

The list of the released models are:

* Image generation on AFHQ Cat dataset

* Image embedding on ImageNet dataset

* Image generation model on CIFAR10 dataset 

* Image generation (conditional) and classification model on CIFAR10 dataset

* Image generation model on LSUN Bedroom dataset 

* Image generation model on Celeba-HQ dataset

The models are trained from scratch.

### Key information

* Developed by: Zinan Lin

* Model type: Image generation models, image embedding models, and image classification models

* Language(s): The models do NOT have text input or output capabilities

* License: MIT

### Model sources

* Model repository: https://huggingface.co/microsoft/latent-zoning-networks

* Code repository:  https://github.com/microsoft/latent-zoning-networks

* Paper: https://arxiv.org/abs/2509.15591

* Project page: https://zinanlin.me/blogs/latent_zoning_networks.html#post

## Uses

### Direct intended uses

* Image generation models on AFHQ Cat, LSUN Bedroom, CelebA-HQ: These are unconditional image generation models. The models do not require any input such as class conditions. The model can generate new images similar to the training set. 

* Image generation and classification model on CIFAR10: This model can work as either a conditional image generation model or an image classification model. For conditional image generation, the model can generate a random image based on the index of the desired class (1\~10). For image classification, the model can generate the class index (1\~10) for the input image.

* Image embedding models: Given an image, the model can give the embedding (i.e., a vector of float numbers) of the image.

The released models do not currently have real-world applications. It is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

### Out-of-scope uses

These models do NOT have text-conditioned image generation capabilities, and cannot generate anything beyond images. We do not recommend using the models in commercial or real-world applications without further testing and development. It is being released for research purposes.

## Risks, limitations, and mitigation

The quality of generated images is not perfect and might contain artifacts such as blurry or unrecognizable objects. If users find failure cases of the models, please contact us and we will update the arXiv paper to report such failure cases. If the models have severe and unexpected issues, we will remove the models from HuggingFace.

These models inherit any biases, errors, or omissions characteristic of their training data, which may be amplified by any AI-generated interpretations.

We used five specific datasets to demonstrate our technique for training image generation and embedding models. If users/developers wish to test our technique using other datasets, it is their responsibility to source those datasets legally and ethically. This could include securing appropriate rights, ensuring consent for the use of images, and/or the anonymization of data prior to use. Users are reminded to be mindful of data privacy concerns and comply with relevant data protection regulations and organizational policies.

## How to get started with the model

Please see the GitHub repo for instructions: https://github.com/microsoft/latent-zoning-networks

## Training details

### Training data

* Image generation:
    * AFHQ Cat dataset https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq
    * CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html 
    * LSUN Bedroom dataset (photos of bedrooms): https://github.com/fyu/lsun
    * CelebA-HQ dataset: https://github.com/tkarras/progressive_growing_of_gans

* Image embedding: ImageNet dataset http://www.image-net.org/

Some public image datasets, including datasets containing human or celebrity images, were used only for research benchmarking and evaluation. The models are not designed or trained to recreate or generate identifiable individuals, do not accept identity‑based inputs, and cannot be steered to produce specific people. Any human‑like images are generated from random noise and are not controllable or repeatable. These models are released as part of a research effort and are not intended for real-world application including image generation.

### Training procedure

### Preprocessing

* Image generation and classification: Please see the paper for details: https://arxiv.org/abs/2509.15591 

* Image embedding: Please see the paper for details: https://arxiv.org/abs/2509.15591 


### Training hyperparameters

Please see the paper for details: https://arxiv.org/abs/2509.15591

### Speeds, sizes, times

Please see the paper for details: https://arxiv.org/abs/2509.15591

## Evaluation

### Testing data, factors, and metrics

#### Testing data

* Image generation: AFHQ Cat, CIFAR10, LSUN Bedroom, CelebA-HQ datasets

* Image classification: CIFAR10 dataset

* Image embedding: ImageNet dataset

#### Metrics

* Image generation: Image quality metrics including FID, Inception Score, Precision, Recall

* Image classification: classification accuracy

* Image embedding: Downstream image classification accuracy

## Evaluation results

* Image generation: The image quality of Latent Zoning Network models are better than the baselines. For example, on the AFHQ Cat dataset, latent zoning networks improve the FID, sFID, IS, Precision, Recall, and Reconstruction from 6.08, 49.60, 1.80, 0.86, 0.28, 17.92 to 5.68, 49.32, 1.96, 0.87, 0.30, 10.29, respectively.

* Image embedding: The downstream image classification accuracy of Latent Zoning Network is on par with state-of-the-art approaches. For example, we train a linear classifier on top of the embedding and evaluate its accuracy on the ImageNet test set. The accuracy of latent zoning networks is 69.5%, beating the seminal MoCo method by 9.3% and SimCLR by 0.2%.

* Image classification: The image classification accuracy on CIFAR10 dataset is 94.47%, which is close to the state-of-the-art 95.47%.

## Summary

Overall, the results demonstrate that the Latent Zoning Network is a viable, unified framework to address multiple machine learning problems.

## License

MIT

Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under.

## Citation

* BibTeX: 
```
@article{lin2025latent,
  title={Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification},
  author={Lin, Zinan and Liu, Enshu and Ning, Xuefei and Zhu, Junyi and Wang, Wenyu and Yekhanin, Sergey},
  journal={arXiv preprint arXiv:2509.15591},
  year={2025}
}
```

## Model card contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at Zinan Lin, zinanlin@microsoft.com.

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.