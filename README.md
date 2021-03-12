This repository contains code for the following paper:
> Deepak Mittal, Shweta Bhardwaj, Balaraman Ravindran, Mitesh M. Khapra. *Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks*. IEEE Conference on Winter Applications in Computer Vision 2018 [https://arxiv.org/abs/1801.10447].

- [Requirements](#Requirements)
- [Dataset](#Dataset)
- [Pre-Trained Models](#pretrainedModels)
- [Code organization](#code-organization)

# Requirements
* tensorflow: 1.3.0
* skimage: 0.14.5
* Python: 2.7.18 (supports Python3)
* tqdm: 4.59.0
* cPickle: 1.71
* argparse: 1.1

# Dataset
- ImageNet-1000: 2017 version
- To convert raw images to tf-records, you can refer to this code: https://github.com/shwetabhardwaj44/ImageNet_images_to_TFRecords

# Pre-Trained Models
Pretrained full VGG-16 model trained on ImageNet is uploaded here: https://drive.google.com/file/d/103FkgQqjClsBjx9PHVKNdV5RnSywI2qy/view?usp=sharing.
Save this checkpoint in ```model_baseline``` folder under ```models``` folder.

# Code Organization

1. Shell Scripts:
- ```run_TrainAndPrune.sh```: Stage-I
- ```run_Finetune.sh```: Stage-II

2. Main Code Files:
- ```train_and_prune.py```: Binary to prune layer-by-layer and train the model for one epoch.
- ```finetune_PrunedModel.py```: Binary to fine-tune (re-train) the final pruned model for around 20-25 epochs.

3. Pruning Masks: 
- ```generate_RandomMask.py```: 
- ```generate_EntropyMask.py```: 
- ```generate_ScaledEntropyMask.py```:
- ```generate_L1normMask.py```:

4. Configuration Files:
- ```config.py```:
- ```vgg_preprocessing.py```:

