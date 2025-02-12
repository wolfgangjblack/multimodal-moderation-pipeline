# autoimagerater
Modeling Suite for labeling images with movie ratings PG, PG13, R, X, and XXX


This repo contains code for training ResNets, Berts, a Roberta, and a Vision Transformer on image and text modalities to determine the movie-like nsfw level for images in the CivitAI ecosystem. The repo gathers data from the [postgres tables](https://github.com/civitai/autoimagerater/blob/main/EDA/getData.ipynb); [downloads](https://github.com/civitai/autoimagerater/blob/main/EDA/downloadData.ipynb) the images, prompts, and tags locally; and then trains the various architectures off of community labeled data generated from the `research_rater` mini-game found on the site. The training data is selected iff the data has over 3 votes with >50% agreement on the voted label. Data is also downsampled to the minority class, which is often a lower NSFW rating (sometimes R). After training, the models then view data rated specifically by Justin or the hold-out unsampled data to determine performance. Each model rates the data and afterwards we can compare scores from any number of the individual models to determine a `mixture`. Performance data is then saved in `results`. 

## Model Details

### Training Data Description

The models are trained on data provided by the following query:

```
redacted
```

### Models

#### ResNet Architectures
We have the ability to train resnet18s and 50s with augmentation to the training data, with k-folds validation, or with no-agumentation and a randomly-sampled test/train split. As of May, we have not found increasing depth beyond 50 to be beneficial comparitively to training costs. For training loops see [train_resnet_architecture](https://github.com/civitai/autoimagerater/blob/main/project/utils/utils.py#L289)


### NLP Architectures
We train a number of [Bert](https://github.com/civitai/autoimagerater/blob/main/EDA/trainPromptTransformerClassifier.ipynb) based transformer models providing the prompts, or a combination of the prompts and all available tags to the models. This may be semi-fragile since we don't have control of all of the tags. So these models should be updated as we change the tagging systems. 

### Vision Transformer Architecture
Added in May2024, we wanted to determine if we had enough data for a [ViT](https://github.com/civitai/autoimagerater/blob/main/EDA/trainViTClassifier.ipynb) to replace the more simple ResNets. The ViT performs very well, with a slight performance increase over the resnet50s. Due to the improvements in the transformers library this also trains very quickly. However, replacing the three resnet models in the mixture slightly worsened the overall mixture performance by 1-2% in each category. As such, more experimentation should be done to determine if replacing a single resnet with ViT yields better results. 

## Uses

To be used to rate images as PG, PG13, R, X and XXX. A style guide will be released and this will be updated for more explicit definitions later.

PG - Family Friend
PG13 - Appropriate for Teens
R - More adult themes
X - Nudity, nipples, genetalia
XXX - Sexual activity or implied sexual activity 

### Out-of-Scope Use
As of this moment video/multiframe objects are out of scope. However, I'm looking into 

## Bias, Risks, and Limitations

The data fed into this model isn't perfect. A lot of the images in each class are mislabeled and the labeling has been done without a strict style guide. The labeling has also been done by multiple moderators over a long amount of time

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

- increase label fidelity
  
## Results

There is now a results directory where we'll save all tests/updates to track individual model performance/training performance. This will contain markdown table text, classification reports, and confusion matrix plots. We'll also save data for disagreement plots to be reviewed later. 
