# Multiclass Fish Image Classification

## Project Overview

This project focuses on classifying fish images into multiple categories using deep learning techniques.  
The objective is to compare a custom CNN model with several transfer learning models and deploy the best-performing model through a Streamlit web application.

The project includes:

- Image preprocessing and augmentation
- CNN model built from scratch
- Transfer learning using five pretrained architectures
- Model evaluation using multiple classification metrics
- Model comparison and best-model selection
- Streamlit deployment for real-time fish species prediction

---

## Problem Statement

The goal of this project is to classify fish images into multiple species categories using deep learning models.

The project involves:

1. Training a CNN model from scratch
2. Training and fine-tuning pretrained models
3. Comparing all models using evaluation metrics
4. Saving the best-performing model
5. Building a Streamlit application for prediction on uploaded fish images

---

## Dataset

The dataset contains fish images organized into separate folders based on species classes.

### Dataset Structure

```text
data/
├── train/
├── val/
└── test/
Each of the above folders contains the following 11 classes:

animal fish
animal fish bass
fish sea_food black_sea_sprat
fish sea_food gilt_head_bream
fish sea_food hourse_mackerel
fish sea_food red_mullet
fish sea_food red_sea_bream
fish sea_food sea_bass
fish sea_food shrimp
fish sea_food striped_red_mullet
fish sea_food trout
```

## Dataset Size

|Split	|Number of Images|
|:---:|:---:|
|Training|	6,225|
|Validation|	1,092|
|Testing|	3,187|

# Project Workflow

## 1.Data Preprocessing

The following preprocessing steps were applied:

 - Resizing images to 224 x 224
 - Data augmentation on training images:
     - rotation
     - zoom
     - horizontal flipping
     - width shift
 - height shift
 - Handling class imbalance using class weights


## 2.Models Trained

The following models were trained and evaluated:

 - CNN from scratch
 - EfficientNetB0
 - MobileNetV2
 - ResNet50
 - VGG16
 - InceptionV3

## 3.Evaluation Metrics

The models were compared using:

 - Accuracy
 - Precision
 - Recall
 - F1-score
 - Confusion matrix

Because the dataset is imbalanced, macro F1-score was used as the primary metric for final model selection.

## Model Performance

## Baseline CNN Result

|Model|	Accuracy|	Macro F1-score|	Weighted F1-score|
|:---:|:---:|:---:|:---:|
|CNN from Scratch|	0.5601|	0.4912|	0.5506|

## Best Model Result
|Model	|Accuracy|	Macro F1-score|
|:---:|:---:|:---:|
|MobileNetV2|	0.9981|	0.9879|

MobileNetV2 achieved the best overall performance and was selected as the final deployment model.

## Project Structure
```
fish_image_classification/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebook/
│   ├── 01_data_preparation.ipynb
│   ├── 02_cnn_from_scratch.ipynb
│   ├── 03_transfer_learning_models.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_best_model_prediction.ipynb
│
├── models/
│   ├── cnn_from_scratch.h5
│   ├── efficientnetb0_finetuned.h5
│   ├── mobilenetv2_finetuned.h5
│   ├── resnet50_finetuned.h5
│   ├── vgg16_finetuned.h5
│   ├── inceptionv3_finetuned.h5
│   └── best_fish_classifier.h5
│
├── streamlit_app/
│   └── app.py
│
├── model_comparison_results.csv
├── sorted_model_comparison_results.csv
├── best_model_metadata.csv
├── requirements.txt
└── README.md
```

## Notebooks Description

```01_data_preparation.ipynb```
 - Dataset inspection
 - Class distribution analysis
 - Image augmentation
 - Data generator creation

```02_cnn_from_scratch.ipynb```
 - CNN architecture design
 - Model training
 - Evaluation using classification report and confusion matrix

```03_transfer_learning_models.ipynb```
 - Training pretrained models
 - Fine-tuning selected layers
 - Saving trained models
 - Collecting evaluation metrics

```04_model_comparison.ipynb```
 - Comparing all trained models
 - Selecting the best model using macro F1-score
 - Saving the best model as best_fish_classifier.h5

```05_best_model_prediction.ipynb```
 - Loading the best saved model
 - Predicting on sample fish images
 - Displaying predicted class and confidence score

## Streamlit Application
The Streamlit app allows users to:

 - Upload a fish image
 - View the uploaded image
 - Predict the fish species
 - Display prediction confidence
 - View the top 3 predicted classes

## Run the App
From the project root directory, run: 
```streamlit run streamlit_app/app.py```

## Install Dependencies

```pip install -r requirements.txt```

## Requirements

 - tensorflow
 - numpy
 - pandas
 - matplotlib
 - scikit-learn
 - streamlit
 - pillow

## Key Findings
 - The CNN trained from scratch achieved moderate performance but struggled with several classes.
 - Transfer learning significantly improved classification performance.
 - MobileNetV2 achieved the best balance of accuracy and class-wise performance.
 - Class imbalance affected the rare class animal fish bass, making macro F1-score an important evaluation metric.
 - Transfer learning proved more effective than training a CNN from scratch for this dataset.

## Final Model
The final deployment model is:
```MobileNetV2```.  Saved as:
```models/best_fish_classifier.h5```

The best model is selected automatically in the comparison notebook and stored under a generic filename so that the Streamlit app does not need to be changed if another model performs better in future experiments.