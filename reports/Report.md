# Multiclass Fish Image Classification
## Project Report

## 1.Introduction

Image classification is an important application of deep learning in which images are assigned to predefined categories based on their visual features. In this project, a multiclass fish image classification system was developed to identify different fish species from images.

The main objective of the project was to compare a Convolutional Neural Network trained from scratch with multiple pretrained transfer learning models and identify the best-performing model for fish species classification. The final selected model was then prepared for deployment through a Streamlit web application that allows users to upload fish images and receive real-time predictions.

## 2.Problem Statement

The goal of this project is to classify fish images into multiple species categories using deep learning models.

The project includes:

 - Training a CNN model from scratch
 - Training and fine-tuning pretrained deep learning models
 - Comparing the performance of all models using evaluation metrics
 - Selecting and saving the best-performing model
 - Building a Streamlit application for end-user prediction

## 3.Dataset Description

The dataset contains fish images grouped into species-specific folders. It is organized into three subsets:

|Dataset Split|	Number of Images|
|:---:|:---:|
|Training Set|	6,225|
|Validation Set|	1,092|
|Test Set|	3,187|

The dataset contains 11 classes:

 - animal fish
 - animal fish bass
 - fish sea_food black_sea_sprat
 - fish sea_food gilt_head_bream
 - fish sea_food hourse_mackerel
 - fish sea_food red_mullet
 - fish sea_food red_sea_bream
 - fish sea_food sea_bass
 - fish sea_food shrimp
 - fish sea_food striped_red_mullet
 - fish sea_food trout

### Class Imbalance Observation

The dataset was found to be imbalanced. For example:

 - animal fish had 1,096 training images
 - animal fish bass had only 30 training images

Because of this imbalance, overall accuracy alone was not sufficient to judge model performance. Therefore, precision, recall, F1-score, and especially macro F1-score were used during evaluation.

## 4.Tools and Technologies Used

 - Python
 - TensorFlow / Keras
 - NumPy
 - Pandas
 - Matplotlib
 - Scikit-learn
 - Streamlit
 - VS Code Python Notebooks

## 5.Data Preprocessing and Augmentation

The following preprocessing steps were applied before model training:

 - Images were resized to 224 x 224
 - Data augmentation was applied to training images:
     - rotation
    - zoom
     - horizontal flipping
     - width shifting
     - height shifting
 - Validation and test sets were kept unaugmented for fair evaluation
 - Class weights were used during training to reduce the effect of class imbalance

Different pretrained models required different preprocessing functions. Appropriate preprocessing was applied according to the architecture used, such as MobileNetV2 preprocessing, ResNet preprocessing, VGG16 preprocessing, and InceptionV3 preprocessing.

## 6.Models Used

The following models were trained and evaluated:

### 6.1 CNN from Scratch

A custom CNN model was built using:

 - Convolutional layers
 - Batch normalization
 - Max pooling layers
 - Dense layers
 - Dropout layers

This model served as the baseline for comparison.

### 6.2 Transfer Learning Models

Five pretrained architectures were trained and fine-tuned:

 - EfficientNetB0
 - MobileNetV2
 - ResNet50
 - VGG16
 - InceptionV3

For each pretrained model:

 - The base layers were initially frozen
 - A custom classification head was trained
 - The top layers were later unfrozen and fine-tuned using a low learning rate

## 7.Evaluation Metrics

The following evaluation metrics were used:

 - Accuracy
 - Precision
 - Recall
 - F1-score
 - Confusion matrix

### Why Macro F1-score Was Important:
Since the dataset was imbalanced, macro F1-score was used as the primary model selection metric. Unlike weighted averages, macro F1-score gives equal importance to every class, including minority classes such as animal fish bass.

## 8.Baseline CNN Performance
The CNN trained from scratch produced the following results:

|Metric|	Score|
|:---:|:---:|
|Test Accuracy|	0.5601|
|Macro Precision|	0.5475|
|Macro Recall	|0.5424|
|Macro F1-score	|0.4912|
|Weighted F1-score	|0.5506|

The CNN performed moderately on some classes, but struggled with several visually similar or underrepresented classes. This showed the limitation of training from scratch on the given dataset.

## 9.Transfer Learning Results

The transfer learning models achieved significantly better performance than the CNN baseline.

## Best Model

The best-performing model was:

|Model|	Accuracy|	Macro F1-score|
|:---:|:---:|:---:|
|MobileNetV2|	0.9981|	0.9879|

MobileNetV2 achieved the highest overall performance and was selected as the final deployment model.

### EfficientNetB0 Example Result

EfficientNetB0 also performed very strongly:

|Metric|	Score|
|:---:|:---:|
|Test Accuracy|	0.9925|
|Macro Precision|	0.9450|
|Macro Recall	|0.9822|
|Macro F1-score	|0.9572|
|Weighted F1-score|	0.9933|

This comparison demonstrated that transfer learning was much more effective than the CNN trained from scratch.

## 10.Model Selection

MobileNetV2 was selected as the final model because it achieved:

 - the highest test accuracy
 - the highest macro F1-score
 - strong and balanced performance across nearly all classes
 - better generalization than the scratch CNN and other transfer learning models

The selected model was saved under a generic filename: ```best_fish_classifier.h5```

This design allows future experiments to replace the best model without requiring changes to the Streamlit application code.

## 11.Prediction Workflow
The final prediction pipeline performs the following steps:

 - Load the saved best model
 - Read an input image
 - Resize the image to 224 x 224
 - Apply the preprocessing required by the selected model
 - Generate prediction probabilities Display:
     - predicted fish class
     - confidence score
     - top 3 predicted classes

## 12.Streamlit Deployment
A Streamlit web application was developed to provide a user-friendly interface for prediction.

 - Features of the Application
 - Upload fish images in JPG, JPEG, or PNG format
 - Display uploaded image
 - Predict fish species
 - Show confidence score
 - Display top 3 predicted classes
 - Automatically use the current best model metadata for preprocessing and prediction

The application can be launched using:

```streamlit run streamlit_app/app.py```

## 13.Project Deliverables
The completed project includes:

 - Trained CNN model
 - Five trained transfer learning models
 - Best saved deployment model
 - Jupyter notebooks for preprocessing, training, evaluation, and prediction
 - Model comparison CSV files
 - Streamlit web application
 - GitHub-ready README file
 - Project report

## 14.Key Findings
 - The CNN trained from scratch achieved limited performance compared to pretrained models.
 - Transfer learning greatly improved classification accuracy and F1-score.
 - MobileNetV2 produced the best overall result with:
     - accuracy: 0.9981
     - macro F1-score: 0.9879
 - Class imbalance was present in the dataset, especially for animal fish bass
 - Macro F1-score was more informative than accuracy alone for final model selection
 - A reusable model-selection design was implemented so the app can load the current best model without hard-coding a specific architecture

## 15.Conclusion

This project successfully developed a multiclass fish image classification system using deep learning. Multiple models were trained and compared, including a custom CNN and five pretrained architectures. The results showed that transfer learning significantly outperformed a CNN trained from scratch.

Among all trained models, MobileNetV2 achieved the best overall performance and was chosen for deployment. The final Streamlit application provides an interactive interface for real-time fish species prediction from uploaded images.

The project demonstrates the effectiveness of transfer learning for image classification tasks, especially when the dataset contains class imbalance and visually similar categories.