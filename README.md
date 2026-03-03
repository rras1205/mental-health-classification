# mental-health-classification
Binary Detection of Depressive vs Non-Depressive Text

## Project Overview
This repository contains a natural language processing (NLP) project that uses TF-IDF, Decision Trees, and Neural Networks to classify user statements into 'Non-Depressed' or 'Depressed' mental health categories. 

The goal is to explore NLP-based binary classification using classical machine learning models and a neural network, and to compare their performance on TF-IDF vectorized text data.

The project includes:

* Data preprocessing
* Exploratory data analysis (EDA)
* TF-IDF feature extraction
* Logistic Regression (baseline)
* Decision Tree (with hyperparameter tuning)
* Artificial Neural Network (ANN)
* Performance comparison and evaluation

## Problem Framing
Although the original dataset contains multiple psychological categories (e.g., anxiety, stress, bipolar disorder), this project focuses specifically on binary depression detection.

The “Non-Depressive” label represents the absence of depressive indicators within the dataset.
It does not imply the individual is free from other psychological conditions.

This binary framing simplifies interpretability and aligns with real-world screening use cases where the goal is risk detection rather than full diagnostic classification.

## Dataset
The dataset contains text statements labeled with mental health categories.

For this project:
* Only Depression and Non-Depression were used.
* The task is treated as a supervised binary classification problem.
