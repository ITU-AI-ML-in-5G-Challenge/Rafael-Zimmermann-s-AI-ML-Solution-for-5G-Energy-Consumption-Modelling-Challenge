# Rafael-Zimmermann-s-AI-ML-Solution-for-5G-Energy-Consumption-Modelling-Challenge
Managed by Rafael Zimmermann, this repo contains the solution for the ITU AI/ML in 5G Energy Consumption Modelling Challenge.

## Requirements
#### Python Version
Python 3.10.12 is necessary for this project.

#### Libraries
Install the libraries from requirements.txt:
pip install -r requirements.txt

## Introduction
This repo presents a comprehensive solution that takes into account three key objectives, each affecting the design and methodology of our modeling approach.
1. Objective A: Time-series forecasting methods were most effective for estimating energy consumption in specific base station products.
1. Objective B: For generalized forecasting across different but similar base stations, a hybrid model combining elements of time-series analysis and complex methods yielded the best results.
1. Objective C: Simplicity reigns supreme when generalizing across significantly different base station configurations. A simpler model ensured better performance and avoided overfitting.

## Data Segmentation
Data was segmented specifically for each objective, based on features like BS_cat and RUType_cat. Masks were used to filter the test data accordingly.

## Subsampling
Adversarial Validation was used for subsampling, notably for Objectives B and C, to align the training data distribution more closely with the test data.

## Modeling Workflow
### Common Steps for all models:
1. Data Cleaning
1. Feature Engineering
1. Ensemble Modeling: Ridge Regression + XGBoost
1. Training and Validation: MultiLabelStratifiedKFold with 10 folds

### Modeling Strategy
The ensemble model merges Ridge Regression for handling linear trends with XGBoost to address non-linear patterns.

### Prediction Phase
Ridge Regression provides the initial predictions, which are adjusted using XGBoost on the residuals, summing these up for the final estimates.

Best,
Rafael Zimmermann
