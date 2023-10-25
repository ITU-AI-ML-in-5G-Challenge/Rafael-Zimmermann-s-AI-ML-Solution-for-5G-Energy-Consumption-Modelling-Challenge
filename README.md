# Rafael-Zimmermann-s-AI-ML-Solution-for-5G-Energy-Consumption-Modelling-Challenge
Managed by Rafael Zimmermann, this repo contains the solution for the ITU AI/ML in 5G Energy Consumption Modelling Challenge. It employs meticulous data preprocessing, advanced feature engineering, and ensemble modeling for accurate energy forecasts across different base station scenarios

## Requirements

### Python Version
- Python version required: 3.10.12

### Libraries
The required libraries are listed in the `requirements.txt` file and can be installed using pip:
pip install -r requirements.txt

## Introduction:
In this competition, I confronted three main objectives that influenced both the design and the methodology. A holistic approach was adopted, incorporating meticulous pre-processing, advanced feature engineering, and a distinct modeling strategy to suit the outlined objectives.

## Objectives:

**Objective A: Estimating Energy Consumption in Specific Base Station Products:** For this objective, it's understood that specific base information, their configurations, and a historical usage, typically spanning a complete week, were received. In this scenario, the training data, and primarily the history of energy expenditure, were used to determine the best forecasting patterns. It was found that treating this as a time-series problem yielded the best results. The model uses energy and load along with the seasonality of each base's use to pinpoint the most accurate energy estimates.

**Objective B: Generalization Across Different Base Station Products:** In this scenario, I understood that having a history of bases with similar configurations enables using that history to forecast a new base that uses similar configurations. Here, we face a different problem because the user usage history and its hourly seasonality become irrelevant, as does the energy history. It's still possible to treat it as a time-series problem by using lags and other such features. However, what showed the best result was the use of complex features concerning the new bases' configurations. This shows that there still exists a way to use historical data to forecast unknown futures and that they still have a certain strong correlation. The issue transforms into a hybrid problem, combining elements of time-series analysis and complex methods to identify patterns in new base configurations for precise energy prediction.

**ObjC - Generalization across different base station configurations:** In this scenario, it's understood that besides new databases, the model should also generalize to new base configurations, often with significant differences, making it practically impossible to use other bases' history for forecasting energy consumption. After several optimizations and tests, it was understood that there wasn't a possibility to create a complex model and that the history of data did not hold much relevance. Characteristics such as load, EMS Modes 1, 2, and 6 gained importance. It was noticed that the simpler the model, the better it performed and the more general it became. Therefore, all robust features and all time-series features were removed, generating greater generalization power and avoiding overfitting.

## Data Segmentation:

In our journey to develop models that catered to objectives A, B, and C, we recognized the need to segment our data specifically for each scenario. This segmentation ensured that each data set aligned with the specific goal at hand, making sure the dataset's features matched the model's requirements.

## Segmentation Technique:

To segment the test data, we examined two main features: BS_cat (which represents station data) and RUType_cat (which denotes the types of configurations and products used). We established unique sets of these features from the training data and based on them, crafted masks to segment the test data.
    # (Minimal example)
    # w1 mask for Objective A
    mask_w1 = X_test['BS_cat'].isin(bs_train_unique) & X_test['RUType_cat'].isin(rUType_train_unique)

    # w5 mask for Objective B
    mask_w5 = (~X_test['BS_cat'].isin(bs_train_unique)) & X_test['RUType_cat'].isin(rUType_train_unique)

    # w10 mask for Objective C
    mask_w10 = (~X_test['BS_cat'].isin(bs_train_unique)) & ~X_test['RUType_cat'].isin(rUType_train_unique)

## Subsampling
A crucial step was the implementation of subsampling through Adversarial Validation, with a special focus on Objectives B and C; Objective A already had a distribution very similar to the test data. The central idea was to adjust the training data distribution to more closely align with what is observed in the test data for these specific objectives.

W5 subsampling - Objective B
> ![](https://drive.google.com/uc?export=view&id=1TGjEERTl4jdwTdaEHRj35Ai7NsdK2dtW)



W10 subsampling - Objective C
> ![](https://drive.google.com/uc?export=view&id=1fB6IcTZa6J_ucX6Vg6B1a-8Pk_Ldb63-)

## Modeling Workflow:

With the data segmented, the modeling approach for each objective became more structured:
#### Common Steps for All Models:
1. **Data Cleaning**: Ensuring the quality of the data and merge all the data into a single table.
1. **Feature Engineering**: Creating new features based on existing ones to provide more insights or transforming features to better suit the modeling techniques.
1. **Modeling**: Using an ensemble approach that combines Ridge Regression with XGBoost.
1. **Training and Validation:** Employed MultiLabelStratifiedKFold with 10 folds for robust and balanced validation.

#### Specific Workflows:
1. ObjA (w1):
>* Preprocessing (Feature Selection)
>* Prediction for w1 data test
1. ObjB (w5):
>* Preprocessing (Feature Selection, Subsampling)
>* Prediction for w5 data test
1. ObjC (w10):
>* Preprocessing(Feature Selection, Subsampling)
>* Prediction for w10 data test

![](https://drive.google.com/uc?export=view&id=1qFGGuRm_XhuiwhAgFngKXZ-vC2UZy3jv)

By following these dedicated workflows for each objective, we aim to create models that are both accurate and adaptable, addressing the unique challenges of each data segment.

## Modeling Strategy - Ensemble Model:

The core of our solution lies in the ensemble model that merges Ridge Regression with XGBoost. Here's how it operates:
1. **Ridge Regression**: This acts as our foundational model, skilled at capturing linear trends within the data. Depending on the mask or objective, it's trained on different datasets tailored to each specific scenario.
1. **XGBoost on Residuals**: After obtaining predictions from Ridge, we compute the residuals (the difference between predictions and actual values) and train XGBoost on these residuals. This enables XGBoost to grasp non-linear nuances and tendencies.

![](https://drive.google.com/uc?export=view&id=1ezEkXZR94gXS_HFesyKQIdO4Dk_MQDki)

## Prediction Phase:
In the prediction stage, we initiate with predictions from Ridge Regression and subsequently adjust those predictions with those made by XGBoost on the residuals. The sum of these predictions renders our final estimate.

## About Data Leakage

The main approach used to ensure a top-10 placement was to initially accept the data leakage, and then validate with a code that had no data leakage at all. This was done because it was clear that the top-ranked participants had scores far lower than what would be possible without data leakage. Therefore, I chose this approach to genuinely track my progress and understand my standing in the competition.

My final submission was completely free of data leakage. The notebook can execute the three main models created for the competition:

1. Data Leakage + Subsampling
>* LB public = 0.050252038
>* LB private = 0.050298939
1. No Data Leakage + using Subsampling (final submission 1)
>* LB public = 0.069020300
>* LB private = 0.069932182
1. No Data Leakage + without Subsampling (final submission 2)
>* LB public = 0.104512931
>* LB private = 0.105793493

I would like to emphasize that in the feature engineering section, I discuss how data leakage is done. In this dataset, from what I have identified, there are three main ways to leak data:
1. Directly and willingly visualizing future data (Lead features, BFIL null values, etc...)
>* The only way to prevent data contamination is by not using these methods.
1. Viewing all data at once, thereby including future data, for example through aggregation methods like target encoding.
>* It's possible to avoid contaminating the data by only filling the aggregation with past data, without using future data. There is a specific method for this.
1. Finding patterns that are clearly data leaks, such as quirks in the data that lead to outcomes, but which will not be available in the real data.
>* Usually generated by errors in data creation, separation, or normalization.

