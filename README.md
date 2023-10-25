# Rafael-Zimmermann-s-AI-ML-Solution-for-5G-Energy-Consumption-Modelling-Challenge
Managed by Rafael Zimmermann, this repo contains the solution for the ITU AI/ML in 5G Energy Consumption Modelling Challenge. It employs meticulous data preprocessing, advanced feature engineering, and ensemble modeling for accurate energy forecasts across different base station scenarios

# Introduction:
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
