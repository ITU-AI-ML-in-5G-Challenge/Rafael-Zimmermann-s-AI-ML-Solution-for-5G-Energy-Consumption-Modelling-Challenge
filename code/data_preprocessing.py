import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from tabulate import tabulate
from feature_engineering import create_features

# Global variables
SUBSAMPLING = True


def train_adversarial_model(train, w_test, lista_features,
                            num_samples=25000, adjustment_factor=1000, max_iterations=100,
                            threshold_max=0.55, threshold_min=0.45, plot_features=True,
                            output_filename='train_data.csv'):
    """
    Trains an adversarial model for balancing between training and test datasets.

    Parameters:
    -----------
    train : DataFrame
        The training dataset.
    w_test : DataFrame
        The test dataset.
    lista_features : list
        List of features to be used for adversarial balancing.
    num_samples : int, optional
        Maximum number of samples to be used (default is 25000).
    adjustment_factor : int, optional
        Adjustment factor for the number of samples (default is 1000).
    max_iterations : int, optional
        Maximum number of iterations (default is 100).
    threshold_max : float, optional
        Upper AUC-ROC limit to stop balancing (default is 0.55).
    threshold_min : float, optional
        Lower AUC-ROC limit to stop balancing (default is 0.45).
    plot_features : bool, optional
        If True, will plot the feature distribution (default is True).
    output_filename : str, optional
        Filename where the balanced training data will be saved (default is 'train_data.csv').

    Returns:
    --------
    df_filtered : DataFrame
        The training DataFrame after the balancing process.

    Inner Workings:
    ---------------
    1. Data Preparation: Aggregates training and test datasets and prepares the features and labels.
    2. Model Initialization: Creates an instance of RandomForestClassifier with a fixed random seed.
    3. Optimization Loop: Runs a loop until the maximum number of iterations or until AUC-ROC limits are met.
        a. Model Validation: Performs cross-validation and calculates AUC-ROC.
        b. Limit Check: Checks whether AUC-ROC is within the defined limits.
        c. Sample Adjustment: Adjusts the number of samples based on the adjustment_factor.
        d. Sample Selection: Updates the training dataset based on predicted probabilities.
    4. Data Saving: Saves the DataFrame after the balancing process.
    5. Feature Plotting: If necessary, plots the feature distribution.
    """

    print(f'base_train train: {train.shape}')
    print(f'base_test w_test: {w_test.shape}')

    # Extract features from both training and test datasets.
    X_train = train[lista_features]
    X_test = w_test[lista_features]
    y_train = train['Energy']

    # Stack the training and test feature matrices vertically.
    X = np.vstack((X_train, X_test))
    # Create target array with zeros for training samples and ones for test samples.
    y = np.hstack((np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])))

    # Loop through iterations to train the model and adjust the samples.
    for iteration in range(max_iterations):
        clf = RandomForestClassifier(random_state=7)
        oof_probs_all = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
        score = roc_auc_score(y, oof_probs_all)
        print(f"Iteração {iteration + 1} - AUC-ROC: {score:.4f}")

        # Check if AUC-ROC score is within the desired range.
        if threshold_min <= score <= threshold_max:
            print("AUC-ROC is within desired range. Stopping...")
            print(X_train.shape)
            df_filtered = train.loc[X_train.index]
            df_filtered.to_csv(output_filename, index=False)
            if plot_features:
                plot_feature_distribution(train, X_train, X_test)
            return df_filtered
        elif score < threshold_min:  # If score is below the range, increase the number of samples
            num_samples += adjustment_factor
            print(X_train.shape)
            print(f"Score is below threshold_min. Increasing num_samples to {num_samples}")
        else:  # If score is above the range, decrease the number of samples
            num_samples -= adjustment_factor
            print(X_train.shape)
            print(f"Score is above threshold_max. Decreasing num_samples to {num_samples}")

        # Ensure num_samples does not go below half the size of w_test or exceed train length.
        num_samples = max(int(0.5 * w_test.shape[0]), min(train.shape[0], num_samples))
        print(f"New num_samples: {num_samples}")

        # Select the samples based on out-of-fold probabilities.
        train_length = X_train.shape[0]
        probs = oof_probs_all[:train_length]
        selected_idx = np.argsort(probs)[-num_samples:]
        X_train = X_train.iloc[selected_idx]
        y_train = y_train.iloc[selected_idx]

        # Update the complete feature and target arrays for the next iteration.
        X = np.vstack((X_train.values, X_test.values))
        y = np.hstack((np.zeros(X_train.shape[0]), np.ones(X_test.shape[0])))

        print(X_train.shape)
        print(X.shape)


def plot_feature_distribution(train, X_train, X_test):
    """Plot feature distributions for the original, subsampled training, and test datasets.

    Parameters:
    -----------
    train: DataFrame
        Original training dataset.
    X_train: DataFrame
        Subsampled training dataset.
    X_test: DataFrame
        Testing dataset.

    """
    # Loop through each feature column
    for feature in X_train.columns:
        plt.figure(figsize=(10, 6))

        try:
            # Plot KDE for features
            train[feature].plot(kind='kde', label='Original Train', legend=True)
            X_train[feature].plot(kind='kde', label='Subsampled Train', legend=True)
            X_test[feature].plot(kind='kde', label='Test', legend=True)
            plt.title(feature + " (KDE)")

        except:
            # If except plot histogram for features
            train[feature].hist(alpha=0.5, label='Original Train', density=True, bins=30)
            X_train[feature].hist(alpha=0.5, label='Subsampled Train', density=True, bins=30)
            X_test[feature].hist(alpha=0.5, label='Test', density=True, bins=30)
            plt.legend()
            plt.title(feature + " (Histogram)")

        plt.show()


def get_subsampled_dataframes(train, test, mask_w, w, plot_=True, subsampling=True):
    """
    Generates and returns subsampled dataframes based on the provided train and test data.

    Parameters:
    - train : DataFrame
        The training dataframe.
    - test : DataFrame
        The testing dataframe.
    - mask_w : Mask
        A mask or condition to filter the test dataframe.
    - w : int
        Value to distinguish the type of training (5 or 10).
    - plot_ : bool, optional
        If True, will plot the features (default is True).
    - subsampling : bool, optional
        If True, performs subsampling (default is True).

    Inner Workings:
    ----------------------
    1. Training Data Separation: Based on the value of 'w', it separates relevant features and adjusts parameters such as 'threshold_max', 'threshold_min', etc.
    2. Test Data Separation: Uses the 'mask_w' to filter the test data.
    3. Subsampling: If 'subsampling' is True, performs subsampling using the 'train_adversarial_model' function.
    4. Data Merging: Combines the subsampled dataframe with the original training dataframe.
    5. Printing Results: Prints information like the output filename and dimensions of the final dataframe.

    Returns:
    - Tuple: w_train_new_ dataframe after subsampling and merging.

    """

    # Separating training data
    if w == 5:
        w_train = train[
            (train['RUType_Type1'] == 1) | (train['RUType_Type5'] == 1) | (train['RUType_Type7'] == 1)].reset_index(
            drop=True)
        lista_features = ['load', 'Hour', 'Day']
        output_filename = '../data/train_data_w5.csv'
        threshold_max = 0.5
        threshold_min = 0.2
        num_samples = 2500
    elif w == 10:
        w_train = train[train['Day'] == 2].reset_index(drop=True)
        lista_features = ['load', 'ESMode1', 'ESMode6', 'Hour']
        output_filename = '../data/train_data_w10.csv'
        threshold_max = 0.72
        threshold_min = 0.2
        num_samples = 1500

    # Separating test data
    w_test = test[mask_w]

    if subsampling:
        # Performing subsampling for training data
        w_train_new = train_adversarial_model(w_train, w_test, lista_features, num_samples=num_samples,
                                              adjustment_factor=500, max_iterations=100,
                                              threshold_max=threshold_max, threshold_min=threshold_min,
                                              plot_features=plot_,
                                              output_filename=output_filename)
    else:
        w_train_new = w_train

    # Merging subsampled data with training data
    w_train_new_ = pd.merge(train, w_train_new[['Time', 'BS']], on=['Time', 'BS'], how='inner')

    # Printing results
    print(f'name: {output_filename}')
    print(w_train_new_)
    print(w_train_new_.shape)

    return w_train_new_


def generate_masks(df):
    """
    Generate masks for w1, w5, and w10 based on given conditions.

    Parameters:
    - df : pd.DataFrame
        Complete dataframe containing both training and test data.

    Internal Working:
    -----------------
    1. Data Splitting: Divides the input dataframe into training (where 'Energy' is not -1) and test (where 'Energy' is -1) data.
    2. Categorizing Columns: Converts the 'BS' and 'RUType' columns to category data types and creates encoded 'BS_cat' and 'RUType_cat' columns.
    3. Data Overview: Prints basic information about training and test data, and unique values in the 'BS_cat' and 'RUType_cat' columns.
    4. Identifying Valid Categories: Determines the 'BS_cat' values with more than 10 occurrences and 'RUType_cat' values with more than 50 occurrences in the training data.
    5. Generating Masks:
       a. w1 (objA) mask: Both 'BS_cat' and 'RUType_cat' from the test data are found in the valid categories from the training data.
       b. w5 (objB) mask: 'BS_cat' from the test data is not found, but 'RUType_cat' is found in the valid categories from the training data.
       c. w10 (objC) mask: Neither 'BS_cat' nor 'RUType_cat' from the test data are found in the valid categories from the training data.

    Returns:
    - tuple : Contains masks for w1, w5, and w10.
    """

    df['BS_cat'] = df['BS'].astype('category').cat.codes
    df['RUType_cat'] = df['RUType'].astype('category').cat.codes

    # Split data for train and test sets
    X_train = df[df['Energy'] != -1]
    X_test = df[df['Energy'] == -1]

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Unique values in BS_cat:", df['BS_cat'].nunique())
    print("Unique values in RUType_cat:", df['RUType_cat'].nunique())

    # Identify BS_cat and RUType_cat values in the training data that meet the occurrence conditions
    bs_train_valid = set(X_train['BS_cat'].value_counts()[X_train['BS_cat'].value_counts() > 10].index)
    rUType_train_valid = set(X_train['RUType_cat'].value_counts()[X_train['RUType_cat'].value_counts() > 50].index)

    print("BS_cat values with more than 10 occurrences:", len(bs_train_valid))
    print("RUType_cat values with more than 50 occurrences:", len(rUType_train_valid))

    # w1 mask
    mask_w1 = X_test['BS_cat'].isin(bs_train_valid) & X_test['RUType_cat'].isin(rUType_train_valid)

    # w5 (objB) mask
    mask_w5 = (~X_test['BS_cat'].isin(bs_train_valid)) & X_test['RUType_cat'].isin(rUType_train_valid)

    # w10 (objC) mask
    mask_w10 = (~X_test['BS_cat'].isin(bs_train_valid)) & ~X_test['RUType_cat'].isin(rUType_train_valid)

    print("mask_w1 True count:", mask_w1.sum())
    print("mask_w5 True count:", mask_w5.sum())
    print("mask_w10 True count:", mask_w10.sum())

    return mask_w1, mask_w5, mask_w10


def get_features_for_w1(data_leakage, use_pickle, read_feature_file,
                        fold_group_divide=None,
                        terms_to_check_in_cols_remove=None,
                        specific_cols_remove=None
                        ):
    """
    Generates dataframes with feature engineering, data separation between training and testing, and feature selection,
    all ready to be used for the validation and training model.

    Parameters:
    - data_leakage : bool
        Enables or disables the generation of aggressive features that may lead to data leakage.
    - use_pickle : bool
        If True, uses a previously saved pickle file to improve performance if no changes are made in this preprocessing step.
    - read_feature_file : bool
        Instead of executing feature creation, reads a CSV containing already created features to improve performance if possible.
    - fold_group_divide : list, optional
        List of columns to be used in fold splitting during training (default is ['RUType_cat', 'Energy_qcut', 'BS_cat']).
    - terms_to_check_in_cols_remove : list, optional
        List of terms to search for in column names that should be removed during feature selection (default is ['M1']).
    - specific_cols_remove : list, optional
        List of specific columns to be removed from the dataframe (default is ['Energy', 'Time', 'BS', 'Energy_qcut']).

    Inner Workings:
    ----------------------
    1. Feature Engineering: Based on the value of 'data_leakage', generates new features using the 'create_features' function.
    2. Training and Testing Data Separation: Uses the 'Energy' column to split the data into training and testing sets.
    3. Feature Selection: Removes columns based on 'terms_to_check' and 'specific_cols_to_remove', performing specific feature selection for the w1 model.
    4. Data Splitting: Uses the training and testing datasets to generate the variables x_train, x_test, y_train, y_divide, mask_w1 that will be applied in the validation model.
    5. Data Saving: Saves the sets x_train, x_test, y_train, y_divide into a pickle file for future use.

    Returns:
    - Tuple: containing x_train, x_test, y_train, y_divide
    """
    if specific_cols_remove is None:
        specific_cols_remove = ['Energy', 'Time', 'BS', 'Energy_qcut']
    if terms_to_check_in_cols_remove is None:
        terms_to_check_in_cols_remove = ['M1', 'poly']
    if fold_group_divide is None:
        fold_group_divide = ['RUType_cat', 'Energy_qcut', 'BS_cat']
    if use_pickle:
        with open('../data/data_w1.pkl', 'rb') as file:
            X_train, X_test, y_train, y_divide = pickle.load(file)
        return X_train, X_test, y_train, y_divide

    # Feature Engineering
    df = create_features('', data_leakage=data_leakage, use_arq=read_feature_file)

    # Split data for train and test sets
    x_train_w = df[df['Energy'] != -1]
    x_test_w = df[df['Energy'] == -1]

    # Perform feature selection specifically for the w1 model
    cols_to_remove = [
        col for col in x_train_w.columns
        if any(term in col for term in terms_to_check_in_cols_remove) or col in specific_cols_remove
    ]

    # Data division for multi-label K-fold validation model application
    y_train = x_train_w['Energy']
    y_divide = x_train_w[fold_group_divide]
    X_train = x_train_w.drop(columns=cols_to_remove)
    X_test = x_test_w.drop(columns=cols_to_remove)

    with open('../data/data_w1.pkl', 'wb') as file:
        pickle.dump((X_train, X_test, y_train, y_divide), file)

    print(tabulate(X_train.head(), headers='keys', tablefmt='psql'))
    print(X_train.shape)

    return X_train, X_test, y_train, y_divide


def get_features_for_w5(data_leakage, use_pickle, read_feature_file, mask,
                        fold_group_divide=None,
                        terms_to_check_in_cols_remove=None,
                        specific_cols_remove=None,
                        ):
    """
    Generates dataframes with feature engineering, robust variable creation, subsampling, data separation between training and testing, and feature selection,
    all ready to be used for the validation and training model.

    Parameters:
    - data_leakage : bool
        Enables or disables the generation of aggressive features that may lead to data leakage.
    - use_pickle : bool
        If True, uses a previously saved pickle file to improve performance if no changes are made in this preprocessing step.
    - read_feature_file : bool
        Instead of executing feature creation, reads a CSV containing already created features to improve performance if possible.
    - mask : bool
        mask for test w5 objective B
    - fold_group_divide : list, optional
        List of columns to be used in fold splitting during training (default is ['RUType_cat', 'Energy_qcut', 'BS_cat']).
    - terms_to_check_in_cols_remove : list, optional
        List of terms to search for in column names that should be removed during feature selection (default is ['Energy', 'BS', 'M2']).
    - specific_cols_remove : list, optional
        List of specific columns to be removed from the dataframe (default is ['Time']).

    Inner Workings:
    ----------------------
    1. Feature Engineering: Based on the value of 'data_leakage', generates new features using the 'create_features' function.
    2. Robust Variable Creation: Utilizes the 'generate_robust_features' function to create robust polynomial features.
    3. Subsampling: Uses the 'get_subsampled_dataframes' function to rebalance the training set based on the testing set.
    4. Training and Testing Data Separation: Uses the 'Energy' column to split the data into training and testing sets.
    5. Feature Selection: Removes columns based on 'terms_to_check' and 'specific_cols_to_remove', performing specific feature selection for the w5 model.
    6. Data Splitting: Uses the training and testing datasets to generate the variables x_train, x_test, y_train, y_divide, mask_w5 that will be applied in the validation model.
    7. Data Saving: Saves the sets x_train, x_test, y_train, y_divide into a pickle file for future use.

    Returns:
    - Tuple: containing x_train, x_test, y_train, y_divide.
    """
    if specific_cols_remove is None:
        specific_cols_remove = ['Time']
    if terms_to_check_in_cols_remove is None:
        terms_to_check_in_cols_remove = ['Energy', 'BS', 'M2']
    if fold_group_divide is None:
        fold_group_divide = ['RUType_cat', 'Energy_qcut', 'BS_cat']
    if use_pickle:
        with open('../data/data_w5.pkl', 'rb') as f:
            X_train, X_test, y_train, y_divide = pickle.load(f)
        return X_train, X_test, y_train, y_divide

    # Feature Engineering
    df_ = create_features('', data_leakage=data_leakage, use_arq=read_feature_file)

    # Split data for train and test sets
    X_train = df_[df_['Energy'] != -1]
    X_test = df_[df_['Energy'] == -1]

    # Mask that divides the w5 data in the test set
    mask_w5 = (X_test['w'] == 5) & (~(X_test['RUType_Type11'] | X_test['RUType_Type12']))

    # Perform subsampling to rebalance the training data to be more similar to the test data and increase generalization power
    X_train = get_subsampled_dataframes(X_train, X_test, mask, w=5, plot_=True, subsampling=SUBSAMPLING)

    # Create the variables that will be used by the validation model y and y_divide
    y_train = X_train['Energy']
    y_divide = X_train[fold_group_divide]

    # Perform feature selection specifically for the w5 model
    cols_to_remove = [
        col for col in X_train.columns
        if any(term in col for term in terms_to_check_in_cols_remove) or col in specific_cols_remove
    ]
    X_train = X_train.drop(columns=cols_to_remove)
    X_test = X_test.drop(columns=cols_to_remove)

    # Save in a pickle file the variables that will be used by the validation and training model for w5
    with open('../data/data_w5.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_divide), f)

    print(tabulate(X_train.head(), headers='keys', tablefmt='psql'))
    print(X_train.shape)

    return X_train, X_test, y_train, y_divide


def get_features_for_w10(data_leakage, use_pickle, read_feature_file, mask,
                         fold_group_divide=None,
                         terms_to_check_in_cols_remove=None,
                         specific_cols_remove=None
                         ):
    """
    Generates dataframes with feature engineering, robust variable creation, subsampling, data separation between training and testing, and feature selection,
    all ready to be used for the validation and training model for w10.

    Parameters:
    - data_leakage : bool
        Enables or disables the generation of aggressive features that may lead to data leakage.
    - use_pickle : bool
        If True, uses a previously saved pickle file to improve performance if no changes are made in this preprocessing step.
    - read_feature_file : bool
        Instead of executing feature creation, reads a CSV containing already created features to improve performance if possible.
    - mask : bool
        mask for test w10 objective C
    - fold_group_divide : list, optional
        List of columns to be used in fold splitting during training (default is ['RUType_cat', 'Energy_qcut', 'BS_cat']).
    - terms_to_check_in_cols_remove : list, optional
        List of terms to search for in column names that should be removed during feature selection (default is ['Energy', 'BS', 'RUT', 'M3', 'cluster', 'lag']).
    - specific_cols_remove : list, optional
        List of specific columns to be removed from the dataframe (default is ['Time', 'Weekday', 'ESMode5', 'ESMode4', 'ESMode3', 'Weekend', 'Month', 'Year', 'w', 'Day', 'Hour']).

    Inner Workings:
    ----------------------
    1. Feature Engineering: Based on the value of 'data_leakage', generates new features using the 'create_features' function.
    2. Data Separation: Uses the 'Energy' column to split the data into training and testing sets.
    3. Mask Generation: Creates a mask for the w10 data in the test set based on certain conditions.
    4. Subsampling: Uses the 'get_subsampled_dataframes' function to rebalance the training set based on the testing set.
    5. Variables for Validation: Creates the variables y_train and y_divide that will be used in the validation model.
    6. Feature Selection: Removes columns based on 'terms_to_check' and 'specific_cols_to_remove', performing specific feature selection for the w10 model.
    7. Data Saving: Saves the sets X_train, X_test, y_train, y_divide, mask_w10 into a pickle file for future use.

    Returns:
    - Tuple: containing X_train, X_test, y_train, y_divide.
    """
    if specific_cols_remove is None:
        specific_cols_remove = ['Time', 'Weekday', 'ESMode5', 'ESMode4', 'ESMode3', 'Weekend', 'Month',
                                'Year', 'w', 'Day', 'Hour']
    if terms_to_check_in_cols_remove is None:
        terms_to_check_in_cols_remove = ['Energy', 'BS', 'RUT', 'M3', 'cluster', 'lag', 'lead', 'poly']
    if fold_group_divide is None:
        fold_group_divide = ['RUType_cat', 'Energy_qcut', 'BS_cat']
    if use_pickle:
        with open('../data/data_w10.pkl', 'rb') as f:
            X_train, X_test, y_train, y_divide = pickle.load(f)
        return X_train, X_test, y_train, y_divide

    # Feature Engineering
    df_ = create_features('', data_leakage=data_leakage, use_arq=read_feature_file)

    # Split data for train and test sets
    X_train = df_[df_['Energy'] != -1]
    X_test = df_[df_['Energy'] == -1]

    # Perform subsampling to rebalance the training data to be more similar to the test data and increase generalization power
    X_train = get_subsampled_dataframes(X_train, X_test, mask, w=10, plot_=True, subsampling=SUBSAMPLING)

    # Create the variables that will be used by the validation model y and y_divide
    y_train = X_train['Energy']
    y_divide = X_train[fold_group_divide]

    # Perform feature selection specifically for the w10 model
    cols_to_remove = [
        col for col in X_train.columns
        if any(term in col for term in terms_to_check_in_cols_remove) or col in specific_cols_remove
    ]
    X_train = X_train.drop(columns=cols_to_remove)
    X_test = X_test.drop(columns=cols_to_remove)

    # Save in a pickle file the variables that will be used by the validation and training model for w10
    with open('../data/data_w10.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_divide), f)

    print(tabulate(X_train.head(), headers='keys', tablefmt='psql'))
    print(X_train.shape)

    return X_train, X_test, y_train, y_divide
