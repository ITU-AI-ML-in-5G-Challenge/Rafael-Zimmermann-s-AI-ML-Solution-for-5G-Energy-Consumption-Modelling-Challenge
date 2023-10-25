import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


def cumulative_target_encoding_by_time(df, col, target_col, time_unit='Hour'):
    """
    Applies target encoding using cumulative average by hour or day to avoid look-ahead bias in time series.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    col : str
        The column to be encoded.
    target_col : str
        The target column used for encoding.
    time_unit : str
        The unit of time, can be 'Hour' or 'Day'.

    Returns:
    --------
    pd.Series:
        Series after cumulative target encoding by time.

    Usage Example:
    --------------
    data['BS_hourly_target_cum'] = cumulative_target_encoding_by_time(data, 'BS', 'Energy', 'Hour')

    Inner Workings:
    ---------------
    1. Time Unit Verification: Checks the time unit and creates the corresponding column.
    2. Data Sorting: Sorts the DataFrame based on the 'Time' column (important for ensuring temporal order).
    3. Cumulative Sums: Calculates cumulative sums by hour or day (excluding the current value in the average to avoid leakage).
    4. Cumulative Counts: Counts the occurrences of each unique value in the specified column, broken down by time_unit.
    5. Division by Zero: Avoids division by zero by replacing counts of zero with one.
    6. Cumulative Average: Calculates the cumulative average by time and assigns it to a new column.
    """

    # Checking the time unit and creating the corresponding column
    if time_unit == 'Hour':
        df[time_unit] = pd.to_datetime(df['Time']).dt.hour
    elif time_unit == 'Day':
        df[time_unit] = pd.to_datetime(df['Time']).dt.day
    else:
        raise ValueError("time_unit must be 'Hour' or 'Day'")

    # Sorting the DataFrame based on the 'Time' column for ensuring temporal order
    df = df.sort_values('Time')

    # Calculating cumulative averages by hour or day (excluding the current value in the average to avoid leakage)
    df['CumSum'] = df.groupby([col, time_unit])[target_col].cumsum() - df[target_col]
    df['CumCount'] = df.groupby([col, time_unit]).cumcount()

    # Avoiding division by zero
    df['CumCount'] = df['CumCount'].replace(0, 1)

    # Calculating the cumulative average
    df[f'{col}_{time_unit.lower()}_target_cum'] = df['CumSum'] / df['CumCount']

    # Filling missing values in the cumulative average column with 0
    df[f'{col}_{time_unit.lower()}_target_cum'].fillna(0, inplace=True)

    return df[f'{col}_{time_unit.lower()}_target_cum']


def cumulative_target_encoding(df, col, target_col):
    """
    Applies target encoding using cumulative average to avoid look-ahead bias in time series.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the original data.
    col : str
        The column to be encoded.
    target_col : str
        The target column used for encoding.

    Returns:
    --------
    pd.Series:
        Series after cumulative target encoding.

    Usage Example:
    --------------
    data['BS_target_cum'] = cumulative_target_encoding(data, 'BS', 'Energy')

    Inner Workings:
    ---------------
    1. Data Copy: Creates a copy of the data so as not to alter the original DataFrame.
    2. Data Sorting: Sorts the DataFrame based on the 'Time' column (important for ensuring temporal order).
    3. Cumulative Sums: Calculates cumulative sums (excluding the current value in the average to avoid leakage).
    4. Cumulative Counts: Counts the occurrences of each unique value in the specified column.
    5. Division by Zero: Avoids division by zero by replacing counts of zero with one.
    6. Cumulative Average: Calculates the cumulative average and assigns it to a new column.
    """

    # Creating a copy of the data to not alter the original DataFrame
    temp_df = df.copy()

    # Sorting the DataFrame based on the 'Time' column (important for ensuring temporal order)
    temp_df = temp_df.sort_values('Time')

    # Calculating cumulative sums (excluding the current value in the average to avoid leakage)
    cumsum = temp_df.groupby(col)[target_col].cumsum() - temp_df[target_col]
    cumcount = temp_df.groupby(col).cumcount()

    # Avoiding division by zero
    cumcount = cumcount.replace(0, 1)

    # Calculating the cumulative average
    temp_df[f'{col}_Energy_mean'] = cumsum / cumcount

    return temp_df[f'{col}_Energy_mean']


class Grouper(BaseEstimator, TransformerMixin):
    """
    A class to perform feature grouping and transformation.

    Parameters:
    -----------
    method : str
        The clustering method to apply (default is 'kmeans').
    n_components : int
        The number of clusters or components (default is 2).
    cols_transform : list
        The columns to be transformed (default is None).
    drop_original : bool
        Whether to drop the original columns after transformation (default is True).
    normalize : str
        The normalization method to apply (default is None) (Choose from: 'standard', 'minmax').

    Attributes:
    -----------
    model : object
        The clustering model.
    scaler : object
        The scaler object if normalization is applied.

    Inner Workings:
    ---------------
    1. Model Selection: Initializes the clustering model based on the specified 'method'.
    2. Scaler Initialization: Initializes the scaling method if 'normalize' is specified.
    3. Data Fitting: Fits the model on the DataFrame.
    4. Data Transformation: Transforms the DataFrame based on the fitted model.
    5. Column Management: Manages the addition and removal of columns in the DataFrame.
    """

    def __init__(self, method='kmeans', n_components=2, cols_transform=None, drop_original=True, normalize=None):
        self.method = method
        self.n_components = n_components
        self.cols_transform = cols_transform
        self.drop_original = drop_original
        self.normalize = normalize

        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_components, random_state=7, n_init=10)

        if self.normalize:
            if self.normalize == 'standard':
                self.scaler = StandardScaler()
            elif self.normalize == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Normalization method not recognized. Choose from: 'standard', 'minmax'")

    def fit(self, df, y=None):
        if self.normalize and self.cols_transform:
            self.scaler.fit(df[self.cols_transform])
            scaled_data = self.scaler.transform(df[self.cols_transform])
            self.model.fit(scaled_data)
        elif self.normalize:
            self.scaler.fit(df)
            scaled_data = self.scaler.transform(df)
            self.model.fit(scaled_data)
        elif self.cols_transform:
            self.model.fit(df[self.cols_transform])
        else:
            self.model.fit(df)
        return self

    def transform(self, df, y=None):
        df_ = df.copy()
        if self.normalize and self.cols_transform:
            scaled_data = self.scaler.transform(df[self.cols_transform])
            labels = self.model.fit_predict(scaled_data)
        elif self.normalize:
            scaled_data = self.scaler.transform(df)
            labels = self.model.fit_predict(scaled_data)
        elif self.cols_transform:
            labels = self.model.fit_predict(df[self.cols_transform])
        else:
            labels = self.model.fit_predict(df)

        name = f'cluster_{self.method}_{self.n_components}'
        df_[name] = labels

        if self.drop_original:
            if self.cols_transform:
                df_ = df_.drop(columns=self.cols_transform)
            else:
                df_ = df_[[name]]

        return df_


def compute_target_encoding_without_current(group, coluna, target_column='Energy_null'):
    """
    Compute target encoding for a given column without considering the current row.

    Parameters:
    -----------
    group : DataFrame
        A Pandas DataFrame containing the group of rows for which to compute target encoding.
    coluna : str
        The column name to target for encoding.
    target_column : str, optional
        The target column for which the encoding is computed. Defaults to 'Energy_null'.

    Returns:
    --------
    encoding : Series
        A Pandas Series containing the target encoding for the specified column.

    Inner Workings:
    ---------------
    1. Sum Calculation: Calculates the sum of values for each unique value in the given column, based on the target column.
    2. Count Calculation: Determines the count of each unique value in the given column.
    3. Encoding Calculation: Computes the encoding using these sums and counts, excluding the current row.
    4. Exception Handling: Catches any value errors and returns NaN in such cases.
    """
    # Calculate the sum and counts for each unique value within the group
    hourly_sums = group.groupby(coluna)[target_column].sum()
    hourly_counts = group.groupby(coluna).size()

    # Calculate the encoding without considering the current row's value
    total_sum = group[coluna].map(hourly_sums)
    total_count = group[coluna].map(hourly_counts)

    try:
        encoding = (total_sum - group[target_column]) / (total_count - 1)
    except Exception as e:
        print(f"Value error occurred with column: {coluna}")
        print(e)
        print(group[coluna])
        encoding = np.nan

    return encoding


def generate_robust_features(dataframe, list_columns=['load', 'ESMode6', 'Antennas', 'TXpower', 'Frequency'], degree=2):
    """
    Generate robust polynomial features for selected columns in the DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        A Pandas DataFrame containing the original data.
    list_columns : list of str, optional
        A list of column names to be considered for generating polynomial features. Defaults to ['load', 'ESMode6', 'Antennas', 'TXpower', 'Frequency'].
    degree : int, optional
        The degree of the polynomial features. Default is 2.

    Returns:
    --------
    dataframe : pd.DataFrame
        A new Pandas DataFrame with added polynomial features.

    Inner Workings:
    ---------------
    1. Feature Selection: Selects columns based on the 'list_columns' parameter for polynomial transformation.
    2. Polynomial Features: Uses scikit-learn's PolynomialFeatures to generate polynomial and interaction terms.
    3. Column Transformation: Fits and transforms the selected columns into new polynomial features.
    4. Feature Names: Retrieves the new feature names generated.
    5. DataFrame Conversion: Converts the numpy array of transformed features back to a DataFrame.
    6. Column Removal: Removes the original columns that were transformed.
    7. DataFrame Concatenation: Concatenates the original DataFrame with the newly generated polynomial features DataFrame.
    """
    # Instantiate PolynomialFeatures with the desired degree
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

    # Fit and transform the selected features
    transformed_features = polynomial_features.fit_transform(dataframe[list_columns])

    # Get the names of the new features
    new_feature_names = polynomial_features.get_feature_names_out(input_features=list_columns)

    # Convert the output into a DataFrame
    transformed_features_df = pd.DataFrame(transformed_features, columns=new_feature_names)

    # Remove the columns that are already in the original DataFrame
    for column in list_columns:
        if column in transformed_features_df.columns:
            transformed_features_df.drop(column, axis=1, inplace=True)

    unique_id = 'poly'
    transformed_features_df.columns = [column + "_" + unique_id for column in transformed_features_df.columns]

    # Concatenate the new DataFrame of polynomial features to the original DataFrame
    dataframe = pd.concat([dataframe, transformed_features_df], axis=1)

    return dataframe


def add_time_series_features(df_group, column='Energy', num_lags=3, num_diff=0, target_encoding=False,
                             data_leakage=False):
    """
    Add time series features such as lags, differences, and target encoding to a DataFrame group based on a specified column (ID of the time series).

    Parameters:
    -----------
    df_group : pd.DataFrame
        A DataFrame group for which to add time series features.
    column : str
        The column to be considered for time series feature generation.
    num_lags : int
        Number of lag features to add.
    num_diff : int
        Number of differences between lag features to add.
    target_encoding : bool
        Whether to add a target encoding feature.
    data_leakage : bool
        Executes the function in a way that could cause data leakage, such as using 'bfill' to remove null values and adding lead features.

    Returns:
    --------
    df_group : pd.DataFrame
        A DataFrame group with added lag, lead, and difference features.

    Inner Workings:
    ---------------
    1. NaN Handling: Determines how to fill NaN values based on the 'data_leakage' parameter.
    2. Column Copy: Creates a temporary copy of the specified column for feature engineering.
    3. Lag Generation: Adds lag features based on the 'num_lags' parameter.
    4. Difference Features: Adds difference features between lag values.
    5. Target Encoding: Optionally adds target encoding features if 'target_encoding' is True.
    6. DataFrame Return: Returns the modified DataFrame group with new features.
    """
    # Handle NaN values
    fill_fn = lambda x: x.interpolate(method='linear').ffill().bfill().fillna(0) if data_leakage else x.interpolate(
        method='linear').ffill().fillna(0)

    # Create a copy of the column
    if (df_group[column] == -1).any() or df_group[column].isnull().any():
        temp_column = fill_fn(df_group[column].replace(-1, np.nan))
    else:
        temp_column = df_group[column].copy()

    # Add lag and lead features
    if num_lags > 0:
        for i in range(1, num_lags):
            df_group[f'{column}_lag{i}'] = temp_column.shift(i)
            if data_leakage:
                df_group[f'{column}_lead{i}'] = temp_column.shift(-i)

    # Add difference features
    if num_diff > 0:
        for i in range(1, num_diff + 1):
            df_group[f'{column}_diff_lag_{i}'] = temp_column - temp_column.shift(i)
            if data_leakage:
                df_group[f'{column}_diff_lead_{i}'] = temp_column.shift(i) - temp_column.shift(-i)
    elif num_diff < 0:
        if data_leakage:
            for i in range(1, abs(num_diff) + 1):
                df_group[f'{column}_diff_lag_lead_{i}'] = temp_column.shift(i) - temp_column.shift(-i)

    if target_encoding:
        df_group["Energy_null"] = fill_fn(df_group["Energy"].replace(-1, np.nan))
        if data_leakage:
            df_group[f'{column}_target_encoding_Energy'] = compute_target_encoding_without_current(df_group, column)
        else:
            df_group[f'{column}_target_encoding_Energy'] = cumulative_target_encoding(df_group, column, "Energy_null")
            df_group[f'{column}_target_encoding_Energy'] = fill_fn(df_group[f'{column}_target_encoding_Energy'])
        df_group.drop(["Energy_null"], axis=1, inplace=True)

    return df_group


def create_features(path, data_leakage=False, use_arq=False):
    """
    Adds time series features, grouping, discretization, etc., to the provided DataFrame.

    Parameters:
    -----------
    path : str
        The path to the folder containing 'data_pivot_load.csv' (the final file generated by feature aggregation).
    data_leakage : bool
        Executes the function in a way that may cause data leakage, mainly when applying the add_time_series_features function.
    use_arq : bool
        If True, loads data from a file that has already had all features added, mainly used to avoid calling this function more than once if not necessary.

    Returns:
    --------
    df : pd.DataFrame
        A DataFrame with all the added features.

    Inner Workings:
    ---------------
    1. Data Loading: Decides between loading data from 'data_train_test.csv' or 'data_pivot_load.csv' based on the 'use_arq' parameter.
    2. DateTime Handling: Converts the 'Time' column to datetime format and extracts features like hour, day, and month.
    3. Data Sorting: Sorts the DataFrame based on the 'BS' and 'Time' columns.
    4. Temporal Feature Creation: Uses the 'add_time_series_features' function to add lags and differences to specified columns.
    5. Label Encoding: Converts categories in multiple columns to numerical codes using label encoding.
    6. Discretization: Divides some continuous columns into 10 discrete intervals.
    7. One-Hot Encoding: Applies one-hot encoding to the 'RUType' column.
    8. Grouped Features: Creates new columns through grouping and averaging specific features.
    9. K-means Application: Uses the k-means algorithm to group specific sets of columns.
    10. NaN and Inf Handling: Replaces Inf and -Inf values with NaN and then fills the NaN with 0.
    11. Index Reset: Resets the DataFrame index to facilitate future operations.
    12. Saves DataFrame: Stores the modified DataFrame in 'data_train_test.csv'.
    """
    # Check if the 'use_arq' flag is set to True
    if use_arq:
        df = pd.read_csv('../data/data_train_test.csv')
        return df

    # Otherwise, proceed with the following data processing steps
    df = pd.read_csv(f'{path}data_pivot_load.csv')

    # Extract temporal features from the 'Time' column
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['Day'] = df['Time'].dt.day
    df['Weekday'] = df['Time'].dt.weekday
    df['Weekend'] = (df['Weekday'] >= 5).astype(int)
    df['Month'] = df['Time'].dt.month
    df['Year'] = df['Time'].dt.year

    # Sort the dataframe by 'BS' and 'Time' columns
    df = df.sort_values(['BS', 'Time'])

    # Apply the function 'add_time_series_features' to each group defined by 'BS'
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='Energy', num_lags=5, num_diff=-1,
                                                  data_leakage=data_leakage)
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='load', num_lags=3, num_diff=2,
                                                  data_leakage=data_leakage)
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='Hour', num_lags=3, num_diff=2,
                                                  target_encoding=True, data_leakage=data_leakage)
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='Day', num_lags=3, num_diff=0,
                                                  data_leakage=data_leakage)
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='ESMode6', num_lags=3, num_diff=2,
                                                  data_leakage=data_leakage)
    df = df.groupby('BS', group_keys=False).apply(add_time_series_features, column='ESMode1', num_lags=3, num_diff=2,
                                                  data_leakage=data_leakage)

    # Label encoder for categorical columns
    df['BS_cat'] = df['BS'].astype('category').cat.codes
    df['RUType_cat'] = df['RUType'].astype('category').cat.codes
    df['CellName'] = df['CellName'].astype('category').cat.codes
    df['Mode'] = df['Mode'].astype('category').cat.codes

    # Discretize the main continuous variables into bins
    df['Energy_qcut'] = pd.cut(df['Energy'], bins=10, labels=list(range(1, 11))).astype(int)
    df['load_qcut'] = pd.cut(df['load'], bins=10, labels=list(range(1, 11))).astype(int)
    df['ESMode6_qcut'] = pd.cut(df['ESMode6'], bins=10, labels=list(range(1, 11))).astype(int)
    df['ESMode1_qcut'] = pd.cut(df['ESMode1'], bins=10, labels=list(range(1, 11))).astype(int)

    # Perform one-hot encoding for the 'RUType' column
    df = pd.get_dummies(df, columns=['RUType'], prefix='RUType')

    # Create aggregation features
    if data_leakage:
        df['Hour_TG_load_M2_M3'] = df.groupby(['Hour'])['load'].transform('mean')
        df['Hour_TG_ESMode1'] = df.groupby(['Hour'])['ESMode1'].transform('mean')
    else:
        df['Hour_TG_load_M2_M3'] = cumulative_target_encoding_by_time(df, 'load', 'Energy', 'Hour')
        df['Hour_TG_ESMode1'] = cumulative_target_encoding_by_time(df, 'ESMode1', 'Energy', 'Hour')

    # Apply k-means clustering for feature engineering
    transformer_kmeans_20 = Grouper(method='kmeans', n_components=20,
                                    cols_transform=["Frequency", "Bandwidth", "Antennas", "TXpower", "RUType_cat",
                                                    "CellName", "Mode"], drop_original=False, normalize='standard')
    transformer_kmeans_18 = Grouper(method='kmeans', n_components=18,
                                    cols_transform=["Frequency", "Bandwidth", "Antennas", "TXpower"],
                                    drop_original=False, normalize='standard')
    transformer_kmeans_17 = Grouper(method='kmeans', n_components=17, cols_transform=["RUType_cat", "CellName", "Mode"],
                                    drop_original=False, normalize='standard')
    df = transformer_kmeans_20.fit_transform(df)
    df = transformer_kmeans_18.fit_transform(df)
    df = transformer_kmeans_17.fit_transform(df)

    # Generate robust polynomial features
    df = generate_robust_features(df, degree=2)

    # Replace infinite values with NaN and Fill NaN values with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        try:
            df[col].fillna(0, inplace=True)
        except TypeError as e:
            print(f"Error in col: {col}")
            print(e)
            df[col] = df[col].astype('object')
            df[col].fillna(0, inplace=True)

    # Reset the index of the dataframe
    df = df.reset_index(drop=True)

    # Save the processed dataframe to a CSV file
    df.to_csv('data_train_test.csv', index=False)

    print(f'Dataframe with new features')
    print(df.head(10))
    print(df.shape)

    return df
