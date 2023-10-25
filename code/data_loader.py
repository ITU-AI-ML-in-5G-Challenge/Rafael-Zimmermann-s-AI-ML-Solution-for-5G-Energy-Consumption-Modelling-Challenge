import pandas as pd


def merge_and_save_dataframes(file_path):
    """
    Consolidates various datasets related to energy consumption, power consumption,
    cell levels, and base station information into a single DataFrame.
    The resulting DataFrame is then saved as 'data_total.csv'.

    Parameters:
    -----------
    file_path : str
        The path to the directory where the input CSV files are stored.

    Returns:
    --------
    consolidated_dataframe : DataFrame
        The consolidated DataFrame, also saved as 'data_total.csv'.

    Inner Workings:
    ---------------
    1. Data Reading: Reads multiple CSV files to create individual DataFrames.
    2. Time Conversion: Converts the 'Time' column to datetime format.
    3. DataFrame Merging: Performs outer and left joins to consolidate data.
    4. Data Sorting: Sorts data by 'BS', 'Time', and 'CellName' columns.
    5. Data Saving: Saves the unified DataFrame as 'data_total.csv'.
    """

    # Creating individual DataFrames
    energy_consumption_df = pd.read_csv(f'{file_path}energy_consumption.csv')
    power_consumption_df = pd.read_csv(f'{file_path}power_consumption.csv')
    cell_level_df = pd.read_csv(f'{file_path}cell_level.csv')
    base_station_info_df = pd.read_csv(f'{file_path}base_station_inf.csv')

    # Converting 'Time' column to datetime format
    energy_consumption_df["Time"] = pd.to_datetime(energy_consumption_df["Time"])
    power_consumption_df["Time"] = pd.to_datetime(power_consumption_df["Time"])
    cell_level_df["Time"] = pd.to_datetime(cell_level_df["Time"])

    # Merging DataFrames on 'Time' and 'BS' columns
    temp_merge = pd.merge(energy_consumption_df, power_consumption_df,
                          on=["Time", "BS"], how="outer", suffixes=('_energy', '_power'))
    sorted_temp_merge = temp_merge.sort_values(by=["BS", "Time"])

    # Combining energy columns
    sorted_temp_merge["Energy"] = sorted_temp_merge["Energy_energy"].combine_first(sorted_temp_merge["Energy_power"])
    sorted_temp_merge.drop(columns=["Energy_energy", "Energy_power"], inplace=True)
    sorted_temp_merge = sorted_temp_merge[['Time', 'BS', 'Energy', 'w']]

    # Left join with cell_level_df
    temp_merge = pd.merge(sorted_temp_merge, cell_level_df, on=["Time", "BS"], how="left")
    sorted_temp_merge = temp_merge.sort_values(by=["BS", "Time"])

    # Left join with base_station_info_df
    consolidated_dataframe = pd.merge(sorted_temp_merge, base_station_info_df, on=["BS", "CellName"], how="left")
    sorted_consolidated_dataframe = consolidated_dataframe.sort_values(by=["BS", "Time", "CellName"])

    # Saving the consolidated DataFrame
    sorted_consolidated_dataframe.to_csv('data_total.csv', index=False)

    return sorted_consolidated_dataframe


def process_agg_dataframe(df_=None, agg_numeric_method='sum',
                          agg_numeric_special_method=('mean', ['Energy', 'w', 'Frequency', 'Antennas']),
                          cat_one_hot=None):
    """
    Aggregates and transforms a given DataFrame based on specified numerical and categorical methods.
    The resulting DataFrame is then saved as 'data_pivot_load.csv'.

    Parameters:
    -----------
    df_ : DataFrame, optional
        The DataFrame to be processed. If None, reads from 'data_total.csv'. Default is None.
    agg_numeric_method : str, optional
        The aggregation method for numerical columns. Default is 'sum'.
    agg_numeric_special_method : tuple, optional
        A tuple containing the special aggregation method and the columns it should be applied to.
        Default is ('mean', ['Energy', 'w', 'Frequency', 'Antennas']).
    cat_one_hot : str or None, optional
        Method for one-hot encoding categorical columns. If None, dont use one-hot encoding. Default is None.

    Returns:
    --------
    df_agg : DataFrame
        The aggregated DataFrame, also saved as 'data_pivot_load.csv'.

    Inner Workings:
    ---------------
    1. Data Loading: Reads 'data_total.csv' if no DataFrame is provided.
    2. Null Handling: Fills NA/NaN values with -1 for identification. (distinguish between train and test sets)
    3. Column Identification: Identifies numerical and categorical columns.
    4. Data Aggregation: Groups data by 'Time' and 'BS', applying specified aggregation methods.
    5. One-Hot Encoding: Optionally performs one-hot encoding for categorical columns.
    6. Data Sorting: Sorts the DataFrame by 'BS' and 'Time'.
    7. Data Saving: Saves the aggregated DataFrame as 'data_pivot_load.csv'.
    """
    # Load the DataFrame from the file data_total.csv
    if df_ is None:
        df = pd.read_csv("../data/data_total.csv")
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        df = df_.copy()

    # Replace NaN values with -1 to identify natural nulls in the data (distinguish between train and test sets)
    df.fillna(-1, inplace=True)

    # Define columns for grouping
    group_by_columns = ['Time', 'BS']

    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()[1:]

    print(f'numeric_columns {numeric_columns}')
    print(f'categorical_columns {categorical_columns}')

    # Create an aggregation dictionary
    aggregation_dict = {}

    for col in numeric_columns:
        aggregation_dict[col] = agg_numeric_method

    for col in agg_numeric_special_method[1]:
        aggregation_dict[col] = agg_numeric_special_method[0]

    # Handle optional one-hot encoding for categorical variables
    if cat_one_hot is not None:
        df_dummies = pd.get_dummies(df[categorical_columns], prefix=categorical_columns, drop_first=False)
        df = pd.concat([df, df_dummies], axis=1)
        for col in df.columns:
            if col not in group_by_columns + numeric_columns:
                aggregation_dict[col] = cat_one_hot
        for col in categorical_columns:
            if col not in group_by_columns:
                aggregation_dict[col] = lambda x: ','.join(set(x)) if len(set(x)) > 1 else list(set(x))[0]
    else:
        for col in categorical_columns:
            if col not in group_by_columns:
                aggregation_dict[col] = lambda x: ','.join(set(x)) if len(set(x)) > 1 else list(set(x))[0]

    # Perform the aggregation
    df_agg = df.groupby(group_by_columns).agg(aggregation_dict).reset_index()
    df_agg = df_agg.sort_values(['BS', 'Time'])

    # Save the processed DataFrame
    df_agg.reset_index(drop=True, inplace=True)
    df_agg.to_csv('data_pivot_load.csv', index=False)

    return df_agg
