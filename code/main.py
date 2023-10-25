from data_loader import merge_and_save_dataframes, process_agg_dataframe
from data_preprocessing import generate_masks, get_features_for_w1, get_features_for_w5, get_features_for_w10
from model_training import KFoldValidator, generate_submission
from base_model import Model, validate_and_predict

# Global Variables --------------------------------------------------------------------------------------------------------
PATH = '//'
SUBSAMPLING = True
DATA_LEAKAGE = False

# Creating an aggregated data table ----------------------------------------------------------------------------------------
df_merge = merge_and_save_dataframes(PATH)
df_total = process_agg_dataframe(agg_numeric_method='sum', agg_numeric_special_method=('mean', ['Energy', 'w', 'Frequency', 'Antennas']), cat_one_hot=None)

print(f'Dataframe df_merge ...')
print(df_merge)
print(f'Dataframe df_total ...')
print(df_total[20:30])

# Generate masks for Obj- A B C
mask_w1, mask_w5, mask_w10 = generate_masks(df_total)

# Preprocessing and Feature Selection --------------------------------------------------------------------------------------
usa_pickle_w1 = False
usa_pickle_w5 = False
usa_pickle_w10 = False

# Fold Group
fold_group_divide_w1=['RUType_cat', 'Energy_qcut', 'BS_cat']
fold_group_divide_w5=['load_qcut', 'Hour', 'Day']
fold_group_divide_w10=['load_qcut', 'ESMode1_qcut', 'ESMode6_qcut', 'Hour']

KFoldValidator._seed_everything(7)
# Special preprocessing for w1, w5, w10 models
X_train_w1, X_test_w1, y_train_w1, y_divide_w1 = get_features_for_w1(data_leakage=DATA_LEAKAGE, use_pickle=usa_pickle_w1, read_feature_file=False, fold_group_divide=fold_group_divide_w1)
X_train_w5, X_test_w5, y_train_w5, y_divide_w5 = get_features_for_w5(data_leakage=DATA_LEAKAGE, use_pickle=usa_pickle_w5, read_feature_file=False, mask=mask_w5, fold_group_divide=fold_group_divide_w5)
X_train_w10, X_test_w10, y_train_w10, y_divide_w10 = get_features_for_w10(data_leakage=DATA_LEAKAGE, use_pickle=usa_pickle_w5, read_feature_file=True, mask=mask_w10, fold_group_divide=fold_group_divide_w10)

# Training and Validation of Models ----------------------------------------------------------------------------------------
model = Model

# Optimized parameters
param_w1 = {
    'ridge': {
        'alpha': 1.0,
        'solver': 'auto',
    },
    'xgb': {
        'colsample_bytree': 0.8284861576665423,
        'gamma': 0.1295678854543193,
        'learning_rate': 0.13155971750935957,
        'max_depth': 7,
        'min_child_weight': 4,
        'n_estimators': 283,
        'subsample': 0.8908677567703485,
        'objective': 'reg:squarederror',
        'random_state': 7
    }
}

param_w5 = {
    'ridge': {'alpha': 0.01, 'solver': 'auto'},
    'xgb': {
          'objective': 'reg:squarederror',
          'random_state': 7
    }
}

param_w10 = {
    'ridge': {'alpha': 1, 'solver': 'auto'},
    'xgb': {
        'objective': 'reg:squarederror',
        'random_state': 7
    }
}

# Training and validation of w1, w5, and w10 models
val_w1, score_w1, preds_w1 = validate_and_predict(X_train_w1, y_train_w1, X_test_w1, y_divide_w1, model, usa_pickle_w1, 1, print_top_features=True, params=param_w1)
val_w5, score_w5, preds_w5 = validate_and_predict(X_train_w5, y_train_w5, X_test_w5, y_divide_w5, model, usa_pickle_w5, 5, print_top_features=True, params=param_w5)
val_w10, score_w10, preds_w10 = validate_and_predict(X_train_w10, y_train_w10, X_test_w10, y_divide_w10, model, usa_pickle_w10, 10, print_top_features=True, params=param_w10)

# Calculate the final score
final_score_wmape = KFoldValidator.calculate_final_score(score_w1[0], score_w5[0], score_w10[0])
final_score_mae = KFoldValidator.calculate_final_score(score_w1[1], score_w5[1], score_w10[1])

# Generating the Submission ------------------------------------------------------------------------------------------------
submission_df_final = generate_submission(mask_w1, mask_w5, mask_w10, preds_w1, preds_w5, preds_w10, PATH)

# Check and print null values information
print("Null values in each column:\n", submission_df_final.isnull().sum())
print(f"Number of rows with null values: {submission_df_final.isnull().any(axis=1).sum()}")

# Final data for submission
print(f"Modelo usado {model}")

print(f'WMAPE: {score_w1[0]:.5f}, {score_w5[0]:.5f}, {score_w10[0]:.5f}, {final_score_wmape:.5f}')
print(f'MAE: {score_w1[1]:.5f}, {score_w5[1]:.5f}, {score_w10[1]:.5f}, {final_score_mae:.5f}')