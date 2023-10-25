import pickle
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from model_training import KFoldValidator


class Model(BaseEstimator, RegressorMixin):
    """
    Model using Ridge regression and XGBoost for ensemble prediction.

    Parameters:
    - regressor_original : BaseEstimator
        Original regressor, default is Ridge regression.
    - regressor_residuals_xgb : xgb.XGBRegressor
        XGBoost regressor trained on residuals, with the default objective being 'reg:squarederror' and a random state of 7.

    Internal Working:
    -----------------
    1. Setup: Initializes Ridge regression as the base regressor and XGBoost for modeling residuals.
    2. Fit:
      a. Train Linear Regression on the original series.
      b. Calculate residuals using predictions from the linear regression.
      c. Train XGBoost on these residuals.
    3. Predict:
      a. Predict using the Ridge regression.
      b. Predict the residuals using XGBoost.
      c. The final prediction is the sum of the above two predictions.

    Returns:
    - array-like: Containing the predictions for the given test data.
    """

    def __init__(self, ridge_params=None, xgb_params=None):
        self.regressor_original = Ridge(**ridge_params) if ridge_params else Ridge()
        self.regressor_residuals_xgb = xgb.XGBRegressor(**xgb_params) if xgb_params else xgb.XGBRegressor(
            objective='reg:squarederror', random_state=7)

    def fit(self, X, y):
        self.X_train = X.copy()

        # 1. Train Linear Regression on original series
        self.regressor_original.fit(self.X_train, y)
        y_pred_original = self.regressor_original.predict(self.X_train)

        # 2. Train XGBoost on residuals
        residuals_xgb = y - y_pred_original
        self.regressor_residuals_xgb.fit(self.X_train, residuals_xgb)

        return self

    def predict(self, X_test):
        y_pred_original = self.regressor_original.predict(X_test)
        y_pred_xgb = self.regressor_residuals_xgb.predict(X_test)
        return y_pred_original + y_pred_xgb


def wmape(actual, forecast):
    """
    Calculates the WMAPE between lists of actual and forecasted values.

    Parameters:
    - actual (list): List of actual observed values.
    - forecast (list): List of forecasted values.

    Returns:
    - float: The value of WMAPE.
    """
    # Ensures that both lists are of the same size
    assert len(actual) == len(forecast), "Both lists should be of the same size."

    # Calculates the numerator and the denominator
    numerator = sum(abs(a - f) for a, f in zip(actual, forecast))
    denominator = sum(actual)

    # If the denominator is zero, the result is undefined (could be handled differently if desired)
    if denominator == 0:
        raise ValueError("Zero denominator. WMAPE is undefined.")

    return numerator / denominator


def validate_and_predict(X_train, y_train, X_test, y_divide, model, use_pickle, w, print_top_features=False,
                         params=None):
    """
    Validates and predicts using a combined approach of Ridge regression and XGBoost.

    Parameters:
    - X_train, y_train : array-like or pd.DataFrame
        Training data and corresponding labels.
    - X_test, y_divide : array-like or pd.DataFrame
        Test data and labels used for validation.
    - model : Model object
        Model (typically an ensemble of Ridge regression and XGBoost) to be validated.
    - use_pickle : bool
        If True, loads the model and predictions from a previously saved pickle file.
    - w : int or string
        Parameter for naming the saved pickle files.
    - print_top_features : bool
        If True, prints the top features of the Ridge and XGBoost models.

    Internal Working:
    -----------------
    1. Check for Existing Pickle: If `use_pickle` is True, it checks for an existing pickle file and loads the model and predictions from it.
    2. Model Validation:
      a. Validates the model using KFoldValidator.
      b. Generates scores based on the validation.
      c. Predicts on the test set.
    3. Save to Pickle:
      a. If `use_pickle` is False, it saves the model, scores, and predictions to a pickle file for later use.
    4. Feature Importance (optional):
      a. If `print_top_features` is True, it prints the most important features from the Ridge regression and XGBoost models.

    Returns:
    - tuple: Contains three elements - the validation object, scores, and predictions.
    """

    if params is None:
        params = {}
    if use_pickle:
        with open(f'model_w{w}.pkl', 'rb') as f:
            val, score, preds = pickle.load(f)
        return val, score, preds

    # Execute model
    val = KFoldValidator(model, n_splits=10, seed=7, k_fold_type=MultilabelStratifiedKFold,
                         score_type=[wmape, mean_absolute_error], params=params)
    score = val.validate(X_train, y_train, y_divide)
    preds = val.predict_avg(X_test)

    # Save model to a pickle file
    with open(f'model_w{w}.pkl', 'wb') as f:
        pickle.dump((val, score, preds), f)

    def print_top_features_ridge(n=50):
        """Print the most relevant coefficients of the Ridge model."""
        model_instance = val.models[0]
        coef = model_instance.regressor_original.coef_
        feature_names = model_instance.X_train.columns
        sorted_indices = np.argsort(np.abs(coef))[::-1]

        print(f"\nTop {n} features for {w} Linear:")
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"{i + 1}. {feature_names[idx]}: {coef[idx]:.4f}")

    def print_top_features_xgb(n=50):
        """Print the top features of the XGBoost model."""
        model_instance = val.models[0]
        importance_dict = model_instance.regressor_residuals_xgb.get_booster().get_score(importance_type='weight')
        sorted_importances = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)

        print(f"\nTop {n} features for {w} XGBoost:")
        for i, (feature, importance) in enumerate(sorted_importances[:n]):
            print(f"{i + 1}. {feature}: {importance:.4f}")

    if print_top_features:
        print_top_features_ridge()
        print_top_features_xgb()

    return val, score, preds
