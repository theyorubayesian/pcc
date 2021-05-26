"""
Helper functions for Churn Prediction

Author: theyorubayesian
"""
from typing import Tuple
import os

import joblib
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pandas import Series

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.plots import _plot_age_hist
from src.plots import _plot_churn_hist
from src.plots import _plot_correlation
from src.plots import _plot_marital_status_hist
from src.plots import _plot_total_trans_ct_dist
from src.plots import classification_report_image
from src.plots import explain_model
from src.plots import feature_importance_plot
from src.plots import roc_curve_image

sns.set()


def import_data(pth: str = "data/bank_data.csv") -> DataFrame:
    """
    Returns dataframe for the csv found at pth.
    Creates target variable: `Churn`.

    :param pth: Path to CSV dataset
    :return: Dataframe
    """
    temp = pd.read_csv(pth, index_col=0)
    temp['Churn'] = temp['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )
    return temp


def perform_eda(data: DataFrame, output_dir: str = "images/eda") -> None:
    """
    Perform EDA on df and save figures to images folder

    :param data: Data to perform EDA on
    :param output_dir: Output directory for plots
    :return None
    """
    temp = data.copy()
    os.makedirs(output_dir, exist_ok=True)

    _plot_churn_hist(temp, os.path.join(output_dir, "Churn-Histogram.png"))
    _plot_age_hist(temp, os.path.join(
        output_dir, "Customer-Age-Histogram.png"))
    _plot_marital_status_hist(
        temp, os.path.join(
            output_dir, "Marital-Status-Distribution-BarPlot.png")
    )
    _plot_total_trans_ct_dist(
        temp, os.path.join(output_dir, "Total-Trans-Ct-DistPlot.png")
    )
    _plot_correlation(temp, os.path.join(
        output_dir, "Correlation-Heatmap.png"))


def encoder_helper(data: DataFrame, category_lst: list, suffix: str = "_Churn") -> DataFrame:
    """
    Helper function to encode categorical column

    :param data: Dataframe containing categorical columns to be encoded
    :param category_lst: List of categorical columns to encode
    :param suffix: Added to column names to create new, encoded columns
    :return: Dataframe containing existing columns and new, encoded columns
    """
    temp = data.copy()

    for col in category_lst:
        new_col = col + suffix
        temp[new_col] = temp.groupby(col)["Churn"].transform("mean")
    return temp


def perform_feature_engineering(df: DataFrame, keep_columns: list, test_size: float = 0.3):
    """
    Engineer features to be used in training. Feature selection occurs here

    :param df: Dataset to be engineered
    :param keep_columns: Features selected to be used for training
    :param test_size:" For `train_test_split`
    :return: X_train, X_test, y_train, y_test containing only `keep_columns`
    """
    y = df.pop("Churn")
    X = df[keep_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_models(
        model_dir: str,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        param_grid: dict = None
):
    """
    Trains and persists model on provided dataset

    :param model_dir: Output dir for persisted models
    :param X_train: Train Dataframe of X values
    :param X_test: Test Dataframe of X values
    :param y_train: Train Series of y values
    :param param_grid: Dict of `Random Forest` params for GridSearch
    :return: dictionary with keys: ['Logistic Regression', 'Random Forest']
        Each key maps to a dictionary with keys: ['model', 'predictions']
    """
    rfc = RandomForestClassifier(random_state=42)

    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    joblib.dump(lrc, os.path.join(model_dir, "logreg.pkl"))
    joblib.dump(cv_rfc.best_estimator_, os.path.join(model_dir, "rfc.pkl"))

    return {
        "Logistic Regression": {
            "model": lrc,
            "predictions": (y_train_preds_lr, y_test_preds_lr)
        },
        "Random Forest": {
            "model": cv_rfc.best_estimator_,
            "predictions": (y_train_preds_rf, y_test_preds_rf)
        }
    }


def evaluate_model(
        model,
        model_name: str,
        X_test: DataFrame,
        y_original: Tuple[Series, Series],
        y_predicted: Tuple[Series, Series],
        output_dir: str = "images/results",
        explain: bool = False
):
    """
    Evaluate fitted model. Creates and saves the ff plots:
        - Classification Report
        - ROC Curve
        - Other plots may be saved. See `explain`

    :param model: Fitted model to be evaluated
    :param model_name: Used to name image file
    :param X_test: Test Dataframe of X values
    :param y_original: Tuple(y_train, y_test)
    :param y_predicted: Tuple(predicted_y_train, predicted_y_test)
    :param output_dir: Output directory for plots
    :param explain: If True, two additional plots are created"
        - Feature Importance Plot
        - SHAP Summary Plot
    :return: None
    """
    _, y_test = y_original

    classification_report_image(model_name, y_original, y_predicted, output_dir=output_dir)
    roc_curve_image(model, model_name, X_test, y_test, output_dir=output_dir)
    if explain:
        explain_model(model, X_test, output_dir=output_dir)
        feature_importance_plot(model, X_test)
