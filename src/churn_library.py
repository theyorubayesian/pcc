"""
Helper functions for Churn Prediction

- theyorubayesian
"""
from typing import Any
from typing import Tuple
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pandas import Series

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

sns.set()


def import_data(pth: str = "data/bank_data.csv") -> DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    temp = pd.read_csv(pth, index_col=0)
    temp['Churn'] = temp['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1
    )
    return temp


def perform_eda(data: DataFrame, output_dir: str = "images/eda") -> None:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    temp = data.copy()
    os.makedirs(output_dir, exist_ok=True)

    def _plot_churn_hist(df, img_path):
        fig = plt.figure(figsize=(20, 10))
        sns.countplot(data=df, x="Churn")
        fig.suptitle("Churn Histogram", fontweight='bold')
        plt.ylabel("Frequency")
        plt.xlabel("Churn")
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    def _plot_age_hist(df, img_path):
        fig = plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        fig.suptitle("Age Histogram", fontweight='bold')
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    def _plot_marital_status_hist(df, img_path):
        fig = plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        fig.suptitle("Marital Status Histogram", fontweight='bold')
        plt.xlabel("Marital Status")
        plt.ylabel("Frequency (Normalized")
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    def _plot_total_trans_ct_dist(df, img_path):
        fig = plt.figure(figsize=(20, 10))
        sns.distplot(df['Total_Trans_Ct'])
        fig.suptitle("Total_Trans_Ct DistPlot", fontweight="bold")
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

    def _plot_correlation(df, img_path):
        fig = plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        fig.suptitle("Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        fig.savefig(img_path)
        plt.close(fig)

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
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
    output:
            df: pandas dataframe with new columns for
    """
    temp = data.copy()

    for col in category_lst:
        new_col = col + suffix
        temp[new_col] = temp.groupby(col)["Churn"].transform("mean")
    return temp


def perform_feature_engineering(df: DataFrame, keep_columns: list, test_size: float = 0.3):
    """
    input:
              df: pandas dataframe
              response: string of response name
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y = df.pop("Churn")
    X = df[keep_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(
        model_name: str,
        y_original: tuple,
        y_predicted: tuple,
        output_dir: str = "images/results"
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    y_train, y_test = y_original
    y_train_preds, y_test_preds = y_predicted

    plt.rc('figure', figsize=(7, 7))
    plt.text(0.50, 0.85, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.15, 0.60, str(classification_report(y_train, y_train_preds)), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.50, 0.50, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.15, 0.25, str(classification_report(y_test, y_test_preds)), {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.suptitle("Classification Report", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_Classification_Report.png"), bbox_inches="tight"
    )
    plt.close()


def feature_importance_plot(model: Any, data: DataFrame, output_pth: str = "images/results"):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [data.columns[i] for i in indices]

    plt.figure(figsize=(20, 20))
    plt.title("Random Forest Feature Importance")
    plt.bar(range(data.shape[1]), importances[indices])
    plt.ylabel("Importance")
    plt.xticks(range(data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(
        output_pth, "feature_importance_plot.png"), bbox_inches="tight")


def explain_model(model, X_test: DataFrame, output_dir: str = "images/results"):
    """
    Explain model using SHAP
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(20, 40))
    shap.summary_plot(
        shap_values, X_test, plot_type="bar", class_names=["Not churned", "Churned"], show=False
    )
    plt.suptitle("Shapley Additive Explanation", fontweight="bold")
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"),
                bbox_inches="tight")


def roc_curve_image(
        fitted_models: list,
        X_test: DataFrame,
        y_test: Series,
        output_dir: str = "images/results"
):
    """
    Plots Receiver-Operating-Characteristic
    """
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    for model in fitted_models:
        plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.suptitle("Receiver Operating Characteristic Curve", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ROC_Curve.png"))
    plt.close(fig)


def evaluate_model(
        model,
        model_name: str,
        X_test: DataFrame,
        y_original: Tuple[Series, Series],
        y_predicted: Tuple[Series, Series],
        explain: bool = False
):
    _, y_test = y_original

    classification_report_image(model_name, y_original, y_predicted)
    roc_curve_image([model], X_test, y_test)
    if explain:
        explain_model(model, X_test)
        feature_importance_plot(model, X_test)


def train_models(model_dir: str, X_train: DataFrame, X_test: DataFrame, y_train: Series, y_test: Series):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)

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

    evaluate_model(
        lrc,
        "Logistic_Regression",
        X_test,
        (y_train, y_test),
        (y_train_preds_lr, y_test_preds_lr),
        explain=False
    )
    evaluate_model(
        cv_rfc.best_estimator_,
        "Random Forest",
        X_test,
        (y_train, y_test),
        (y_train_preds_rf, y_test_preds_rf),
        explain=True
    )

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
