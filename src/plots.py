"""
Plotting functions for Churn Prediction System.
EDA Plotting functions come first. Then model evaluation plotting functions.

Author: theyorubayesian
26 May 2021
"""
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from pandas import DataFrame
from pandas import Series

sns.set()


def _plot_churn_hist(df, img_path):
    """
    Plots distribution of Churn in data
    """
    fig = plt.figure(figsize=(20, 10))
    sns.countplot(data=df, x="Churn")
    fig.suptitle("Churn Histogram", fontweight='bold')
    plt.ylabel("Frequency")
    plt.xlabel("Churn")
    plt.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def _plot_age_hist(df, img_path):
    """
    Plots distribution of Age in data
    """
    fig = plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    fig.suptitle("Age Histogram", fontweight='bold')
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def _plot_marital_status_hist(df, img_path):
    """
    Plots distribution of marital status in data
    """
    fig = plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    fig.suptitle("Marital Status Histogram", fontweight='bold')
    plt.xlabel("Marital Status")
    plt.ylabel("Frequency (Normalized")
    plt.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def _plot_total_trans_ct_dist(df, img_path):
    """
    Plots density of Total transactions completed
    """
    fig = plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    fig.suptitle("Total_Trans_Ct DistPlot", fontweight="bold")
    plt.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def _plot_correlation(df, img_path):
    """
    Plots correlation heatmap for dataset
    """
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.suptitle("Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)


def classification_report_image(
        model_name: str,
        y_original: tuple,
        y_predicted: tuple,
        output_dir: str = "images/results"
):
    """
    Produces classification report for training and testing results

    :param model_name: Used to name image file
    :param y_original:  Tuple(y_train, y_test)
    :param y_predicted: Tuple(predicted_y_train, predicted_y_test)
    :param output_dir: Output directory for plot
    :return: None
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
    Creates and stores the feature importances in pth

    :param model: Model object containing feature_importances_
    :param data: Dataframe of X values
    :param output_pth: Output directory for plot
    :return: None
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


def roc_curve_image(
        model,
        model_name: str,
        X_test: DataFrame,
        y_test: Series,
        output_dir: str = "images/results"
):
    """
    Plots Receiver-Operating-Characteristic

    :param model: Fitted model to create plot for
    :param model_name: Used to name image file
    :param X_test: Test Dataframe of X values
    :param y_test: Test Series of y values
    :param output_dir: Output directory for plot
    :return: None
    """
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.suptitle("Receiver Operating Characteristic Curve", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_ROC_Curve.png"))
    plt.close(fig)


def explain_model(model, X_test: DataFrame, output_dir: str = "images/results"):
    """
    Explain model using SHAP

    :param model: Fitted model to create plot for
    :param X_test: Test Dataframe of X values
    :param output_dir: Output directory for plot
    :return: None
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
