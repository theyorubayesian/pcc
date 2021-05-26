"""
CLI for Churn Prediction System
See README.md for details on components of system

Author: theyorubayesian
26 May 2021
"""
import argparse

from src.churn_library import import_data
from src.churn_library import encoder_helper
from src.churn_library import evaluate_model
from src.churn_library import perform_eda
from src.churn_library import perform_feature_engineering
from src.churn_library import train_models

DEFAULT_DATA_PATH = 'data/bank_data.csv'

CAT_COLS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

TEST_SIZE = 0.3

PARAM = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to CSV data")
    args = parser.parse_args()

    data_path = args.data_path or DEFAULT_DATA_PATH
    df = import_data(data_path)
    perform_eda(df)

    encoded_df = encoder_helper(df, CAT_COLS)
    X_train, X_test, y_train, y_test = \
        perform_feature_engineering(encoded_df, keep_columns=KEEP_COLS, test_size=TEST_SIZE)

    models = train_models("models", X_train, X_test, y_train, param_grid=PARAM)
    evaluate_model(
        models["Logistic Regression"]["model"],
        "Logistic_Regression",
        X_test,
        (y_train, y_test),
        models["Logistic Regression"]["predictions"],
        output_dir="images/results",
        explain=False
    )
    evaluate_model(
        models["Random Forest"]["model"],
        "Random_Forest",
        X_test,
        (y_train, y_test),
        models["Random Forest"]["predictions"],
        output_dir="images/results",
        explain=True
    )
