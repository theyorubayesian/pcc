import os
import logging
from typing import Callable

import pandas as pd
from pandas import DataFrame

import src.churn_library as cl
from main import CAT_COLS
from main import KEEP_COLS

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def test_import(import_data: Callable):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        assert 'Churn' in df.columns
        assert (df.Churn.nunique() == 2) and (df.Churn.unique() == [0, 1])
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing `import_data`: Error in `Churn` column created")
        raise err


def test_eda(perform_eda: Callable, df: DataFrame):
    """
    test perform eda function
    """
    output_dir = "tests/output"

    try:
        perform_eda(df, output_dir)
        output_files = os.listdir(output_dir)
        assert len(output_files) != 0
        assert "Churn-Histogram.png" in output_files
        assert "Customer-Age-Histogram.png" in output_files
        assert "Marital-Status-Distribution-BarPlot.png" in output_files
        assert "Total-Trans-Ct-DistPlot.png" in output_files
        assert "Correlation-Heatmap.png" in output_files
        logging.info("Testing `perform_eda`: SUCCESS")
    except AssertionError as err:
        logging.error("Testing `perform_eda`: One or more plot images not created")
        raise err


def test_encoder_helper(encoder_helper: Callable, df: DataFrame):
    """
    test encoder helper
    """
    try:
        assert all([col in list(df.columns) for col in CAT_COLS])
        assert all([col in list(df.columns) for col in CAT_COLS])
    except AssertionError as err:
        logging.info(
            "Testing `encoder_helper`: One or more categorical column is not in dataframe"
        )
        raise err

    encoded_columns_suffix = "_Churn"
    try:
        encoded_df = encoder_helper(df, CAT_COLS, encoded_columns_suffix)
    except Exception as err:
        logging.info("Testing `encoder_helper`: Error occurred in function")
        raise err

    encoded_columns = [col + encoded_columns_suffix for col in CAT_COLS]

    try:
        assert all([col in list(encoded_df.columns) for col in encoded_columns])
        with pd.option_context('mode.user_inf_as_na', True):
            assert encoded_df[encoded_columns].notna().any().sum() == 0
    except AssertionError as err:
        logging.info("Testing `encoder_helper`: Error in one or more encoded columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering: Callable, df: DataFrame):
    """
    test perform_feature_engineering
    """
    test_size = 0.3
    try:
        result = perform_feature_engineering(df, KEEP_COLS, test_size=test_size)
    except Exception as err:
        logging.info(
            "Testing `perform_feature_engineering`: Function errors out before returning result"
        )
        raise err

    try:
        assert len(result) == 4

        X_train, X_test, y_train, y_test = result
        assert list(X_train.columns) == list(X_test.columns) == KEEP_COLS
        assert X_test.shape[0] == y_test.shape[0] == int(test_size * df.shape[0])
        assert X_train.shape[0] == y_train.shape[0] == int((1-test_size) * df.shape[0])
    except AssertionError as err:
        logging.info("Testing `perform_feature_engineering`: Problems with result returned by function")
        raise err


def test_train_models(train_models, *args):
    """
    test train_models
    """
    try:
        output = train_models(*args)
    except Exception as err:
        logging.info("Testing `train_models`: Function errors out before returning result")
        raise err

    try:
        assert isinstance(output, dict)
        assert ("Logistic Regression" in output) and ("Random Forest" in output)

        assert isinstance(output["Logistic Regression"], dict) and isinstance(output["Random Forest"], dict)
        assert ("model" in output["Logistic Regression"]) and ("model" in output["Random Forest"])
        assert isinstance(output["Logistic Regression"]["predictions"], tuple)
        assert isinstance(output["Random Forest"]["predictions"], tuple)
    except AssertionError as err:
        # TODO
        # logging.info()
        print(err)


if __name__ == "__main__":
    test_import(cl.import_data)

    test_data = cl.import_data("tests/encoder_test_fixture.txt")
    test_eda(cl.perform_eda, test_data)
    test_encoder_helper(cl.encoder_helper, test_data)
    test_perform_feature_engineering(cl.perform_feature_engineering, test_data)

    encoded_test_data = cl.encoder_helper(test_data, CAT_COLS, suffix="_Churn")
    dataset = cl.perform_feature_engineering(encoded_test_data, KEEP_COLS, test_size=0.3)

    test_train_models(cl.train_models, *dataset)
