"""
Test suite for Churn Prediction system

Author: theyorubayesian
26 May 2021
"""
import unittest
import logging
import tempfile
from unittest.mock import patch

import pandas as pd
from sklearn.linear_model import LogisticRegression

import src.churn_library as cl
from main import CAT_COLS
from main import KEEP_COLS

logfile = './logs/churn_library.log'
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


class ChurnLibraryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_csv_file = "testfile.csv"
        try:
            data = pd.read_csv("tests/encoder_test_fixture.txt")
            encoded_columns = pd.read_csv("tests/encoded_columns.txt")
        except Exception as err:
            logging.error("ERROR: Could not create fixtures for test")
            raise err
        self.test_data = data
        self.encoded_columns = encoded_columns
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def modelTrainingSetUp(self):
        encoded_df = pd.concat([self.test_data, self.encoded_columns], axis=1)
        self.dataset = cl.perform_feature_engineering(encoded_df, KEEP_COLS)

    def modelEvaluationSetUp(self):
        self.modelTrainingSetUp()
        self.model = LogisticRegression().fit(self.dataset[0], self.dataset[2])

    @patch("src.churn_library.pd.read_csv")
    def test_import_data(self, mock_read_csv):
        err = None
        mock_read_csv.return_value = self.test_data.drop("Churn", axis=1).copy()

        result = cl.import_data(self.test_csv_file)
        try:
            mock_read_csv.assert_called_with(self.test_csv_file, index_col=0)
            pd.testing.assert_frame_equal(result, self.test_data)
        except AssertionError as err:
            logging.error("Testing `import_data`: Error in function result")
            raise err

        if err is None:
            logging.info("Testing `import_data`: SUCCESS")

    @patch("src.churn_library._plot_age_hist")
    @patch("src.churn_library._plot_churn_hist")
    @patch("src.churn_library._plot_correlation")
    @patch("src.churn_library._plot_marital_status_hist")
    @patch("src.churn_library._plot_total_trans_ct_dist")
    def test_perform_eda(
            self,
            mock_total_trans_plot,
            mock_marital_status_plot,
            mock_corr_plot,
            mock_churn_plot,
            mock_age_plot
    ):
        err = None

        try:
            cl.perform_eda(self.test_data, output_dir=self.test_dir.name)
            mock_churn_plot.assert_called_once()
            mock_age_plot.assert_called_once()
            mock_marital_status_plot.assert_called_once()
            mock_total_trans_plot.assert_called_once()
            mock_corr_plot.assert_called_once()
        except AssertionError as err:
            logging.error("Testing `perform_eda`: One or more plots not created")
            raise err

        if err is None:
            logging.info("Testing `perform_eda`: SUCCESS")

    def test_encoder_helper(self):
        err = None
        try:
            assert all([col in list(self.test_data.columns) for col in CAT_COLS])
        except AssertionError as err:
            logging.error(
                "Testing `encoder_helper`: "
                "One or more categorical columns is not in fixture dataframe"
            )
            raise err

        encoded_columns_suffix = "_Churn"
        try:
            encoded_df = cl.encoder_helper(self.test_data, CAT_COLS, encoded_columns_suffix)
        except Exception as err:
            logging.error("Testing `encoder_helper`: Error occurred in function")
            raise err

        encoded_columns = [col + encoded_columns_suffix for col in CAT_COLS]
        try:
            assert all([col in list(encoded_df.columns) for col in encoded_columns])
            with pd.option_context('mode.use_inf_as_na', True):
                assert encoded_df[encoded_columns].isna().any().sum() == 0
            pd.testing.assert_frame_equal(encoded_df[encoded_columns], self.encoded_columns)
        except AssertionError as err:
            logging.error("Testing `encoder_helper`: Error in one or more encoded columns")
            raise err

        if err is None:
            logging.info("Testing `encoder helper`: SUCCESS")

    @patch("src.churn_library.train_test_split", return_value=(None, None, None, None))
    def test_perform_feature_engineering(self, mock_train_test_split):
        err = None

        try:
            _ = cl.perform_feature_engineering(
                pd.concat([self.test_data, self.encoded_columns], axis=1),
                KEEP_COLS, test_size=0.3
            )
            mock_train_test_split.assert_called_once()
        except AssertionError as err:
            logging.error(
                "Testing `perform_feature_engineering`: Ensure that `train_test_split` is called"
            )
            raise err

        if err is None:
            logging.info("Testing `perform_feature_engineering`: SUCCESS")

    @patch("src.churn_library.joblib")
    def test_train_model(self, mock_joblib):
        err = None
        self.modelTrainingSetUp()

        try:
            training_output = cl.train_models(self.test_dir.name, *self.dataset[:-1])
        except Exception as err:
            logging.error("Testing `train_model`: Error occurred in function")
            raise err

        try:
            mock_joblib.dump.assert_called()
        except AssertionError as err:
            logging.error("Testing `train_model`: Ensure that model is persisted after training")
            raise err

        try:
            assert isinstance(training_output, dict)
            assert "Logistic Regression" in training_output
            assert "Random Forest" in training_output

            assert isinstance(training_output["Logistic Regression"], dict)
            assert isinstance(training_output["Logistic Regression"]["predictions"], tuple)

            assert isinstance(training_output["Random Forest"], dict)
            assert isinstance(training_output["Random Forest"]["predictions"], tuple)
        except AssertionError as err:
            logging.error("Testing `train_models`: Error in output produced")

        if err is None:
            logging.info("Testing `train_model`: SUCCESS")

    @patch("src.churn_library.classification_report_image")
    @patch("src.churn_library.roc_curve_image")
    @patch("src.churn_library.explain_model")
    @patch("src.churn_library.feature_importance_plot")
    def test_evaluate_model(
            self,
            mock_feature_plot,
            mock_explain_model,
            mock_roc_image,
            mock_classification_image
    ):
        err = None
        self.modelEvaluationSetUp()

        try:
            cl.evaluate_model(
                self.model,
                "test_model",
                self.dataset[1],
                (None, None),
                (None, None),
                output_dir="tests/results",
                explain=True
            )
            mock_classification_image.assert_called_once()
            mock_roc_image.assert_called_once()
            mock_explain_model.assert_called_once()
            mock_feature_plot.assert_called_once()
        except AssertionError as err:
            logging.error("Testing `evaluate_model`: Unexpected function behavior")
            raise err

        mock_classification_image.reset_mock()
        mock_roc_image.reset_mock()
        mock_explain_model.reset_mock()
        mock_feature_plot.reset_mock()

        try:
            cl.evaluate_model(
                self.model,
                "test_model",
                self.dataset[1],
                (None, None),
                (None, None),
                output_dir="tests/results",
                explain=False
            )
            mock_classification_image.assert_called_once()
            mock_roc_image.assert_called_once()
            mock_explain_model.assert_not_called()
            mock_feature_plot.assert_not_called()
        except AssertionError as err:
            logging.error("Testing `evaluate_model`: Unexpected function behavior")
            raise err

        if err is None:
            logging.info("Testing `evaluate_model`: SUCCESS")


if __name__ == "__main__":
    with open(logfile, 'a') as f:
        unittest.main()
