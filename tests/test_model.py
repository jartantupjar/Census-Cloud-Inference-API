import sys
sys.path.append('./')
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.base import MultiOutputMixin, ClassifierMixin, BaseEstimator
from starter.ml import model as mlutils
from starter.ml import data as datautils
import pandas as pd
import numpy as np
import pytest
np.random.seed(42)


@pytest.fixture
def model_data():
    # generate dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)
    return X, y


@pytest.fixture
def dataframe_data():
    # generate dummy data
    category = 'feature1'
    target = 'target'
    data = pd.DataFrame({
        'feature1': np.random.choice(['A', 'B', 'C'], size=100),
        'feature2': np.random.randint(10, size=100),
        'target': np.random.randint(2, size=100)
    })
    return data, category, target


def test_model_type(model_data):
    # get dummy data
    X, y = model_data

    # get trained model
    model = mlutils.train_model(X, y)

    # compute metrics on slices
    assert isinstance(model, BaseEstimator)\
        or isinstance(model, ClassifierMixin)\
        or isinstance(model, MultiOutputMixin)


def test_compute_model_metrics_on_slices(dataframe_data):
    # get dummy data
    data, category_to_slice, target = dataframe_data
    y_data = data[target]
    y_pred = y_data

    # compute metrics on slices
    slice_metrics_df = mlutils.compute_model_metrics_on_slices(
        category_to_slice, data, y_data, y_pred)

    # check that the dataframe has the expected columns
    expected_columns = ['column', 'slice', 'precision', 'recall', 'fbeta']
    assert slice_metrics_df.columns.tolist() == expected_columns

    # check that the dataframe has the expected number of rows
    assert slice_metrics_df.shape[0] == data[category_to_slice].nunique()


def test_process_data(dataframe_data):
    # Create test data with categorical and continuous features and a label
    data, category_to_slice, target = dataframe_data

    # Test training mode
    X_train, y_train, encoder_train, lb_train = datautils.process_data(
        X=data,
        categorical_features=[category_to_slice],
        label=target,
        training=True
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder_train, OneHotEncoder)
    assert isinstance(lb_train, LabelBinarizer)
    assert X_train.shape[0] == data.shape[0]
    assert y_train.shape[0] == data.shape[0]

    # Test inference mode
    X_test, y_test, encoder_test, lb_test = datautils.process_data(
        X=data,
        categorical_features=[category_to_slice],
        label=target,
        training=False,
        encoder=encoder_train,
        lb=lb_train
    )
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(encoder_test, OneHotEncoder)
    assert isinstance(lb_test, LabelBinarizer)
    assert X_train.shape[0] == data.shape[0]
    assert y_train.shape[0] == data.shape[0]
