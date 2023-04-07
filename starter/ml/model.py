from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import pickle
# Optional: implement hyperparameter tuning.


def load_model(model_path: str):
    """
    gets current path and outputs the model
        that was saved in a pickle file
    """
    return pickle.load(open(model_path, 'rb'))


def load_encoders(encoders_path: str):
    """
    gets current path and outputs the encoders
        that were saved in a pickle file
    """
    return pickle.load(open(encoders_path, 'rb'))


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_on_slices(
        category: str,
        data: pd.DataFrame,
        y_data: np.array,
        y_pred: np.array,
        save_path: str = None):
    """
    calculate metrics without needing to run inference again
    category: the column of the dataframe to explore
    data: the dataframe to query
    y_data: the true labels
    y_pred: the predicted labels
    """
    assert y_data.shape[0] == y_pred.shape[0] == data.shape[0]
    slice_results = []
    for value in data[category].unique().tolist():
        mask = data[category] == value
        filt_y = y_data[mask]
        filt_y_pred = y_pred[mask]
        results = compute_model_metrics(filt_y, filt_y_pred)
        slice_results.append([category, value, *results])

    slice_metrics_df = pd.DataFrame(
        slice_results,
        columns=[
            'column',
            'slice',
            'precision',
            'recall',
            'fbeta'])

    if save_path:
        slice_metrics_df.to_string(open(save_path, 'w'))
    return slice_metrics_df


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
