# Script to train machine learning model
from ml import data as datautils
from ml import model as mlutils
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import logging
# setup results logger
logging.basicConfig(
    filename='model/train_results.log',
    level=logging.INFO,
    format='%(asctime)s |%(name)s | %(message)s',
    encoding=None)
logger = logging.getLogger('train_results')

# initialize values
path = 'data/census_cleaned.csv'
encoders_path = 'model/encoders.pkl'
model_path = 'model/model.pkl'

slice_output_path = "slice_output.txt"
category_to_slice = 'education'
# Add code to load in the data.
data = pd.read_csv(path)

# additional preprocessing because '?' adds an uncessary amount of
# uncertainty to the model:
data = datautils.preprocess_data(data)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# Process the train data with the process_data function.
X_train, y_train, encoder, lb = datautils.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# save encoders
pickle.dump([encoder, lb], open(encoders_path, 'wb'))

# Process the test data with the process_data function.
X_test, y_test, _, _ = datautils.process_data(
    test, cat_features, "salary", training=False, encoder=encoder, lb=lb)

# Train and save the model.
model = mlutils.train_model(X_train, y_train)
pickle.dump(model, open(model_path, 'wb'))

# run inference on train
y_train_pred = mlutils.inference(model, X_train)
train_results = mlutils.compute_model_metrics(y_train, y_train_pred)
logger.info("training results")
logger.info(train_results)
# run inference on test
y_test_pred = mlutils.inference(model, X_test)
test_results = mlutils.compute_model_metrics(y_test, y_test_pred)
logger.info("test results")
logger.info(test_results)
# Process the test data with the process_data function.
X_test, y_test, _, _ = datautils.process_data(
    test, cat_features, "salary", training=False, encoder=encoder, lb=lb)

# compute metrics based on the column/feature selected
slice_metrics_df = mlutils.compute_model_metrics_on_slices(
    category_to_slice, test, y_test, y_test_pred, slice_output_path)
