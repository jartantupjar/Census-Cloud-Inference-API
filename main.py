from typing import Annotated
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml import model as mlutils
from starter.ml import data as datautils
# from typing_extensions import Annotated
import pandas as pd
import yaml

# import json
app = FastAPI(
    title="Salary Prediction API"

)

DATA_FORMAT = yaml.safe_load(open('data_model.yml'))
CLASSIFIER_MODEL = mlutils.load_model('model/model.pkl')
ENCODER, LABEL_BINARIZER = mlutils.load_encoders('model/encoders.pkl')
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class InputData(BaseModel):
    age: Annotated[int, Field(**DATA_FORMAT["age"])]
    workclass: Annotated[str, Field(**DATA_FORMAT["workclass"])]
    fnlgt: Annotated[int, Field(**DATA_FORMAT["fnlgt"])]
    education: Annotated[str, Field(**DATA_FORMAT["education"])]
    education_num: Annotated[int, Field(**DATA_FORMAT["education_num"])]
    marital_status: Annotated[str, Field(**DATA_FORMAT["marital_status"])]
    occupation: Annotated[str, Field(**DATA_FORMAT["occupation"])]
    relationship: Annotated[str, Field(**DATA_FORMAT["relationship"])]
    race: Annotated[str, Field(**DATA_FORMAT["race"])]
    sex: Annotated[str, Field(**DATA_FORMAT["sex"])]
    capital_gain: Annotated[int, Field(**DATA_FORMAT["capital_gain"])]
    capital_loss: Annotated[int, Field(**DATA_FORMAT["capital_loss"])]
    hours_per_week: Annotated[int, Field(**DATA_FORMAT["hours_per_week"])]
    native_country: Annotated[str, Field(**DATA_FORMAT["native_country"])]

    class Config:
        schema_extra = {
            "example": {'age': 52,
                        'workclass': 'Self-emp-not-inc',
                        'fnlgt': 209642,
                        'education': 'HS-grad',
                        'education-num': 9,
                        'marital-status': 'Married-civ-spouse',
                        'occupation': 'Exec-managerial',
                        'relationship': 'Husband',
                        'race': 'White',
                        'sex': 'Male',
                        'capital-gain': 50000,
                        'capital-loss': 0,
                        'hours-per-week': 45,
                        'native-country': 'United-States'},
            "examples": {
                "low salary example": {'age': "39",
                                       'workclass': 'State-gov',
                                       'fnlgt': 77516,
                                       'education': 'Bachelors',
                                       'education-num': 13,
                                       'marital-status': 'Never-married',
                                       'occupation': 'Adm-clerical',
                                       'relationship': 'Not-in-family',
                                       'race': 'White',
                                       'sex': 'Male',
                                       'capital-gain': 2174,
                                       'capital-loss': 0,
                                       'hours-per-week': 40,
                                       'native-country': 'United-States'},
                "high salary example": {'age': 52,
                                        'workclass': 'Self-emp-not-inc',
                                        'fnlgt': 209642,
                                        'education': 'HS-grad',
                                        'education-num': 9,
                                        'marital-status': 'Married-civ-spouse',
                                        'occupation': 'Exec-managerial',
                                        'relationship': 'Husband',
                                        'race': 'White',
                                        'sex': 'Male',
                                        'capital-gain': 50000,
                                        'capital-loss': 0,
                                        'hours-per-week': 45,
                                        'native-country': 'United-States'}
            }
        }


@app.get("/")
async def intro():
    """
    Welcome message API. Returns a summary of the APIs functionality
    """
    return {'message': 'this api is used for generating salary predictions'}


@app.post("/predict")
async def predict(data: InputData):
    # async def
    # predict(data:Annotated[InputData,Body(None,examples=InputData.Config.schema_extra["examples"])]):
    """
    Returns the Prediction of the passed input

    """
    print(data.dict())
    # reuse the data variable to avoid possible memory overlflow issues
    data = pd.DataFrame.from_records([data.dict()])
    # convert "-" to "_" because json format uses -
    data.columns = [key.replace("_", "-") for key in data.columns]

    data = datautils.preprocess_data(data)
    data, _, _, _ = datautils.process_data(
        data, CATEGORICAL_FEATURES, training=False, encoder=ENCODER, lb=LABEL_BINARIZER)
    prediction = mlutils.inference(CLASSIFIER_MODEL, data)[0]

    prediction_label = ">50k" if prediction > 0.5 else "<=50k"

    return prediction_label
    # return {"salary":prediction_label,"value":float(prediction)}
