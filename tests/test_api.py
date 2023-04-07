import json
from fastapi.testclient import TestClient
import sys
sys.path.append('./')
from main import app
client = TestClient(app)


def test_intro_api():
    """test intro api works"""
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {
        'message': 'this api is used for generating salary predictions'}


def test_api_lowerbound():
    """check prediction that should be <=50k"""
    low_salary_example = {'age': "39",
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
                          'native-country': 'United-States'}
    r = client.post('/predict', json=low_salary_example)

    assert r.status_code == 200
    assert r.json() == '<=50k'


def test_api_upperbound():
    """check prediction that should be >50k"""
    high_salary_example = {'age': 52,
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

    r = client.post('/predict', json=high_salary_example)

    assert r.status_code == 200
    assert r.json() == '>50k'


def test_incorrect_format():
    """check api when workclass is improper"""
    incorrect_format = {'age': 52,
                        'workclass': 'Self-emp-not-inc',
                        'fnlgt': 209642,
                        'education': 'HS-grad',
                        'hours-per-week': 45,
                        'native-country': 'United-States'}

    r = client.post('/predict', json=incorrect_format)

    assert r.status_code == 422
    assert len(json.loads(r.content)['detail']) == 8  # 8 missing columns
    assert json.loads(r.content)['detail'][0]['type'] == "value_error.missing"


def test_incorrect_workclass():
    """check api when workclass is improper"""
    invalid_workclass_example = {'age': 52,
                                 'workclass': 'not-defined',
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

    r = client.post('/predict', json=invalid_workclass_example)

    assert r.status_code == 422
    assert json.loads(r.content)[
        'detail'][0]['type'] == "value_error.str.regex"


def test_incorrect_age():
    """check api when age is improper"""
    invalid_age_example = {'age': 1000,
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

    r = client.post('/predict', json=invalid_age_example)

    assert r.status_code == 422
    assert json.loads(r.content)[
        'detail'][0]['type'] == "value_error.number.not_le"
