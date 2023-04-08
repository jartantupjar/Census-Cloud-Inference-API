import requests
import os

URL_PATH = "https://incomepredictionapi.onrender.com/"


def test_intro_api():
    """test intro api works"""
    r = requests.get(URL_PATH)

    print('URL: {},STATUS CODE: {}, RESPONSE {} DATA {}'.format(
        URL_PATH, r.status_code, r.json(), None))
    assert r.status_code == 200
    assert r.json() == {
        'message': 'this api is used for generating salary predictions'}


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

    full_api_url = os.path.join(URL_PATH, 'predict')
    r = requests.post(full_api_url, json=high_salary_example)
    print('URL: {},STATUS CODE: {}, RESPONSE {} DATA {}'.format(
        full_api_url, r.status_code, r.json(), high_salary_example))
    assert r.status_code == 200
    assert r.json() == '>50k'


if __name__ == '__main__':

    test_intro_api()
    test_api_upperbound()
