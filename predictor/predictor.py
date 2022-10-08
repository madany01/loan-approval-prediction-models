"""
- usage:
    ```py
    import predictor

    predictor.predict({
        "Gender": "Male",
        "Married": "Yes",
        # ..etc
    })
    ```
"""

from pathlib import Path
import pickle
from typing import Any


ordered_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

discrete_column_encoder_dict = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Not Graduate": 0, "Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Credit_History": {"0": 0, "1": 1},
}


models: dict[str, Any] = {}


def _predict_bayes(encoded_x, *, proba=False):
    discrete_bayes = models["bayes"]["discrete"]
    continuous_bayes = models["bayes"]["continuous"]

    discrete_x_arr = []
    continuous_x_arr = []

    for column_name in ordered_columns:
        encoded_val = encoded_x[column_name]
        if column_name in discrete_column_encoder_dict:
            discrete_x_arr.append(encoded_val)
        else:
            continuous_x_arr.append(encoded_val)

    discrete_x_arr = [discrete_x_arr]
    continuous_x_arr = [continuous_x_arr]

    p0_discrete, p1_discrete = discrete_bayes.predict_proba(discrete_x_arr)[0]
    p0_continuous, p1_continuous = continuous_bayes.predict_proba(continuous_x_arr)[0]

    p0 = p0_discrete * p0_continuous
    p1 = p1_discrete * p1_continuous
    p_sum = p0 + p1

    p0 /= p_sum
    p1 /= p_sum

    if not proba:
        return 1 if (p1 >= p0) else 0

    return [p0, p1]


def model_predict(model_name: str, x: dict, *, proba=False):
    encoded_x = {}

    for column_name in ordered_columns:
        val = x[column_name]

        if column_name in discrete_column_encoder_dict:
            val = discrete_column_encoder_dict[column_name][val]

        encoded_x[column_name] = val

    if model_name == "bayes":
        return _predict_bayes(encoded_x, proba=proba)

    x_arr = [list(encoded_x.values())]

    if not proba:
        return models[model_name].predict(x_arr)[0]

    return models[model_name].predict_proba(x_arr)[0]


def predict(x):
    models_prediction = {}

    num_0 = 0
    num_1 = 0

    for model_name in models:
        prediction = model_predict(model_name, x)
        models_prediction[model_name] = prediction

        if prediction == 0:
            num_0 += 1
        else:
            num_1 += 1

    models_prediction["voting_majority"] = 1 if num_1 >= num_0 else 0

    return models_prediction


def main():
    for model_name in ["bayes", "tree", "logistic", "svm", "knn"]:

        path = (Path(__file__).parent / '..' / "pickled_models" / model_name).absolute()
        with open(path, "rb") as file:
            models[model_name] = pickle.load(file)


main()


__all__ = [predict]
