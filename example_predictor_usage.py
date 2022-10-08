import predictor

import warnings

warnings.filterwarnings("ignore")


z = {
    "Gender": "Male",
    "Married": "No",
    "Dependents": "3+",
    "Education": "Graduate",
    "Self_Employed": "Yes",
    "ApplicantIncome": 19795,
    "CoapplicantIncome": 4581.16,
    "LoanAmount": 558.3,
    "Loan_Amount_Term": 360.0,
    "Credit_History": "1",
}

print(predictor.predict(z))
