import numpy as np

def preprocess_input(form_data):
    """
    form_data: dict from Flask request.form
    returns: numpy array shape (1, 19)
    """

    # 1️⃣ Binary mappings (exactly like notebook)
    binary_map = {
        "Yes": 1, "No": 0,
        "Male": 1, "Female": 0
    }

    gender = binary_map[form_data["gender"]]
    Partner = binary_map[form_data["Partner"]]
    Dependents = binary_map[form_data["Dependents"]]
    PhoneService = binary_map[form_data["PhoneService"]]
    PaperlessBilling = binary_map[form_data["PaperlessBilling"]]

    # 2️⃣ LabelEncoder equivalents (MUST match training categories)
    MultipleLines_map = {"No": 0, "No phone service": 1, "Yes": 2}
    InternetService_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    OnlineSecurity_map = {"No": 0, "No internet service": 1, "Yes": 2}
    OnlineBackup_map = {"No": 0, "No internet service": 1, "Yes": 2}
    DeviceProtection_map = {"No": 0, "No internet service": 1, "Yes": 2}
    TechSupport_map = {"No": 0, "No internet service": 1, "Yes": 2}
    StreamingTV_map = {"No": 0, "No internet service": 1, "Yes": 2}
    StreamingMovies_map = {"No": 0, "No internet service": 1, "Yes": 2}
    Contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    PaymentMethod_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }

    # 3️⃣ Numeric fields
    tenure = float(form_data["tenure"])
    MonthlyCharges = float(form_data["MonthlyCharges"])
    TotalCharges = float(form_data["TotalCharges"])

    # 4️⃣ Final feature vector (ORDER MATTERS)
    features = [
        gender,
        int(form_data["SeniorCitizen"]),
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines_map[form_data["MultipleLines"]],
        InternetService_map[form_data["InternetService"]],
        OnlineSecurity_map[form_data["OnlineSecurity"]],
        OnlineBackup_map[form_data["OnlineBackup"]],
        DeviceProtection_map[form_data["DeviceProtection"]],
        TechSupport_map[form_data["TechSupport"]],
        StreamingTV_map[form_data["StreamingTV"]],
        StreamingMovies_map[form_data["StreamingMovies"]],
        Contract_map[form_data["Contract"]],
        PaperlessBilling,
        PaymentMethod_map[form_data["PaymentMethod"]],
        MonthlyCharges,
        TotalCharges
    ]

    return np.array(features).reshape(1, -1)
