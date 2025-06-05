# inference.py

import json
import os
import numpy as np
import xgboost as xgb
from math import log1p, sqrt
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

FEATURE_NAMES = [
    'HCO3',
    'Creatinine',
    'Procalcitonin',
    'Mean Arterial Pressure',
    'Bilirubin',
    'pH',
    'Albumin',
    'Urea',
    'White Blood Cell Count',
    'SOFA',
    'APACHEII',
    'Glasgow'
]

SCALING_FACTORS = {
    "Procalcitonin": 1000.0,
    "White Blood Cell Count": 10.0,
    "Creatinine": 88.4,
    "Urea": 2.14,
    "Bilirubin": 17.1,
    "Albumin": 10.0
}

def model_fn(model_dir):
    """
    model_fn is called by SageMaker to load the model from disk.
    It should return an object representing the loaded model.
    """
    model_path = os.path.join(model_dir, "model.xgb")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

def preprocess_payload(payload):
    """
    The payload will be a JSON dict mapping feature names to numeric strings.
    We need to:
      1) Convert to float
      2) Apply the scaling factors
      3) Apply log1p or sqrt transforms
      4) Return a 2D numpy array of shape (1, 12) to feed to XGBoost
    """

    data = json.loads(payload)
    
    raw_values = []
    for feature in FEATURE_NAMES:
        if feature not in data:
            raise ValueError(f"Missing feature in payload: {feature}")
        val = float(data[feature])
        raw_values.append(val)

    row = dict(zip(FEATURE_NAMES, raw_values))

    for col, factor in SCALING_FACTORS.items():
        if col in row:
            row[col] = row[col] / factor

    log_transform_cols = ['Procalcitonin', 'Creatinine', 'Urea', 'Mean Arterial Pressure']
    sqrt_transform_cols = ['White Blood Cell Count', 'APACHEII', 'SOFA']

    for col in log_transform_cols:
        if col in row:
            row[col] = log1p(max(row[col], 0.0))
    for col in sqrt_transform_cols:
        if col in row:
            row[col] = sqrt(max(row[col], 0.0))

    feature_vector = np.array([row[f] for f in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    return feature_vector

def predict_fn(input_data, model):
    """
    Given preprocessed input_data (numpy array) and a loaded model (XGBoost Booster),
    return the predicted probability of AKI (usually the positive‐class proba).
    """
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    proba = model.predict(dmatrix)
    risk_pct = float(proba[0] * 100.0)
    return risk_pct

def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request (JSON) into a string, so we can preprocess it.
    SageMaker will call this first.
    """
    if request_content_type == content_types.JSON:
        return decoder.decode(request_body, request_content_type)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction (a float risk score) back to JSON
    """
    if response_content_type == content_types.JSON:
        return encoder.encode({"risk_score_pct": prediction}, response_content_type)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

