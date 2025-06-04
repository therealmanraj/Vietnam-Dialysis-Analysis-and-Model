# inference.py

import os
import joblib
import json
import pandas as pd

def model_fn(model_dir):
    """
    SageMaker calls this once at container startup.
    We expect our joblib file to be in model_dir/xgb_aki_model.joblib
    """
    path = os.path.join(model_dir, "xgb_aki_model.joblib")
    pipeline = joblib.load(path)
    return pipeline

def input_fn(request_body, request_content_type):
    """
    Parse incoming JSON into a pandas DataFrame.
    Expect: {"inputs":[ {col1:val1, col2:val2, ...}, {...} ]}
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    return pd.DataFrame(payload["inputs"])

def predict_fn(input_df, model):
    """
    Run model.predict_proba on preprocessed DataFrame.
    Here `model` is whatever was returned from model_fn (our joblib pipeline or XGBClassifier).
    """
    # If you trained a raw XGBClassifier, `model.predict_proba` works as‐is.
    # If you used a Pipeline that already does preprocessing, these inputs must be raw.
    probs = model.predict_proba(input_df)[:, 1]
    return probs

def output_fn(prediction, response_content_type):
    """
    Return JSON with a "probabilities" array.
    """
    return json.dumps({"probabilities": prediction.tolist()})
