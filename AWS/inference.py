import os
import joblib
import json
import pandas as pd

def model_fn(model_dir):
    # Load the pipeline and model dict
    obj = joblib.load(os.path.join(model_dir, "aki_pipeline_and_model.joblib"))
    return obj['preprocessor'], obj['model']

def input_fn(request_body, content_type):
    # Expect JSON array of records: {"inputs": [ {col1: val, …}, … ]}
    data = json.loads(request_body)
    return pd.DataFrame(data["inputs"])

def predict_fn(input_df, model_objs):
    preprocessor, model = model_objs
    X = preprocessor.transform(input_df)
    probs = model.predict_proba(X)[:, 1]
    return probs

def output_fn(prediction, content_type):
    # Return JSON with probabilities list
    return json.dumps({"probabilities": prediction.tolist()})
