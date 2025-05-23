# train.py

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import joblib

# 1. Rename columns
def rename_columns(df):
    return df.rename(
        columns={
            'Chieucao': 'Height',
            'Cannang': 'Weight',
            'Duongvao': 'Route of Entry',
            'THA': 'Hypertension',
            'DTD': 'Diabetes',
            'Thomay': 'Mechanical Ventilation',
            'Mach': 'Pulse',
            'Nhietdo': 'Temperature',
            'HATB': 'Mean Arterial Pressure',
            'Nhiptho': 'Respiratory Rate',
            'Lactac0': 'Lactate',
            'Ure': 'Urea',
            'Creatinin': 'Creatinine',
            'PCT0': 'Procalcitonin',
            'BiLIrubin': 'Bilirubin',
            'BC0': 'White Blood Cell Count',
            'Kết cục tổn thương thận cấp':'Outcome of acute kidney injury',
            'Điều trị lọc máu':'Dialysis treatment'
        }
    )

# 2. Drop unwanted columns
def drop_columns(df):
    return df.drop(columns=['Unnamed: 25', 'STT'], errors='ignore')

# 3. Scale selected lab values
def scale_features(df):
    df = df.copy()
    factors = {
        "Procalcitonin": 1000,
        "White Blood Cell Count": 10,
        "Creatinine": 88.4,
        "Urea": 2.14,
        "Bilirubin": 17.1,
        "Albumin": 10
    }
    for c,f in factors.items():
        if c in df:
            df[c] = df[c] / f
    return df

# 4. Group‐median imputation
def impute_features(df):
    df = df.copy()
    cols = ["Respiratory Rate", "Albumin", "Bilirubin", "Procalcetin", "Procalcitonin", "HCO3"]
    # fix typo if needed: 'Procalcetin' vs. 'Procalcitonin'
    cols = [c for c in cols if c in df]
    df[cols] = (
        df.groupby(["Gender", "Hypertension", "Outcome of acute kidney injury"])[cols]
          .transform(lambda x: x.fillna(x.median()))
    )
    return df

# 5. Log and square‐root transforms
def transform_features(df):
    df = df.copy()
    log_cols = [c for c in [
        'Procalcitonin','Creatinine','Urea','Lactate','HCO3','Mean Arterial Pressure'
    ] if c in df]
    sqrt_cols = [c for c in [
        'White Blood Cell Count','APACHEII','SOFA'
    ] if c in df]
    for c in log_cols:
        df[c] = np.log1p(df[c].clip(lower=0))
    for c in sqrt_cols:
        df[c] = np.sqrt(df[c].clip(lower=0))
    return df

# 6. Remove outliers (across the whole set)
def remove_outliers(df):
    df = df.copy()
    cols = [
        'Mechanical Ventilation','Procalcitonin','Creatinine',
        'Bilirubin','White Blood Cell Count'
    ]
    for c in cols:
        if c in df:
            q1, q3 = df[c].quantile([0.25,0.75])
            iqr = q3 - q1
            low, high = q1 - 2.5*iqr, q3 + 2.5*iqr
            df = df[(df[c] >= low) & (df[c] <= high)]
    return df

# 7. Select final feature matrix
FEATURE_COLS = [
    'Glasgow','Mean Arterial Pressure','SOFA','APACHEII',
    'pH','HCO3','Urea','Creatinine','Procalcitonin','Bilirubin',
    'Albumin','White Blood Cell Count'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in FEATURE_COLS]  # ensure names match

def select_features(df):
    return df[FEATURE_COLS]

# Build the pipeline
pipeline = Pipeline([
    ('rename',      FunctionTransformer(rename_columns, validate=False)),
    ('drop',        FunctionTransformer(drop_columns, validate=False)),
    ('scale',       FunctionTransformer(scale_features, validate=False)),
    ('impute',      FunctionTransformer(impute_features, validate=False)),
    ('transform',   FunctionTransformer(transform_features, validate=False)),
    ('outlier',     FunctionTransformer(remove_outliers, validate=False)),
    ('select',      FunctionTransformer(select_features, validate=False)),
    ('clf',         XGBClassifier(
                        colsample_bytree=0.7,
                        gamma=0,
                        learning_rate=0.2,
                        max_depth=3,
                        min_child_weight=1,
                        n_estimators=200,
                        subsample=0.7,
                        enable_categorical=True,
                        random_state=42
                   ))
])

if __name__ == "__main__":
    # 1. Load raw data
    df = pd.read_excel("data.xls")

    # 2. Extract label
    # y = df["Outcome of acute kidney injury"]
    y = df["Kết cục tổn thương thận cấp"]
    
    # 3. Train end‐to‐end
    pipeline.fit(df, y)

    # 4. Save the entire pipeline
    joblib.dump(pipeline, "aki_pipeline.joblib")
    print("✅ Pipeline trained and saved as aki_pipeline.joblib")
