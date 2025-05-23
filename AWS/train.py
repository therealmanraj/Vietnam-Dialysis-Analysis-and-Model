# train.py

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# ── 1. Raw → cleaned DataFrame ────────────────────────────────────────────

def rename_columns(df):
    return df.rename(columns={
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
        'Kết cục tổn thương thận cấp': 'Outcome of acute kidney injury',
        'Điều trị lọc máu': 'Dialysis treatment'
    })

def drop_columns(df):
    return df.drop(columns=['Unnamed: 25', 'STT'], errors='ignore')

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
    for c, f in factors.items():
        if c in df:
            df[c] = df[c] / f
    return df

def impute_features(df):
    df = df.copy()
    cols = ["Respiratory Rate", "Albumin", "Bilirubin", "Procalcitonin", "HCO3"]
    cols = [c for c in cols if c in df]
    df[cols] = (
        df.groupby(
            ["Gender", "Hypertension", "Outcome of acute kidney injury"]
        )[cols]
        .transform(lambda x: x.fillna(x.median()))
    )
    return df

def transform_features(df):
    df = df.copy()
    log_cols = [
        c for c in ['Procalcitonin','Creatinine','Urea','Lactate','HCO3','Mean Arterial Pressure']
        if c in df
    ]
    sqrt_cols = [
        c for c in ['White Blood Cell Count','APACHEII','SOFA']
        if c in df
    ]
    for c in log_cols:
        df[c] = np.log1p(df[c].clip(lower=0))
    for c in sqrt_cols:
        df[c] = np.sqrt(df[c].clip(lower=0))
    return df

FEATURE_COLS = [
    'Glasgow','Mean Arterial Pressure','SOFA','APACHEII',
    'pH','HCO3','Urea','Creatinine',
    'Procalcitonin','Bilirubin','Albumin','White Blood Cell Count'
]

def select_features(df):
    return df[FEATURE_COLS]

# Build the *preprocessing* pipeline (no outlier‐removal here!)
preprocess = Pipeline([
    ('rename',    FunctionTransformer(rename_columns, validate=False)),
    ('drop',      FunctionTransformer(drop_columns, validate=False)),
    ('scale',     FunctionTransformer(scale_features, validate=False)),
    ('impute',    FunctionTransformer(impute_features, validate=False)),
    ('transform', FunctionTransformer(transform_features, validate=False)),
    ('select',    FunctionTransformer(select_features, validate=False)),
])

# ── 2. Outlier removal (must happen *after* we have X & y aligned) ────────

def remove_outliers(X_df, y_series):
    df = X_df.copy()
    df['_y_'] = y_series.values
    for c in ['Mechanical Ventilation','Procalcitonin','Creatinine','Bilirubin','White Blood Cell Count']:
        if c in df:
            q1, q3 = df[c].quantile([0.25,0.75])
            iqr = q3 - q1
            low, high = q1 - 2.5*iqr, q3 + 2.5*iqr
            df = df[(df[c] >= low) & (df[c] <= high)]
    y_clean = df.pop('_y_')
    return df, y_clean

# ── 3. Train / save ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # a) Load raw and rename so we can extract y in English
    df_raw = pd.read_excel("data.xls")
    df_raw = rename_columns(df_raw)

    # b) Extract labels
    y = df_raw["Outcome of acute kidney injury"]

    # c) Build feature‐matrix
    X = preprocess.fit_transform(df_raw)

    # d) Remove outliers *and* keep y in sync
    X_clean, y_clean = remove_outliers(X, y)

    # e) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # f) Fit the final model
    model = XGBClassifier(
        colsample_bytree=0.7,
        gamma=0,
        learning_rate=0.2,
        max_depth=3,
        min_child_weight=1,
        n_estimators=200,
        subsample=0.7,
        enable_categorical=True,
        random_state=42
    )
    model.fit(X_train, y_train)

    # g) Save both the preprocessing + model for production
    joblib.dump({
        'preprocessor': preprocess,
        'model': model
    }, "aki_pipeline_and_model.joblib")

    print("✅ Trained and saved aki_pipeline_and_model.joblib")