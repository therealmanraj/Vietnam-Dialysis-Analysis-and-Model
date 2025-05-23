import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


df = pd.read_excel('data.xls')


df.rename(columns={
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
}, inplace=True)


df.drop(columns=['Unnamed: 25', 'STT'],inplace=True)


scaling_factors = {
    "Procalcitonin": 1000,          
    "White Blood Cell Count": 10,   
    "Creatinine": 88.4,             
    "Urea": 2.14,                   
    "Bilirubin": 17.1,              
    "Albumin": 10                   
}

for col, factor in scaling_factors.items():
    df[col] = df[col] / factor



different_distribution_features = [
    "SOFA",  
    "APACHEII",  
    "Mean Arterial Pressure",  
    "Lactate",  
    "Creatinine",  
    "Procalcitonin",  
    "Bilirubin",  
    "White Blood Cell Count",  
]


columns_to_impute = ["Respiratory Rate", "Albumin", 'Bilirubin', 'Procalcitonin', 'HCO3']

df[columns_to_impute] = df.groupby(["Gender", "Hypertension", "Outcome of acute kidney injury"])[columns_to_impute].transform(lambda x: x.fillna(x.median()))


df_transformed = df.copy()

log_transform_cols = ['Procalcitonin', 'Creatinine', 'Urea', 'Lactate', 'HCO3', 'Mean Arterial Pressure']
sqrt_transform_cols = ['White Blood Cell Count', 'APACHEII', 'SOFA']

for col in log_transform_cols + sqrt_transform_cols:
    if col in log_transform_cols:
        df_transformed[col] = np.log1p(df[col])
        df[col] = np.log1p(df[col])
        
    if col in sqrt_transform_cols:
        df_transformed[col] = np.sqrt(df[col].clip(lower=0))
        df[col] = np.sqrt(df[col].clip(lower=0))


def remove_outliers(df):
    cols = ['Mechanical Ventilation', 'Procalcitonin', 'Creatinine',
            'Bilirubin', 'White Blood Cell Count']

    df_clean = df.copy()

    for col in cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 2.5 * iqr
        upper = q3 + 2.5 * iqr
        
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean



df1 = df.iloc[:98, :].copy()
df2 = df.iloc[99:210, :].copy()
df3 = df.iloc[211:411, :].copy()
df4 = df.iloc[412:531, :].copy()


df1 = remove_outliers(df1)
df2 = remove_outliers(df2)
df3 = remove_outliers(df3)
df4 = remove_outliers(df4)

df = pd.concat([df1,df2,df3,df4])

X = df[
    [
            'Glasgow',
            'Mean Arterial Pressure', 
            'SOFA', 'APACHEII', 'pH', 'HCO3',  'Urea', 'Creatinine',
            'Procalcitonin', 'Bilirubin', 'Albumin', 'White Blood Cell Count', 
    ]
]

y = df['Outcome of acute kidney injury']


categorical_cols = X.select_dtypes(include=['object']).columns

X[categorical_cols] = X[categorical_cols].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


param_grid = {
        'colsample_bytree': 0.7,
        'gamma': 0,
        'learning_rate': 0.2,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 200,
        'subsample': 0.7
    }


best_model = XGBClassifier(enable_categorical=True, random_state=42)
best_model.set_params(**param_grid)
best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)


y_pred = best_model.predict(X_test)

y_scores = best_model.predict_proba(X_test)[:, 1]

threshold_recall = 0.33  
y_pred_adjusted = (y_scores >= threshold_recall).astype(int)