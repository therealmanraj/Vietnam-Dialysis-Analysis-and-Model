import pandas as pd
import warnings

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# Ignore warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('Cleaned.csv')


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


