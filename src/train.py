import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_and_clean_data, get_preprocessor

def train_pipeline():
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_and_clean_data(data_path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = ImbPipeline([
        ('preprocessor', get_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    print("Training XGBoost Pipeline...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/churn_xgb_pipeline.pkl")
    print("\nSaved trained pipeline artifact to models/churn_xgb_pipeline.pkl")

if __name__ == "__main__":
    train_pipeline()