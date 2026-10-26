import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_and_prep_data(filepath="data/data.csv"):
    """Carrega os dados e realiza a limpeza inicial idêntica à Fase 1."""
    df = pd.read_csv(filepath)
    # Remove colunas desnecessárias
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    # Converte o diagnóstico para numérico (M = 1, B = 0)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    
    # Preenchimento de NAs de segurança, caso existam
    df = df.dropna()
    
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_logistic_pipeline(C=1.0, class_weight=None, solver='lbfgs'):
    """Constrói o Pipeline da Regressão Logística com StandardScaler integrado."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=C, 
            class_weight=class_weight, 
            solver=solver, 
            random_state=42, 
            max_iter=2000
        ))
    ])

def build_rf_pipeline(n_estimators=100, max_depth=None, min_samples_split=2):
    """Constrói o Pipeline do Random Forest com StandardScaler integrado."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        ))
    ])
