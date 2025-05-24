import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def cargar_datos(ruta='./Telco-Customer-Churn.csv'):
    df = pd.read_csv(ruta)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    return df

def preprocesar_datos(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.3, random_state=42), label_encoders

def entrenar_y_guardar_modelo():
    df = cargar_datos()
    (X_train, X_test, y_train, y_test), label_encoders = preprocesar_datos(df)
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, 'modelo_churn.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

def cargar_modelo():
    modelo = joblib.load('modelo_churn.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return modelo, label_encoders

def predecir(modelo, entrada):
    return modelo.predict(np.array(entrada).reshape(1, -1))[0]
