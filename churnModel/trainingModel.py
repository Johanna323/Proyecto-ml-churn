import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Cargar y limpiar datos
df = pd.read_csv('../Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# 2. Convertir variable objetivo a numérica
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# 3. Separar variable objetivo antes de codificar
y = df['Churn']
X = df.drop('Churn', axis=1)

# 4. Codificación one-hot
X = pd.get_dummies(X)

# 5. Escalado
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 6. Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 7. Guardar modelo y scaler
joblib.dump(model, 'trainedModel.pkl')
joblib.dump(scaler, 'scalerModel.pkl')

# 8. Graficar matriz de confusión
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='gray')
plt.savefig('../static/modelImages/grafica.png')