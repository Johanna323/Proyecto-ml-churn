import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Entrenamiento del modelo CHURN
# Paso 1. Se cargan y limpian los datos
df = pd.read_csv('../Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Paso 2. Tomamos la variable objetivo, en este caso CHURN, y la convertimos en valor numérico
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Paso 3. Separamos la variable objetivo
y = df['Churn']
X = df.drop('Churn', axis=1)

# Paso 4. Se realiza la codificación one-hot (uno a muchos)
X = pd.get_dummies(X)

# Paso 5. Escalamos las variables numéricas
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Paso 6. Entrenamos el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Paso 7. Guardamos el modelo entrenado y el scaler
joblib.dump(model, 'trainedModel.pkl')
joblib.dump(scaler, 'scalerModel.pkl')

# Paso 8. Graficamos la matriz de confusión del entrenamiento
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.savefig('../static/trainedImages/matriz_confusion.png')
plt.close()

# Paso 10. Gráficas del entrenamiento
# -- Grafica de las variables más importantes
coef = pd.Series(model.coef_[0], index=X.columns)
top_coef = coef.abs().sort_values(ascending=False).head(10)

coef = pd.Series(model.coef_[0], index=X.columns)
top_coef = coef.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_coef.values, y=top_coef.index, color="skyblue")
plt.title("Top 10 Variables más Influyentes en Churn")
plt.xlabel("Importancia (|coeficiente|)")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("../static/trainedImages/importancia_variables.png")
plt.close()

# -- Grafica de ternure vs Churn
df['tenure_grupo'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                            labels=['0-12 meses', '13-24', '25-48', '49-72'])

tenure_churn = df.groupby(['tenure_grupo', 'Churn'], observed=False).size().unstack()

tenure_churn.plot(kind='bar', stacked=True, colormap='PiYG', figsize=(8, 6))
plt.title("Distribución de Churn según Tenure")
plt.ylabel("Número de Clientes")
plt.xlabel("Tenure (meses)")
plt.tight_layout()
plt.savefig("../static/trainedImages/churn_tenure.png")
plt.close()

# -- Grafica de Contract vs Churn
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()

contract_churn.plot(kind='bar', stacked=True, colormap='Paired', figsize=(8, 6))
plt.title("Churn por Tipo de Contrato")
plt.ylabel("Número de Clientes")
plt.xlabel("Tipo de Contrato")
plt.tight_layout()
plt.savefig("../static/trainedImages/churn_contract.png")
plt.close()

# -- Grafica de Internet Service vs Churn

internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()

internet_churn.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 6))
plt.title("Churn según Tipo de Internet")
plt.ylabel("Número de Clientes")
plt.xlabel("InternetService")
plt.tight_layout()
plt.savefig("../static/trainedImages/churn_internet.png")
plt.close()

# -- Grafica de TV Streaming Service vs Churn

tv_churn = df.groupby(['StreamingTV', 'Churn']).size().unstack()

tv_churn.plot(kind='bar', stacked=True, colormap='Accent', figsize=(8, 6))
plt.title("Churn según Servicio de StreamingTV")
plt.ylabel("Número de Clientes")
plt.xlabel("StreamingTV")
plt.tight_layout()
plt.savefig("../static/trainedImages/churn_streamingtv.png")
plt.close()

# -- Grafica de Phone Service vs Churn
phone_churn = df.groupby(['PhoneService', 'Churn']).size().unstack()

phone_churn.plot(kind='bar', stacked=True, colormap='Spectral', figsize=(8, 6))
plt.title("Churn según Servicio Telefónico")
plt.ylabel("Número de Clientes")
plt.xlabel("PhoneService")
plt.tight_layout()
plt.savefig("../static/trainedImages/churn_phoneservice.png")
plt.close()

# Paso 11. Guardamos las métricas para presentar posteriormente en los resultados# Métricas del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

with open("../static/trainedImages/metricas.txt", "w") as f:
    f.write("---- MÉTRICAS DEL MODELO ----\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1-score: {f1:.2f}\n")
    f.write(f"ROC-AUC: {roc_auc:.2f}\n")
    f.write("\nReporte completo:\n")
    f.write(classification_report(y_test, y_pred))