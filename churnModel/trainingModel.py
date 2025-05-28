import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

# Entrenamiento del modelo CHURN
# Paso 1. Se cargan y limpian los datos
df = pd.read_csv('Telco-Customer-Churn.csv')

# Reporte de valores nulos inicial
null_report = df.isnull().sum()
null_report = null_report[null_report > 0]
if not null_report.empty:
    print("🚨 Columnas con valores nulos detectadas:\n", null_report)
    df[null_report.index].to_csv(
        'static/trainedImages/filas_con_nulos.csv', index=False)

# Limpieza de TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eliminar filas con NaN en columnas críticas
df.dropna(subset=['TotalCharges'], inplace=True)

# Eliminar columnas innecesarias
df.drop('customerID', axis=1, inplace=True)

# Paso 2. Convertir variable objetivo
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Paso 3. Separamos la variable objetivo
y = df['Churn']
df.drop('Churn', axis=1, inplace=True)

# Paso 4. Feature engineering
df['AvgMonthlySpend'] = df.apply(
    lambda row: row['TotalCharges'] /
    row['tenure'] if row['tenure'] > 0 else 0,
    axis=1
)

df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                            labels=['0-12', '13-24', '25-48', '49-72'])

services = ['PhoneService', 'InternetService', 'StreamingTV',
            'StreamingMovies', 'OnlineSecurity', 'OnlineBackup']
df['TotalServices'] = df[services].apply(lambda row: sum(row == 'Yes'), axis=1)

df['PaymentMethod_Simplified'] = df['PaymentMethod'].apply(
    lambda x: 'Automatic' if 'auto' in x.lower() else 'Manual'
)

df['Contract_Monthly_Streaming'] = df['Contract'].astype(
    str) + "_" + df['StreamingTV'].astype(str)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                  'StreamingTV', 'StreamingMovies', 'OnlineBackup', 'OnlineSecurity']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Paso 5. Definir X con nuevas variables
X = pd.get_dummies(df)

# Reemplazo de valores nulos detectados
X.fillna(0, inplace=True)

# Validación rápida para asegurar que no haya nulos antes del escalado
assert X.isnull().sum().sum() == 0, "Existen NaNs en X antes del escalado"

# Paso 6. Escalar e imputar
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Imputación por seguridad adicional
imputer = SimpleImputer(strategy='mean')
X_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X.columns)

# Validación final para asegurar que no haya NaNs después de imputar
final_nans = X_scaled.isnull().sum().sum()
if final_nans > 0:
    print(f"❌ Persisten {final_nans} valores nulos luego de imputar.")
    X_scaled.to_csv('static/trainedImages/X_con_nulos.csv', index=False)
    raise ValueError("El dataframe escalado aún contiene NaN.")
else:
    print("✅ No hay valores nulos después de imputar y escalar.")

# Validación final para asegurar que no haya NaNs
assert X_scaled.isnull().sum().sum(
) == 0, "Existen NaNs después del escalado/imputación"


# Paso 6. Entrenamos el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Entrenar el modelo con los datos balanceados
model = LogisticRegression()

param_grid = {
    'C': [0.01, 0.1, 1, 10],        # Regularización
    # Penalización (L2 es común para LogisticRegression)
    'penalty': ['l2'],
    'solver': ['liblinear'],       # Solvers compatibles
    'max_iter': [100, 200, 500]    # Iteraciones máximas
}

grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring='f1')

# Reemplazar el model vacío con GridSearch en los datos balanceados
grid_search.fit(X_train_resampled, y_train_resampled)

# Usa el mejor modelo encontrado ENTRENADO ya con SMOTE
best_model = grid_search.best_estimator_

# Paso 7. Guardamos el modelo entrenado y el scaler
joblib.dump(best_model, 'churnModel/trainedModel.pkl')
joblib.dump(scaler, 'churnModel/scalerModel.pkl')

# Recuperamos la columna 'Churn' para análisis visual
df['Churn'] = y

# Paso 8. Graficamos la matriz de confusión del entrenamiento
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión")
plt.savefig('static/trainedImages/matriz_confusion.png')
plt.close()

# Paso 10. Gráficas del entrenamiento
# -- Grafica de las variables más importantes
# Actualiza esta sección para mostrar signo y magnitud
coef = pd.Series(best_model.coef_[0], index=X.columns)
top_coef = coef.sort_values(key=abs, ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_coef.values, y=top_coef.index, palette=[
            "skyblue" if v < 0 else "salmon" for v in top_coef.values])
plt.title("Top 10 Variables más Influyentes en Churn")
plt.xlabel("Coeficiente (signo y magnitud)")
plt.ylabel("Variable")
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig("static/trainedImages/importancia_variables.png")
plt.close()


# -- Grafica de ternure vs Churn
df['tenure_grupo'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                            labels=['0-12 meses', '13-24', '25-48', '49-72'])

tenure_churn = df.groupby(['tenure_grupo', 'Churn'],
                          observed=False).size().unstack()

tenure_churn.plot(kind='bar', stacked=True, colormap='PiYG', figsize=(8, 6))
plt.title("Distribución de Churn según Tenure")
plt.ylabel("Número de Clientes")
plt.xlabel("Tenure (meses)")
plt.tight_layout()
plt.savefig("static/trainedImages/churn_tenure.png")
plt.close()

# -- Grafica de Contract vs Churn
contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()

contract_churn.plot(kind='bar', stacked=True,
                    colormap='Paired', figsize=(8, 6))
plt.title("Churn por Tipo de Contrato")
plt.ylabel("Número de Clientes")
plt.xlabel("Tipo de Contrato")
plt.tight_layout()
plt.savefig("static/trainedImages/churn_contract.png")
plt.close()

# -- Grafica de Internet Service vs Churn

internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()

internet_churn.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 6))
plt.title("Churn según Tipo de Internet")
plt.ylabel("Número de Clientes")
plt.xlabel("InternetService")
plt.tight_layout()
plt.savefig("static/trainedImages/churn_internet.png")
plt.close()

# -- Grafica de TV Streaming Service vs Churn

tv_churn = df.groupby(['StreamingTV', 'Churn']).size().unstack()

tv_churn.plot(kind='bar', stacked=True, colormap='Accent', figsize=(8, 6))
plt.title("Churn según Servicio de StreamingTV")
plt.ylabel("Número de Clientes")
plt.xlabel("StreamingTV")
plt.tight_layout()
plt.savefig("static/trainedImages/churn_streamingtv.png")
plt.close()

# -- Grafica de Phone Service vs Churn
phone_churn = df.groupby(['PhoneService', 'Churn']).size().unstack()

phone_churn.plot(kind='bar', stacked=True, colormap='Spectral', figsize=(8, 6))
plt.title("Churn según Servicio Telefónico")
plt.ylabel("Número de Clientes")
plt.xlabel("PhoneService")
plt.tight_layout()
plt.savefig("static/trainedImages/churn_phoneservice.png")
plt.close()

# Paso 11. Guardamos las métricas para presentar posteriormente en los resultados# Métricas del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

with open("static/trainedImages/metricas.txt", "w") as f:
    f.write("---- MÉTRICAS DEL MODELO ----\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1-score: {f1:.2f}\n")
    f.write(f"ROC-AUC: {roc_auc:.2f}\n")
    f.write("\nReporte completo:\n")
    f.write(classification_report(y_test, y_pred))
