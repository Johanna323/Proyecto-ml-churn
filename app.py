from flask import Flask, render_template, request, redirect, g, send_file, url_for
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

UPLOAD_FOLDER = 'upload-files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load('churnModel/trainedModel.pkl')
scaler = joblib.load('churnModel/scalerModel.pkl')


@app.route("/")
def home():
    return render_template("fase-uno.html")


@app.route("/fase-uno")
def phaseOne():
    return render_template("fase-uno.html")


@app.route("/fase-dos")
def phaseTwo():
    return render_template("fase-dos.html")


@app.route("/fase-tres")
def phaseThree():
    return render_template("fase-tres.html")


@app.route("/fase-cuatro")
def phaseFour():
    return render_template("fase-cuatro.html")


@app.route("/fase-cinco")
def phaseFive():
    return render_template("fase-cinco.html")


@app.route('/modelo-regresion-logistica', methods=['GET', 'POST'])
def subir_archivo():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()

            if ext == '.csv':
                df = pd.read_csv(file, sep=';')
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            else:
                return "Formato de archivo no soportado", 400

            columnas = df.columns.tolist()
            df.to_csv('archivo_temporal.csv', index=False)

            return render_template('regresion-logistica/resultados-regresion.html', columnas=columnas)

    return render_template('regresion-logistica/cargar-archivo.html')

def plot_churn_by_column(df, column_name, output_path):
    if column_name in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=column_name,
                      hue='Predicción_Churn', palette='Set2')
        plt.title(f"Churn por {column_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_churn_ratio(df, output_path):
    plt.figure(figsize=(5, 5))
    df['Predicción_Churn'].value_counts().plot.pie(
        autopct='%1.1f%%',
        labels=['No', 'Sí'],
        colors=['skyblue', 'orange']
    )
    plt.title("Distribución de Churn")
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_tenure_histogram(df, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Predicción_Churn',
                 multiple='stack', palette='Set2', bins=20)
    plt.title("Distribución de Tenure vs Churn")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def churn_summary_by_column(df, column_name):
    return pd.crosstab(df[column_name], df['Predicción_Churn']).to_dict()


@app.route('/predecir-modelo', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df_user = pd.read_csv(file) if file.filename.endswith(
                '.csv') else pd.read_excel(file)

            df_original = df_user.copy()

            df_user['TotalCharges'] = pd.to_numeric(
                df_user['TotalCharges'], errors='coerce')
            df_user.dropna(inplace=True)
            df_user.drop(['customerID', 'Churn'], axis=1,
                         errors='ignore', inplace=True)
            df_original = df_user.copy()
            df_user = pd.get_dummies(df_user)

            df_user = df_user.reindex(
                columns=model.feature_names_in_, fill_value=0)

            df_user_scaled = pd.DataFrame(
                scaler.transform(df_user), columns=df_user.columns)
            predicciones = model.predict(df_user_scaled)

            df_user['Predicción_Churn'] = predicciones
            df_original['Predicción_Churn'] = predicciones
            df_original['Predicción_Churn'] = df_original['Predicción_Churn'].map({
                                                                                  0: 'No', 1: 'Sí'})

            columnas_vista = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService',
                  'StreamingTV', 'StreamingMovies', 'Predicción_Churn']
            tabla_individual = df_original[columnas_vista].to_dict(orient='records')
                        
            churn_counts = df_user['Predicción_Churn'].value_counts().to_dict()
            churn_si = churn_counts.get(1, 0)
            churn_no = churn_counts.get(0, 0)

            contract_summary = churn_summary_by_column(df_original, 'Contract')
            internet_summary = churn_summary_by_column(df_original, 'InternetService')
            phone_summary = churn_summary_by_column(df_original, 'PhoneService')
            streaming_tv_summary = churn_summary_by_column(df_original, 'StreamingTV')
            streaming_movies_summary = churn_summary_by_column(df_original, 'StreamingMovies')
        
            os.makedirs('static/modelImages', exist_ok=True)
            plot_churn_by_column(df_original, 'gender',
                                 'static/modelImages/grafica_genero.png')
            plot_churn_by_column(df_original, 'PaymentMethod',
                                 'static/modelImages/grafica_pago.png')
            plot_churn_ratio(
                df_original, 'static/modelImages/grafica_resumen.png')
            plot_churn_by_column(df_original, 'Contract',
                                 'static/modelImages/grafica_contract.png')
            plot_churn_by_column(df_original, 'InternetService',
                                 'static/modelImages/grafica_internet.png')
            plot_churn_by_column(df_original, 'PhoneService',
                                 'static/modelImages/grafica_phone.png')
            plot_churn_by_column(df_original, 'StreamingTV',
                                 'static/modelImages/grafica_streamingtv.png')
            plot_churn_by_column(df_original, 'StreamingMovies',
                                 'static/modelImages/grafica_streamingmovies.png')
            plot_tenure_histogram(
                df_original, 'static/modelImages/grafica_tenure.png')

            return render_template('modelo.html', tabla=tabla_individual, churn_si=churn_si, churn_no=churn_no, contract_summary=contract_summary, internet_summary=internet_summary, phone_summary=phone_summary, streaming_tv_summary=streaming_tv_summary, streaming_movies_summary=streaming_movies_summary)

        return 'Archivo no válido'
    return redirect('/modelo-regresion-logistica')
