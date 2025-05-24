from flask import Flask, render_template, request, g, send_file, url_for
import os
import pandas as pd
import modelo_churn
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = 'upload-files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load('churnModel/modelo_entrenado.pkl')
scaler = joblib.load('churnModel/escalador.pkl')

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