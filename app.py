from flask import Flask, render_template, request, g, send_file, url_for

app = Flask(__name__)

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

@app.route("/modelo")
def model():
    return render_template("modelo.html")