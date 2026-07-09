from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("titanic_model.pkl")

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Titanic</title></head>
    <body>
        <h1>Titanic Prediction</h1>
        <form action="/predict_form" method="POST">
            <label>Pclass:</label>
            <input type="number" name="pclass"><br><br>
            <label>Age:</label>
            <input type="number" name="age"><br><br>
            <label>Fare:</label>
            <input type="number" name="fare"><br><br>
            <label>Sex (0=male, 1=female):</label>
            <input type="number" name="sex"><br><br>
            <label>SibSp:</label>
            <input type="number" name="sibsp"><br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    passenger = np.array([[data["pclass"], data["age"], data["fare"], data["sex"], data["sibsp"]]])
    prediction = model.predict(passenger)
    result = "Live" if prediction[0] == 1 else "Dead"
    return jsonify({"result": result})

@app.route("/predict_form", methods=["POST"])
def predict_form():
    pclass = int(request.form["pclass"])
    age = int(request.form["age"])
    fare = int(request.form["fare"])
    sex = int(request.form["sex"])
    sibsp = int(request.form["sibsp"])
    passenger = np.array([[pclass, age, fare, sex, sibsp]])
    prediction = model.predict(passenger)
    result = "Live" if prediction[0] == 1 else "Dead"
    return f"<h1>نتیجه: {result}</h1>"

if __name__ == "__main__":
    app.run(debug=True)