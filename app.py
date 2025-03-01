from flask import Flask, request, jsonify, render_template
import joblib

# Load the trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form['text']  # Get input text
        transformed_input = vectorizer.transform([symptoms])
        
        if transformed_input.shape[1] != model.n_features_in_:
            return jsonify({"error": f"Feature mismatch: expected {model.n_features_in_}, got {transformed_input.shape[1]}"})
        
        prediction = model.predict(transformed_input)[0]

        return render_template("index.html", prediction=prediction, user_input=symptoms)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
