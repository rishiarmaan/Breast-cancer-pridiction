from flask import Flask, request, render_template
import pandas 
import numpy as np
import pickle 

pipeline = pickle.load(open("model.pkl", "rb"))
model = pickle.load(open("model.pkl",'rb'))
selector = pickle.load(open("feature_selector.pkl","rb"))

#flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try :
        features = [
            float(request.form['mean_perimeter']),
            float(request.form['mean_concave_points']),
            float(request.form['worst_radius']),
            float(request.form['worst_perimeter']),
            float(request.form['worst_area']),
            float(request.form['worst_concave_points'])]
        np_features = np.asarray(features).reshape(1, -1)
        prediction = pipeline.predict(np_features)

        output = ["Cancerous" if prediction[0] == 1 else "Not Cancerous"]

        return render_template('index.html', message=output)
    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")


#python main
if __name__ == "__main__": 
    app.run(debug=True)