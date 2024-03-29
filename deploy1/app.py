from flask import Flask, render_template, jsonify,  request
import pickle as pkl
import numpy as np




#initialize the flask app
app = Flask(__name__)
model=pickle.load(open('model1.pkl','rb'))

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features=[float(i) for i in request.form.values()]
    final_features=[np.array(input_features)]
    prediction=model.predict(final_features)
    return render_template('index.html',prediction_text='Prediction :{}'.format(prediction))
    
  

if __name__ == "__main__":
    app.run(debug=True)