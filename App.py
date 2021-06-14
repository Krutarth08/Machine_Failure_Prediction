import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('UI.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = list()
    int_features.append(request.form.get("temp"))
    int_features.append(request.form.get("humid"))
    print(int_features)
    final_features = np.array(int_features).reshape(1,-1)
    print(final_features)
    prediction = model.predict(final_features)
    
    if prediction[0]==0:
        output = "No"
    else:
        output = "Yes"

    return render_template('UI.html', prediction_text='Machine Failure Prediction : {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json()
    data = json.loads(data)
    data_req = list()
    data_req.append(data["Temperature"])
    data_req.append(data["Humidity"])
    
    prediction = model.predict(np.array(data_req).reshape(1,-1))

    output = prediction[0]

    if output==0:
        output = "No"
    else:
        output = "Yes"

    return json.dumps({"result":output})
if __name__ == "__main__":
    app.run(debug=True)    