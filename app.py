# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# from sklearn.svm import SVC

# app = Flask(__name__)

# with open('svc_model.pkl', 'rb') as pickle_file:
#     model = pickle.load(pickle_file)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     tweet = (request.form['tweet'])

#     input_data = np.array([[tweet]])
#     result = model.predict(input_data)[0]

#     return render_template('result.html', prediction=result)

# if __name__ == '_main_':
#     app.run(debug=True)
# app.py

# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('svc_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']

    input_data = np.array([[tweet]])
    result = model.predict(input_data)[0]

    return render_template('index.html', prediction=result, input_tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
