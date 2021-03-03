from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import requests
import numpy as np
import pandas as pd  
from flask import request
import pickle 
import pyreadstat
import joblib


#creating our app and API
app = Flask(__name__)
api = Api(app)

#load our trained QDA model 

xgb_path = 'xgboost.pickle'
with open(xgb_path, 'rb') as f:
    XGB = pickle.load(f)

rfc_path = 'rfc.pickle'
with open(rfc_path, 'rb') as g:
    RFC = pickle.load(g)

lr_path = 'lr.pickle'
with open(lr_path, 'rb') as h:
    LOGIT = pickle.load(h)


#adding 'query' keyword to our parser 
#query will be the data input to the model
parser = reqparse.RequestParser()
parser.add_argument('query',action='append')
parser.add_argument('model')

#creating our PredictDefault class that will handle queries
#implementing our get method that will return the prediction value 

#@app_route('/')
#def

models = {#'XGB':XGB,
            'LOGIT':LOGIT,
            'RFC':RFC
            } 

class PredictDefault(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        #json attribute
        user_query = request.json['query']
        model = models.get(request.json['model'])
        print('user_query',user_query)
        print('model', model)
 
        prediction = model.predict(user_query)
        pred_proba = model.predict_proba(user_query)

        # print(prediction)
        # print(pred_proba)

        if prediction == 0:
            pred_text = 'No Churn'
        else:
            pred_text = 'Churn'
            
        # round the predict proba value and set to new variable

        confidence = list(np.round(pred_proba[0], 3))

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output


#creating another endpoint to our API

api.add_resource(PredictDefault, '/')
  
# example of another endpoint
#api.add_resource(PredictRatings, '/ratings')


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0') 


