from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NBModel

app = Flask(__name__)
api = Api(app)

# create new model object
model = NBModel()

# load trained classifier
model_path = 'nbclassifier.pkl'
with open(model_path, 'rb') as f:
    model.model = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query',type=float,action='append')

class PredictIris(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        print("this is the fikin user query ",user_query)
        user_query=np.array(user_query).reshape(1,-1)
        
        prediction = model.predict(user_query)
   
        # Output 'Negative' or 'Positive' along with the score
        if prediction == 1:
            pred_text = 'setosa'
        elif prediction == 2:
            pred_text = 'versicolor'
        else:
            pred_text = 'virginica'
            
        # create JSON object
        output = {'prediction': pred_text}
        
        return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictIris, '/')

if __name__ == '__main__':
    app.run(debug=True)