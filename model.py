
# coding: utf-8

# # Overview of classfication alorithms for Iris dataset


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pandas as pd
import pickle


class NBModel(object):
    
    def __init__(self):
        self.model=GaussianNB()

    def train(self, X,y):
        # train the model
        print("Training  using Gaussian Naive Bayes model")
        self.model.fit(X, y)

    def predict(self,X):
        # make predictions on our data and show a classification report
        print("[INFO] evaluating...")
        y_pred = self.model.predict(X)
        return y_pred
    
    def pickle_clf(self,path='nbclassifier.pkl'):
        with open(path,'wb') as f:
            pickle.dump(self.model,f)
            print('Pickled classifier at {}'.format(path))
            
    def clf_report(self,X,y):
        print("[INFO] preparing report...")
        y_pred = self.model.predict(X)
        print(classification_report(y, y_pred))

        #print(classification_report(testY, predictions,target_names=dataset.target_names))

