
# coding: utf-8

# # Overview of classfication alorithms for Iris dataset


from model import NBModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import argparse
import pandas as pd


def build_model():
    model=NBModel()
    # load the Iris dataset and perform a training and testing split,
    # using 75% of the data for training and 25% for evaluation
    print("[INFO] loading data...")
    dataset = load_iris()
    (trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.25)
    
    # train the model
    print("Training..")
    model.train(trainX, trainY)
    print("Complete")
    
    model.pickle_clf()
    
    # make predictions on our data and show a classification report
    model.clf_report(testX,testY)
    
    
if __name__=="__main__":
    build_model()
    

