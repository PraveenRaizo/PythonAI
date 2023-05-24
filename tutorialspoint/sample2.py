import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#AI with python

#importing and loading breast cancer data from sklearn
data = load_breast_cancer()
print("Breast Cancer Data: ", data)

#we create 
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print("LabelNames: ",label_names)
print("Labels: ",labels[0])
print("FeatureNames: ",feature_names[0])
print("Features: ",features[0])


train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)
gnb = GaussianNB() #initialize gaussian module
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)

