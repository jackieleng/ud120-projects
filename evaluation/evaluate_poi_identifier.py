#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, random_state=42, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

preds = clf.predict(X_test)
print "Number of predicted POIs in test set %s" % sum(preds)
print "Size test set %s" % len(X_test)
print "Accuracy %s" % ((1.* len(y_test) - sum(y_test)) / len(y_test))
print "Number of true postives %s" % sum((y_test == 1) & (y_test == preds))
print "Precision %s" % precision_score(y_test, preds)
print "Recall %s" % recall_score(y_test, preds)
print confusion_matrix(y_test, preds)

# Quiz values
print "\nQuiz values:"
predictions = [0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,1,0,1,0,1]
true_labels = [0,0,0,0,0,0,1,0,1,1,0,1,0,1,1,1,0,1,0,0]
print confusion_matrix(true_labels, predictions)
print "Precision (Tp/Tp+Fp) %s" % precision_score(true_labels, predictions)
print "Recall (Tp/Tp + Fn) %s" % recall_score(true_labels, predictions)
