import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from WKPI import *
import sys

sigma = 0.1
weights = np.load("saved_parameters/weights.npy")
centers = np.load("saved_parameters/centers.npy")
sigma_for_weight = np.load("saved_parameters/sigma_for_weight.npy")
coordinates = np.loadtxt("iisc_csa_input/coordinates.txt")
pimages_train_normalized = np.load("saved_parameters/pimages_train.npy")
labels = np.load("saved_parameters/label_train.npy")
labellist = np.load("saved_parameters/labellist.npy",allow_pickle=True)


pimages_test_normalized = np.array([np.loadtxt(sys.argv[1] + "_PI.pdg")])
num_classes = len(set(labels.tolist()))
wkpi = WKPI(pimages_train_normalized, coordinates,labellist,num_classes)
wkpi.computeWeight(weights,centers,sigma_for_weight)
train_gram_matrix = wkpi.GramMatrix(sigma)
test_gram_matrix = wkpi.computeTestGramMatrix(pimages_test_normalized ,sigma)
clf = SVC(kernel='precomputed')
clf.fit(train_gram_matrix,labels)
label_pred = clf.predict(test_gram_matrix)[0]

if(label_pred==0):
    print("Professor " + sys.argv[1] + " is inclined to Pool A (Theory)")
elif(label_pred==1):
    print("Professor " + sys.argv[1] + " is inclined to Pool B (Systems)")
else:
    print("Professor " + sys.argv[1] + " is inclined to Pool C (Intelligent Systems)")


    

