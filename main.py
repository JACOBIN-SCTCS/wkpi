from cgi import test
from re import L
import datasets
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from WKPI import *
import os
from sklearn import preprocessing
import glob


'''piimages = np.array([[1.0,2.0],[2.0,3.0],[4.0,5.0],[1.0,2.0],[1.0,1.4]])
coordinates = np.array([[1.0,2.0],[0.0,0.0]])
center_gaussians= np.array([[1.0,1.0],[2.0,2.0]])
weight_gaussian = np.array([1.9,1.0])
sigma_for_weights = np.array([2.0,2.0])
labelTrain = [0,0,0,1,1]
num_classes = 2

classes = [0,0,0,1,1]
g = np.array(classes)
print(g.shape)

class_labels = [np.where(g==i)[0] for i in range(num_classes)]
print(class_labels)

#class_labels = [[0,1,2],[3,4]]
#print(class_labels)
wkpi = WKPI(piimages,coordinates,class_labels,num_classes)
#wkpi.computeWeights(weight_gaussian,center_gaussians,sigma_for_weights)
wkpi.computeWeight(weight_gaussian,center_gaussians,sigma_for_weights)

#print(wkpi.GramMatrix(1))
#print(wkpi.DistanceMetric())
#print(wkpi.computeCost())
#print(wkpi.computeGradients())





getCostandGradients(piimages,coordinates,class_labels,num_classes,weight_gaussian,center_gaussians,sigma_for_weights,1)

train(piimages,coordinates,class_labels,num_classes,weight_gaussian,center_gaussians,sigma_for_weights,1)

'''
#train(piimages,coordinates,class_labels,num_classes,weight_gaussian,center_gaussians,sigma_for_weights, 1
#)

def m():

    # Check if we are working on the Collab Dataset
    collab = True
    if not collab:
        pdiagram_path="mutagPD/"
        pimage_path="mutagPI/"
    else:
        pdiagram_path="collab_input/"
        pimage_path="collab_input/"


    files =[]
    
    # The Number of Data points 
    dataset_size = len(glob.glob1(pdiagram_path,"*_PD.pdg"))
    
    
    if not collab:
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in range(dataset_size)]
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in range(dataset_size)])
    else:
        files = [i for i in range(1,2401)]
        files = files + [i  for i in range(2601,3102)]
        files = files + [i for i in range(3376,5000)]

        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in files]
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in files])
    
    print(persistence_images.shape)
    
    labels = np.loadtxt(pimage_path+"labels.txt")
    labels = np.array([int(labels[i]) for i in range(dataset_size)])[files]
    num_classes = len(set(labels.tolist()))
    coordinates = np.loadtxt(pimage_path + "coordinates.txt")
    print(coordinates.shape)

    k = 3
    sigma = 0.1

    kf = StratifiedKFold(n_splits = 2, shuffle = True)

    scoreList = []

    for train_index,test_index in kf.split(persistence_images,labels):
        pimages_train,pimages_test = persistence_images[train_index],persistence_images[test_index]
        label_train,label_test = labels[train_index], labels[test_index]
        print(label_train.shape)
        pdiagram_train = [persistence_points[e] for e in train_index.tolist()]
        persistencePoints_train = []

        for pdiagram in pdiagram_train:
            if pdiagram.shape[0]==1:
                persistencePoints_train = persistencePoints_train + [pdiagram.tolist()]
            else:
                persistencePoints_train = persistencePoints_train + pdiagram.tolist()
        normalizer = preprocessing.Normalizer().fit(pimages_train) 
        pimages_train_normalized = normalizer.transform(pimages_train)
        pimages_test_normalized = normalizer.transform(pimages_test)

        kmeans = KMeans(n_clusters = k, random_state = 0).fit(persistencePoints_train)
        centers = kmeans.cluster_centers_
        weights = np.array([1.0]*k)
        sigma_for_weights = np.array([0.1]*k)

        labellist = [np.where(label_train==i)[0] for i in range(num_classes)]

        weights,centers,sigma = train(pimages_train_normalized,coordinates,labellist,num_classes,weights,centers,sigma_for_weights,0.1)
           
        wkpi = WKPI(pimages_train_normalized, coordinates,labellist,num_classes)
        wkpi.computeWeight(weights,centers,sigma)
        train_gram_matrix = wkpi.GramMatrix(0.1)
        test_gram_matrix = wkpi.computeTestGramMatrix(pimages_test_normalized ,0.1)
        clf = SVC(kernel='precomputed')
        clf.fit(train_gram_matrix,label_train)
        label_pred = clf.predict(test_gram_matrix)
        result = accuracy_score(label_test, label_pred)
        print("Accuracy = " + str(result))


if __name__=="__main__":
    m()
    


