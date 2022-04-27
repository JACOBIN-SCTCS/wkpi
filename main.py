import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from WKPI import *
import os
from sklearn import preprocessing
import glob

def main():

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
        # Get the list of all persistence diagrams with each element a numpy array containing the coordinates of the persistence points
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in range(dataset_size)]
        # A numpy array storing the persistence images shape = Number of persistence images * Dimension of each persistence image vector.
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in range(dataset_size)])
    else:
        # Working on a reduced set in the collab dataset
        files = [i for i in range(1,2001)]
        files = files + [i  for i in range(2601,3102)]
        files = files + [i for i in range(3376,4500)]

        # Get the list of all persistence diagrams with each element a numpy array containing the coordinates of the persistence points
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in files]
        # A numpy array storing the persistence images shape = Number of persistence images * Dimension of each persistence image vector.
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in files])
    

    # Load the class labels as a numpy array of size equal to the number of data points
    labels = np.loadtxt(pimage_path+"labels.txt")
    labels = np.array([int(labels[i]) for i in range(dataset_size)])[files]
    # Number of classes would be the number of unique labels assigned
    num_classes = len(set(labels.tolist()))
    # Get the persistence image coordinates as a numpy array
    # Array of dimensions 400*2 (400 is the number of persistence image cells)
    coordinates = np.loadtxt(pimage_path + "coordinates.txt")

    # HyperParameter which we can set 
    k = 3        #The number of gaussians in the GMM concerning the weights
    sigma = 0.1  # The standard deviation of each gaussian mixture in the GMM model.



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
    main()
    


