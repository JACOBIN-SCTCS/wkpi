from copyreg import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from WKPI import *
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt

def main():

    # Check if we are working on the Collab Dataset
    collab = False
    choice = 0
    print("Enter the Dataset which you want to operate on")
    print("1. MUTAG Protein Dataset")
    print("2. Collab Input")
    print("3. CSA Dataset")
    ch = int(input())

    if ch == 1: 
        pdiagram_path="mutagPD/"
        pimage_path="mutagPI/"
    elif ch==2:
        collab=True
        pdiagram_path="collab_input/"
        pimage_path="collab_input/"
    else:
        pdiagram_path="iisc_csa_input/"
        pimage_path="iisc_csa_input/"

    # The Number of Data points 
    dataset_size = len(glob.glob1(pdiagram_path,"*_PD.pdg"))
 

    if ch==1:
        # Get the list of all persistence diagrams with each element a numpy array containing the coordinates of the persistence points
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in range(dataset_size)]
        # A numpy array storing the persistence images shape = Number of persistence images * Dimension of each persistence image vector.
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in range(dataset_size)])
        # Load the class labels as a numpy array of size equal to the number of data points
        labels = np.loadtxt(pimage_path+"labels.txt")
        labels = np.array([int(labels[i]) for i in range(dataset_size)])
    
    elif ch==2:
        # Working on a reduced set in the collab dataset
        files =[]
        files = [i for i in range(1,1500)]
        files = files + [i  for i in range(2601,3102)]
        files = files + [i for i in range(3376,4000)]

        # Get the list of all persistence diagrams with each element a numpy array containing the coordinates of the persistence points
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in files]
        # A numpy array storing the persistence images shape = Number of persistence images * Dimension of each persistence image vector.
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in files])
        # Load the class labels as a numpy array of size equal to the number of data points
        labels = np.loadtxt(pimage_path+"labels.txt")
        labels = np.array([int(labels[i]) for i in range(dataset_size)])[files]
    
    else:
         # Get the list of all persistence diagrams with each element a numpy array containing the coordinates of the persistence points
        persistence_points = [np.loadtxt(pdiagram_path + str(i) + "_PD.pdg") for i in range(1,dataset_size+1)]
        # A numpy array storing the persistence images shape = Number of persistence images * Dimension of each persistence image vector.
        persistence_images = np.array([np.loadtxt(pimage_path + str(i) + "_PI.pdg") for i in range(1,dataset_size+1)])
        # Load the class labels as a numpy array of size equal to the number of data points
        labels = np.loadtxt(pimage_path+"labels.txt")
        labels = np.array([int(labels[i]) for i in range(dataset_size)])
    


    # Number of classes would be the number of unique labels assigned
    num_classes = len(set(labels.tolist()))
    # Get the persistence image coordinates as a numpy array
    # Array of dimensions 400*2 (400 is the number of persistence image cells)
    coordinates = np.loadtxt(pimage_path + "coordinates.txt")

    # HyperParameter which we can set 
    k = 3        #The number of gaussians in the GMM concerning the weights
    sigma = 0.1  # The standard deviation of used while computing the kernel
    initial_gaussian_weights = [1.0]*k  #The initial guess for the weights given to each gaussian.
    initial_sigma_for_weights = [0.1]*k # The Standard deviation given ot each gaussian in the weight GMM
    num_epochs=15  # The number of epochs to perform training
    learning_rate=0.999 # The size of the updates to be made in gradient descent.

    indices = np.arange(len(labels))
    # Use Sklearns train_test_split to split the dataset into training and testing
    pimages_train, pimages_test,label_train,label_test,train_index,test_index  = train_test_split(persistence_images,labels,indices,test_size=0.33,shuffle=True,stratify=labels)
    pdiagram_train = [persistence_points[e] for e in train_index.tolist()]

    # Find the persistence diagram points used for training
    # This will be used in finding the centers of the K gaussians. 
    persistencePoints_train = []
   

    for pdiagram in pdiagram_train:
        persistencePoints_train = persistencePoints_train + pdiagram.tolist()
   
    
    # Normalize the values in the persistence image
    normalizer = preprocessing.Normalizer().fit(pimages_train) 
    pimages_train_normalized = normalizer.transform(pimages_train)
    pimages_test_normalized = normalizer.transform(pimages_test)

    # Perform K Means to get the intial guess of the k centers of the gaussian
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(persistencePoints_train)
    centers = kmeans.cluster_centers_
    # Random Initial weights for the gaussians
    weights = np.array(initial_gaussian_weights)
    # The standard deviation for each 
    sigma_for_weights = np.array(initial_sigma_for_weights)
    # Store the Labels of persistence images
    labellist = [np.where(label_train==i)[0] for i in range(num_classes)]
    
    # Get the final weights,gaussian centers and sigma
    weights,centers,sigma_for_weight = train(pimages_train_normalized,coordinates,labellist,num_classes,weights,centers,sigma_for_weights,sigma,num_epochs=num_epochs,lr=learning_rate)
    
    # Save the model paramters
    labellist_numpy = np.array(labellist,dtype='object')
    np.save("saved_parameters/weights",weights)
    np.save("saved_parameters/centers",centers)
    np.save("saved_parameters/sigma_for_weight",sigma_for_weight)
    np.save("saved_parameters/pimages_train",pimages_train_normalized)
    np.save("saved_parameters/label_train",label_train)
    np.save("saved_parameters/labellist",labellist_numpy,allow_pickle=True)

    #Initialize new WKPI with the parameters learned from training.     
    wkpi = WKPI(pimages_train_normalized, coordinates,labellist,num_classes)
    # Compute the weight of each persistence image cell
    wkpi.computeWeight(weights,centers,sigma_for_weight)
    # Compute the train Gram Matrix
    train_gram_matrix = wkpi.GramMatrix(sigma)
    # Compute the test gram matrix which will be used by the SVM class in Sklearn
    test_gram_matrix = wkpi.computeTestGramMatrix(pimages_test_normalized ,sigma)
    # Sklearn 
    clf = SVC(kernel='precomputed')
    # Fit the SVM based on the Gram Matrix supplied
    clf.fit(train_gram_matrix,label_train)
    # Find the labels which are predicted
    label_pred = clf.predict(test_gram_matrix)
    # Compute the accuracy of  the learned SVM Model.
    result = accuracy_score(label_test, label_pred)*100.0
    print("Accuracy = " + str(result)+"%")
    plt.title("k= "+ str(k) + "\n Kernel Sigma = " + str(sigma) +"\n " + "Accuracy = " + str(result) + "%" )
    plt.xlabel("Cost")
    plt.ylabel("Epochs")
    plt.show()

if __name__=="__main__":
    main()
    


