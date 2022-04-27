from random import gauss
import numpy as np

class WKPI:

    def __init__(self,pimages,coordinates,class_labels,num_classes):
       
        self.pimages = pimages              # A numpy array storing the persistence images
        self.num_pimages = pimages.shape[0] # The number of persistence images.
        self.coordinates = coordinates      # The coordinates of each persistence image cell.
        self.coordinate_length = self.coordinates.shape[0]  # The number of dimensions when the persistence image is flattened.
        self.class_labels = class_labels    # The class labels of each persistence image. Length is equal to the number of persistence images.
        self.num_classes = num_classes      # The number of persistence image classes

        self.weight_num_gaussian = None     # Number of Guassian Mixtures in the GMM model determining weight.
        self.weight_gaussian = None         # The weight assigned to each gaussian
        self.weight_gaussian_centers = None # The center of each gaussian cell in (x,y) cooridnates
        self.weights_sigma = None           # The variance values for each gaussian.
        self.weight_func = None             # The weights assigned to each persistence image cell whose center point is given in coordinates.
        self.weight_guassian_pixels = []    # The list containing list storing weight of each persistence image cell wrt the particular gaussian.


    def computeWeight(self, weights, centers, sigma_for_weight):

        # The weighs assigned to each gaussian  
        self.weight_gaussian = weights
        # The number of gaussians
        self.weight_num_gaussian = self.weight_gaussian.shape[0]

        # The list containing the center of each gaussian in (x,y) coordinates
        self.weight_gaussian_centers = np.array(centers)
        # The Standard deviation value assigned to each gaussian.
        self.weights_sigma = np.array(sigma_for_weight)
        # Used to store the weight of each pixel according to the Gaussian Mixture Model sum(weight[i]*exp)
        self.weight_func = np.zeros((self.coordinate_length,1))
        self.weight_guassian_pixels =[]

        for i in range(self.weight_num_gaussian):
            # Get the gaussian center
            center = self.weight_gaussian_centers[i,:]
            # Repeat the the current gaussian centers to facilitate distance computation
            # between the coordinates of the persistence image cell.
            centerCoordinate = np.tile(center, (self.coordinate_length, 1))
            # Perform the exponential step as done in a GMM model
            gaussian = (np.exp(np.sum((self.coordinates - centerCoordinate) ** 2 / (- self.weights_sigma[i] ** 2), axis = 1))).reshape((self.coordinate_length,1))
            # The weight of each persistence image cell wrt ther current gaussian is appended.
            self.weight_guassian_pixels.append(gaussian)
            # Multiply with the weight assigned to each gaussian.
            self.weight_func = self.weight_func + gaussian * self.weight_gaussian[i]
        
        # The list containing list storing weight of each persistence image cell wrt the particular gaussian.
        self.weight_guassian_pixels = np.array(self.weight_guassian_pixels)

    def GramMatrix(self,kernel_sigma):
        
        # Storing the Gram Matrix
        kernel = np.zeros((self.num_pimages, self.num_pimages))
        # Stores the each pixel kernel
        expMatrixList = []
        # Compare pixelwise
        for i in range(self.coordinate_length):
            '''
                    >>> piimages = np.array([[1,2],[2,3],[4,5]])
                    >>> piimages[:,0]
                    array([1, 2, 4])
                    >>> np.tile(piimages[:,0],(3,1))
                    array([[1, 2, 4],
                        [1, 2, 4],
                        [1, 2, 4]])
                    >>> 
            '''
            pimage_ith_pixel = np.tile(self.pimages[:, i], (self.num_pimages, 1))
            
            # Compute the exponent on a pixel.
            expMatrix = np.exp(-(pimage_ith_pixel.T - pimage_ith_pixel) ** 2 / (kernel_sigma ** 2))
            expMatrixList.append(expMatrix)

            # Weight function for each pixel
            kernel += expMatrix * self.weight_func[i]
        
        self.gram_matrix = kernel
        self.expMatrixList = expMatrixList
        return self.gram_matrix

    def computeTestGramMatrix(self, pimages, sigma_for_kernel):
		# Compute the test Gram matrix for svm classifier
		# pimage_2: The test persistence images
		# sigma_for_kernel: sigma of kernels

        test_num = pimages.shape[0]
        kernel = np.zeros((test_num,self.num_pimages))
        for i in range(self.coordinate_length):
            pimage_1_ipixel = np.tile(self.pimages[:,i],(test_num,1))
            pimage_2_ipixel = np.tile(pimages[:,i],(self.num_pimages,1))

            expmatrix = np.exp(-(pimage_1_ipixel - pimage_2_ipixel.T)**2 / (sigma_for_kernel ** 2))
            kernel += expmatrix * self.weight_func[i]
        self.test_gram_matrix = kernel
        return kernel        
      
    def DistanceMetric(self):
        # Compute the distance kw(PI_A,PI_A) + kw(PI_B,PI_B) - 2kw(PI_A,PI_B)
        self.distance = 2*np.sum(self.weight_func) - 2*self.gram_matrix
        return self.distance
    
    def computeCost(self):
        # Within class distance values
        self.intraclass = 0
        # Summation of persistence image class wrt to
        self.interclass = 0

        intra_class_total_distance = [np.sum(self.distance[self.class_labels[i]][:,self.class_labels[i]]) for i in range(self.num_classes)]
        inter_class_total_distance = [np.sum(self.distance[self.class_labels[i]]) for i in range(self.num_classes)]
        self.interclass = inter_class_total_distance
        self.intraclass = intra_class_total_distance

        cost = ((np.array(intra_class_total_distance))/np.array(inter_class_total_distance))
        cost = np.sum(cost)
        return cost


    def computeGradients(self, rate = 1.0):
       
       # Compute the gradients with respect to         
        # Get the gradients with respect to all the gaussian in a list
        gradient_w = []
        gradient_x = []
        gradient_y = []
        gradient_sigma = []
        
        for i in range(self.weight_num_gaussian):
            
            gradient_wi = 0 # Gradient of the weight of the ith gaussian
            gradient_xi = 0 # Gradient of the x coordinate of the center of the ith gaussian
            gradient_yi = 0 # Gradient of the yth coordinate of the center of the ith gaussian
            gradient_sigmai = 0 # Gradient of the standard deviation  of the ith guassian
            
           
            # On taking derivative we get a term 2-2*expMatrixList thats why dimensions chosen this way		
            gradient_metric_wi = np.zeros((self.num_pimages, self.num_pimages))
            gradient_metric_xi = np.zeros((self.num_pimages, self.num_pimages))
            gradient_metric_yi = np.zeros((self.num_pimages, self.num_pimages))
            gradient_metric_sigmai = np.zeros((self.num_pimages, self.num_pimages))
            
            for j in range(self.coordinate_length):
                const = 2 - 2 * self.expMatrixList[j]
                
                # Compute the gradients inside the summation term
                
                gradient_metric_wi += self.weight_guassian_pixels[i][j] * const
                gradient_metric_xi += self.weight_gaussian[i] * self.weight_guassian_pixels[i][j] * ((self.weight_gaussian_centers[i][0] - self.coordinates[j][0])) * (-2) * const / (self.weights_sigma[i] ** 2)
                gradient_metric_yi += self.weight_gaussian[i] * self.weight_guassian_pixels[i][j] * ((self.weight_gaussian_centers[i][1] - self.coordinates[j][1])) * (-2) * const / (self.weights_sigma[i] ** 2)
                gradient_metric_sigmai += self.weight_gaussian[i] * self.weight_guassian_pixels[i][j] * sum((self.coordinates[j] - self.weight_gaussian_centers[i]) ** 2) * 2 * const / (self.weights_sigma[i] ** 3)
           
            
            for j in range(self.num_classes):
                gradient_wi += (np.sum(gradient_metric_wi[self.class_labels[j]][:, self.class_labels[j]]) * self.interclass[j] - np.sum(gradient_metric_wi[self.class_labels[j]]) * self.intraclass[j]) / self.interclass[j] ** 2
                gradient_xi += (np.sum(gradient_metric_xi[self.class_labels[j]][:, self.class_labels[j]]) * self.interclass[j] - np.sum(gradient_metric_xi[self.class_labels[j]]) * self.intraclass[j]) / self.interclass[j] ** 2
                gradient_yi += (np.sum(gradient_metric_yi[self.class_labels[j]][:, self.class_labels[j]]) * self.interclass[j] - np.sum(gradient_metric_yi[self.class_labels[j]]) * self.intraclass[j]) / self.interclass[j] ** 2
                gradient_sigmai += (np.sum(gradient_metric_sigmai[self.class_labels[j]][:, self.class_labels[j]]) * self.interclass[j] - np.sum(gradient_metric_sigmai[self.class_labels[j]]) * self.intraclass[j]) / self.interclass[j] ** 2
            
            
            gradient_w.append(gradient_wi - 1 / np.exp(self.weight_gaussian[i]) * rate)
            gradient_x.append(gradient_xi)
            gradient_y.append(gradient_yi)
            gradient_sigma.append(gradient_sigmai)
                
        self.gradientW = np.array(gradient_w)
        self.gradientX = np.array(gradient_x)
        self.gradientY = np.array(gradient_y)
        self.gradientS = np.array(gradient_sigma)

        return(self.gradientW,self.gradientX,self.gradientY,self.gradientS)


def getCostandGradients(pimages,
    coordinates,labels,num_classes,
    gaussian_weights,gaussian_centers,sigma_for_weight,sigma_for_kernel
    ):

    # Initialize the WKPI kernel with Persistence images, the coordinates of
    # each persistence image cell, labels and the number of classes.
    wkpi = WKPI(pimages,coordinates,labels,num_classes)
    # Calculate the weight to be assigned to each persistence image cell.
    wkpi.computeWeight(gaussian_weights,gaussian_centers,sigma_for_weight)
    # Compute the Kernel matrrix.
    wkpi.GramMatrix(sigma_for_kernel)
    # Compute the distance metric using the formula mentioned in the paper.
    wkpi.DistanceMetric()
    # Compute the cost which is intraclass distance/ total class distance
    cost = wkpi.computeCost()
    # Compute the gradients
    gradients = wkpi.computeGradients()
    # Return the cost and gradients for performing the gradient update rule.
    return cost,gradients


def train(piimages,coordinates,labels,num_classes,
    gaussian_weights,gaussian_centers,sigma_for_weight,sigma_for_kernel,num_epochs=30,lr=0.999):
    
    # Perform training for the specified number of epochs
    for e in range(num_epochs):
        # Get the total cost and the gradients wrt weights, gaussian_center_coordinates,standard_deviation of each gaussian.
        newcost,gradients = getCostandGradients(piimages,coordinates,
            labels,num_classes,gaussian_weights,gaussian_centers,sigma_for_weight,sigma_for_kernel
        )

        # Perform gradient descent over the weights.      
        gaussian_weights -= lr*gradients[0]
        # Perform gradient descent over the x coordinate of the gaussian centers.  
        gaussian_centers[:,0] -= lr*gradients[1]
        # Perform gradient descent over the y coordinate of the gaussian centers
        gaussian_centers[:,1] -= lr*gradients[2]
        # Perform gradient descent over the standard deviation of each gaussian.
        sigma_for_weight -= lr*gradients[3]
        #print(sigma_for_weight)
        print(newcost)
    # Return the final weights,gaussian centers and standard deviation for each gaussian after training.
    return (gaussian_weights,gaussian_centers,sigma_for_weight)









