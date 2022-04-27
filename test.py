import numpy as np
from sklearn.model_selection import StratifiedKFold
piimages = np.array([[1.0,2.0],[2.0,3.0],[4.0,5.0],[1.0,2.0],[1.0,1.4]])
labelTrain = np.array([0,0,0,1,1])

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(piimages, labelTrain)

for train_index, test_index in skf.split(piimages, labelTrain):
    print(train_index)
    print(test_index)
