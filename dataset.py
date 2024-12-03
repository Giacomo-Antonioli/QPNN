import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import torch 

import torchvision.transforms as transforms
from sklearn import decomposition

import pennylane as qml
from pennylane.operation import Operation, AnyWires

import pennylane.numpy as np

import pandas as pd
from sklearn.datasets import *
import wandb

from sklearn.model_selection import train_test_split

from tqdm import tqdm



dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6, "retinamnist": 7,"pca_digits":8}




def stereo_pj(X):
    n,m=np.shape(X)
    newX=np.zeros((n,m+1))
    for rowindex,x in enumerate(X):
        s=np.sum(pow(x,2))
        for index in range(m):
            newX[rowindex,index]=2*x[index]/(s+1)
        newX[rowindex,m]=(s-1)/(s+1)
    return newX


def get_dataset(index, pca=8, split=True, split_percentage=0.33, standardization_mode=1):
    """
    iris:          Load and return the iris dataset (classification).
    digits:        Load and return the digits dataset (classification).
    wine:          Load and return the wine dataset (classification).
    breast_cancer: Load and return the breast cancer wisconsin dataset (classification).
    """
    X=None
    y=None
    n_classes=None
    print("Getting dataset: ",{i for i in dataset_list if dataset_list[i]==index})
    match index:
        case 1:
            dataset=load_iris()
            X = dataset.data
            y = dataset.target
            n_classes=len(np.unique(y))
            print("Nclasses: ",n_classes)
        case 2:
            dataset=load_digits()
            X_digits = dataset.data
            y = dataset.target
            
            # Initialize PCA and fit the data
            pca_2 = decomposition.PCA(n_components=4)
            pca_2.fit(X_digits)
            
            # Transforming DIGITS data to new dimensions(with 2 features)
            X = pca_2.transform(X_digits)
            n_classes=len(np.unique(y))
        case 8:
            dataset=load_digits()
            X_digits = dataset.data
            y = dataset.target
            X_digits=X_digits[y!=0]
            y=y[y!=0]
            y=y-1
            
            # Initialize PCA and fit the data
            pca_2 = decomposition.PCA(n_components=pca)
            pca_2.fit(X_digits)
            
            # Transforming DIGITS data to new dimensions(with 2 features)
            X = pca_2.transform(X_digits)
            n_classes=len(np.unique(y))
        case 3:
            dataset=load_wine()
            X = dataset.data
            y = dataset.target
            n_classes=len(np.unique(y))
        case 4:
            dataset=load_breast_cancer()
            X = dataset.data
            y = dataset.target
            n_classes=len(np.unique(y))
        case 5:
            iris_data = load_iris()
            iris_df = pd.DataFrame(data=iris_data['data'], columns=iris_data['feature_names'])
            
            iris_df['Iris type'] = iris_data['target']
            iris_df = iris_df[iris_df["Iris type"] != 2]
            iris = iris_df.to_numpy()
            
            X = iris[:, :4]  # we only take the first two features.
            y = np.ndarray.tolist(iris[:, 4].astype(int))
            n_classes=len(np.unique(y))
        case 6:
            X, y = make_moons(n_samples=200, noise=0.1)
            n_classes=len(np.unique(y))
        case 7:
            data_flag = 'retinamnist'
            download = True
            info = INFO[data_flag]
            task = info['task']
            n_channels = info['n_channels']
       
            n_classes = len(info['label'])
            
            DataClass = getattr(medmnist, info['python_class'])
            data_transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = DataClass(split='train', download=download,as_rgb=True)
            elements=[]
            targets=[]
            for index,x in enumerate(train_dataset):
                elements.append(np.asarray(x[0].convert('L')).reshape(-1)/255)
                targets.append(x[1][0])
            pca_2 = decomposition.PCA(n_components=4)

            pca_2.fit(elements)
            
            # Transforming DIGITS data to new dimensions(with 2 features)
            X = pca_2.transform(elements)
            y=targets
            n_classes=len(np.unique(targets))
        case _:
            raise Exception("Sorry, the dataset does not exist")


    match standardization_mode:
        case 1:
            X_mean, X_std=np.mean(X,axis=0), np.std(X,axis=0,ddof=1)
            
            X=(X-X_mean)/X_std
            X=np.hstack((X,2.0*np.ones(X.shape[0])[:,None]))
            X=np.array([np.clip(row/np.sqrt(np.sum(row**2)),-1,1) for row in X])
        case 2:
            X=stereo_pj(X)

    y_hot=torch.zeros((len(y),n_classes),requires_grad=False)
    for index,element in enumerate(y):
        y_hot[index][element]=1

    if split:
        return train_test_split(X, y_hot, test_size=split_percentage, random_state=42)
    else:
        return X, y_hot
