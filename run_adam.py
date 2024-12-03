import torch
from torch import nn
from torch.optim import SGD, Adam
from sklearn.datasets import load_iris, load_digits
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
 
from sklearn import decomposition

def run():


    dataset=load_digits()
    X_digits = dataset.data
    y = dataset.target
    X_digits=X_digits[y!=0]
    y=y[y!=0]
    y=y-1
    
    # Initialize PCA and fit the data
    pca_2 = decomposition.PCA(n_components=8)
    pca_2.fit(X_digits)
    
    # Transforming DIGITS data to new dimensions(with 2 features)
    X = pca_2.transform(X_digits)
    n_classes=len(np.unique(y))
    loader = DataLoader(list(zip(X, y)), batch_size=32)
    net = nn.Sequential(
        nn.Linear(8, 10),
        nn.Sigmoid(),
        nn.Linear(10, n_classes)
    )
    opt = Adam(net.parameters(), lr=0.1)

    for e in range(100):
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            y_pred = net(x.float())
            preds = torch.argmax(y_pred, dim=1)
            loss = nn.CrossEntropyLoss()(y_pred, y)

            correct += (preds == y).sum()
            total += len(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = correct / total * 100.0

        print(loss)
        print(acc)


dataset=load_digits()
X_digits = dataset.data
y = dataset.target
X_digits=X_digits[y!=0]
y=y[y!=0]
y=y-1

# Initialize PCA and fit the data
pca_2 = decomposition.PCA(n_components=8)
pca_2.fit(X_digits)

# Transforming DIGITS data to new dimensions(with 2 features)
X = pca_2.transform(X_digits)
n_classes=len(np.unique(y))
 

def classical(arch,x,y):
    loader = DataLoader(list(zip(x, y)), batch_size=32)
    listOfLayers=[]
    listOfLayers.append(nn.Linear(np.shape(x)[1],arch[0]))
    listOfLayers.append(nn.Sigmoid())
    for index in range(len(arch)-1):
        listOfLayers.append(nn.Linear(arch[index],arch[index+1]))
        listOfLayers.append(nn.Sigmoid())
 
    listOfLayers.append(nn.Linear( arch[-1],len(np.unique(y))))
 
    net = nn.Sequential(*listOfLayers)
    opt = Adam(net.parameters(), lr=0.1)

    for e in range(100):
        correct = 0
        total = 0
        for x, y in tqdm(loader):
            y_pred = net(x.float())
            preds = torch.argmax(y_pred, dim=1)
            loss = nn.CrossEntropyLoss()(y_pred, y)

            correct += (preds == y).sum()
            total += len(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = correct / total * 100.0

        print(loss)
        print(acc)

#classical([8,6],X,y)