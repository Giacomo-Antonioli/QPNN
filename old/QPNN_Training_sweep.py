#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch 


from sklearn import decomposition

import pennylane as qml
from pennylane.operation import Operation, AnyWires
import matplotlib.pyplot as plt
import pennylane.numpy as np
import numpy as np
import pandas as pd
from sklearn.datasets import *
import wandb
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pprint
from tqdm import tqdm
target_dataset="wine"
dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6}
s=0.05
init_method=lambda x: torch.nn.init.uniform_(x,a=0.,b=s*np.pi)


# In[2]:


def stereo_pj(X):
    n,m=np.shape(X)
    newX=np.zeros((n,m+1))
    for rowindex,x in enumerate(X):
        s=np.sum(pow(x,2))
        for index in range(m):
            newX[rowindex,index]=2*x[index]/(s+1)
        newX[rowindex,m]=(s-1)/(s+1)
    return newX


# In[3]:


def get_dataset(index, split=True, split_percentage=0.33, standardization_mode=1):
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
        case 2:
            dataset=load_digits()#TODO: PCA
            X_digits = dataset.data
            y = dataset.target
            
            # Initialize PCA and fit the data
            pca_2 = decomposition.PCA(n_components=4)
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


# In[4]:


class RBSGate(Operation):
    num_wires = 2  

    def __init__(self, theta, wires, id=None):
        all_wires = qml.wires.Wires(wires)
        super().__init__(theta, wires=all_wires, id=id)

    @staticmethod
    def compute_decomposition(theta, wires):
        decomp = [
                qml.Hadamard(wires=wires[0]),
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=wires),
                qml.RY(theta/2.,wires=wires[0]),
                qml.RY(-theta/2.,wires=wires[1]),
                qml.CZ(wires=wires),
                qml.Hadamard(wires=wires[0]),
                qml.Hadamard(wires=wires[1])
            ]
        return decomp


# In[5]:


class ProbsToUnaryLayer(torch.nn.Module):

    def __init__(self, size_in):
        super(ProbsToUnaryLayer, self).__init__()
        self.size_q_in=size_in

    def forward(self, input_var):
       # print("probstounitary")
        #print(input_var)

        filt = [2**i for i in range(self.size_q_in)]
        #print("filtered: ",input_var[:, filt])
        #input()
        return input_var[:, filt]*12-6


# In[6]:


class QPNN:
    def __init__(self,structure,X,y,xval=None,yval=None):
        self.X=X
        self.y=y
        self.xval=xval
        self.yval=yval
        self.architecture=[np.shape(X)[1]]+structure+[int((np.shape(y)[1]))]
        print(self.architecture)
        self.nqubits=np.max(self.architecture)
        self.cuda=torch.cuda.is_available()
        self.devs=[]
        self.qnodes=[]
        self.qlayers=[]
        self.model_architecture=[]
        self.model=None
        
        
        
        
        

    
    def init_model(self):
        for layer in range(len(self.architecture)-1):
            n_layers = 1
            n_pars = int((2*max(self.architecture[layer],self.architecture[layer+1])-1-min(self.architecture[layer],self.architecture[layer+1]))*(min(self.architecture[layer+1],self.architecture[layer]))/2)

            dev = qml.device("default.qubit", wires=max(self.architecture[layer],self.architecture[layer+1])) #TODO: ADD CUDA
            qnode = qml.QNode(self.probs_single, dev)
            weight_shapes = {"weights": (n_layers, n_pars),"weights_aux":(self.architecture[layer],self.architecture[layer+1])}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes,init_method=init_method)#
            self.devs.append(dev)
            self.qnodes.append(qnode)
            self.qlayers.append(qlayer)
            self.model_architecture.append(qlayer)
            self.model_architecture.append(ProbsToUnaryLayer(self.architecture[layer+1]))
            self.model_architecture.append(torch.nn.Softmax(dim=1))
            self.model= torch.nn.Sequential(*self.model_architecture)
            for index,x in  enumerate(reversed(list(self.model.parameters()))):
                if index%2==0:
                    x.requires_grad =False    
    
    
        
        
        
    def probs_single(self,inputs, weights,weights_aux):
        shape=np.shape(weights_aux)
        max_q=np.max(shape)
        
        q_base=max_q-shape[0]
        qml.PauliX(wires=q_base)
        prd_fact=1.0
        for qi_idx, qi in enumerate(range(max_q-shape[0],max_q-1)):

            theta_i=torch.arccos((inputs[...,qi_idx])/prd_fact)
            prd_fact=prd_fact*torch.sin(theta_i)
            RBSGate(theta_i,wires=[qi,qi+1],id=f"$\\alpha1_{{{{{qi}}}}}$")

        ctr=0

        for ji,j in enumerate(range(max_q,max_q-shape[1],-1)):
            for i in range(q_base-1-ji,q_base-1-ji+shape[0]):
                if i<0:
                    continue
                RBSGate(weights[0][ctr],[i,i+1],id=f"$\\theta1_{{{{{ctr}}}}}$")
                
                ctr+=1

        return qml.probs(wires=range(max_q-shape[1],max_q))
    
    def predict(self, X):
        X = torch.tensor(X, requires_grad=False).float()
        return torch.argmax(self.model(X), dim=1).detach().numpy()
    
        
    def train(self,verbose=False,wandb_verbose=False):
        
        self.init_model()
        if wandb_verbose:
            run= wandb.init()
            wandb.log({'architecture':self.architecture })
        if wandb_verbose:
            match wandb.config.optimizer:
                case "SGD":
                     opt = torch.optim.SGD(self.model.parameters(), lr=wandb.config.learning_rate)
                case "ADAM":
                    opt = torch.optim.Adam(self.model.parameters(), lr=wandb.config.learning_rate)
                case "ADAMW":
                     opt = torch.optim.AdamW(self.model.parameters(), lr=wandb.config.learning_rate)
                case "RMSPROP":
                     opt = torch.optim.RMSprop(self.model.parameters(), lr=wandb.config.learning_rate)
        else:
            opt = torch.optim.SGD(self.model.parameters(), lr=0.1)

        loss = torch.nn.CrossEntropyLoss()
        
        X = torch.tensor(self.X, requires_grad=False).float()
        y = torch.tensor(self.y, requires_grad=False).float()
        if self.xval is not None:
            Xval = torch.tensor(self.xval, requires_grad=False).float()
            yval = torch.tensor(self.yval, requires_grad=False).float()

        batch_size = wandb.config.batch_size if wandb_verbose else 1
        batches = len(y) // batch_size
        print("batches: ",batches)

        data_loader = torch.utils.data.DataLoader(
            list(zip(X, y)), batch_size=batch_size, shuffle=True, drop_last=True
        )
        
        
        for epoch in tqdm(range(wandb.config.epochs if wandb_verbose else 10)):
        
            running_loss = 0
            
            for xs, ys in data_loader:
                opt.zero_grad()
                res=self.model(xs)
                loss_evaluated = loss(res, ys)

                loss_evaluated.backward()
        
                opt.step()
        
                running_loss += loss_evaluated
        
            avg_loss = running_loss #/ batches
            if verbose: 
                print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
            if wandb_verbose:
                wandb.log({"loss":avg_loss })
            #VALIDATION
            if self.xval is not None:
                y_pred_val = self.model(Xval)
                predictions_val = torch.argmax(y_pred_val, axis=1).detach().numpy()
                loss_evaluated_val = loss(y_pred_val, yval)
                y_val_pos=torch.argmax(yval, axis=1).detach().numpy()
                correct_val = [1 if p == p_true else 0 for p, p_true in zip(predictions_val, y_val_pos)]
                accuracy_val = sum(correct_val) / len(correct_val)
                if verbose:
                    print(f"Validation Accuracy: {accuracy_val * 100}%")
                    print(f"Validation Loss: {loss_evalueted_val}")
                if wandb_verbose:
                    wandb.log({"accuracy_validation":accuracy_val * 100,"validation_loss":loss_evaluated_val})
                    

            y_pred = self.model(X)
            predictions = torch.argmax(y_pred, axis=1).detach().numpy()
            truelabel=torch.argmax(y, axis=1).detach().numpy()
    
            correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, truelabel)]
            accuracy = sum(correct) / len(correct)
            if verbose:
                print(f"Accuracy: {accuracy * 100}%")
            if wandb_verbose:
                wandb.log({"accuracy":accuracy * 100})
        
        


# In[7]:


def main():
    x,xval,y,yval=get_dataset(dataset_list['wine'],split=True)
    net=QPNN([15],x,y)#,xval,yval
    net.train(wandb_verbose=True)


# In[8]:


def generate_sweep_config():
    wandb.login()
    sweep_config = {'method': 'bayes'}
    metric = {'name': 'loss','goal': 'minimize'}
    sweep_config['metric'] = metric
    parameters_dict = { 'epochs': {
                            'values': [10,50,100,1000]},
                       'optimizer':{'values':["SGD","ADAM","ADAMW","RMSPROP"]}
                        }
    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.9
          },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 8,
            'max': 100,
          }
        })
    pprint.pprint(sweep_config)
    return sweep_config


# In[ ]:


sweep_id = wandb.sweep(sweep=generate_sweep_config(), entity='quantum_kets', project='QPNNTRAINSWEEP_wine')
wandb.agent(sweep_id, function=main, count=30)


# Aggiungere roba nello sweep: structure ecc,dataset
# add cuda or cpu
# aggiungere numero variabile fili max
# 
# 

# In[ ]:




