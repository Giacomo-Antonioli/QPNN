import wandb
from QPNN import *
from dataset import *
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


s=1
init_method=lambda x: torch.nn.init.uniform_(x,a=0.,b=s*np.pi)

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

class ProbsToUnaryLayer(torch.nn.Module):

    def __init__(self, size_in):
        super(ProbsToUnaryLayer, self).__init__()
        self.size_q_in=size_in

    def forward(self, input_var):

        filt = [2**i for i in range(self.size_q_in)]

        return  input_var[:, filt] 
        
class QPNN:
    def __init__(self,structure,X,y,xval=None,yval=None):
        self.X=X
        self.y=y

        self.architecture=[np.shape(X)[1]]+structure+structure+[int((np.shape(y)[1]))]
        self.nqubits=np.max(self.architecture)

  
        self.devs=[]
        self.qnodes=[]
        self.qlayers=[]
        self.model_architecture=[]
        self.model=None
        
        
    def init_model(self,mod_arch=None):

       
        if mod_arch != None:
            self.architecture=[self.architecture[0]]+[mod_arch]+[self.architecture[-1]]
        for layer in range(len(self.architecture)-2):
            n_layers = 1
            n_pars = int((2*max(self.architecture[layer],self.architecture[layer+1])-1-min(self.architecture[layer],self.architecture[layer+1]))*(min(self.architecture[layer+1],self.architecture[layer]))/2)
            
            dev = qml.device("default.qubit", wires=max(self.architecture[layer],self.architecture[layer+1])) #TODO: ADD CUDA
            qnode = qml.QNode(self.probs_single, dev)
            weight_shapes = {"weights": (n_layers, n_pars),"weights_aux":(self.architecture[layer],self.architecture[layer+1])}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes,init_method=init_method)#
            #self.devs.append(dev)
            #self.qnodes.append(qnode)
            #self.qlayers.append(qlayer)
            self.model_architecture.append(qlayer)
            self.model_architecture.append(ProbsToUnaryLayer(self.architecture[layer+1]))
            if layer != len(self.architecture)-2:
               
   
                self.model_architecture.append(torch.nn.Sigmoid())
        self.model.architecture.append(torch.nn.Sequential(10,3))
        self.model= torch.nn.Sequential(*self.model_architecture)
        print("Architecture: ")
        print(self.architecture)
        #input()
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

            theta_i=2*torch.arccos((inputs[...,qi_idx])/prd_fact)
            prd_fact=prd_fact*torch.cos(theta_i)

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
        opt = torch.optim.Adam(self.model.parameters(), lr=0.3)

        loss = torch.nn.CrossEntropyLoss()
        
        X = torch.tensor(self.X, requires_grad=False).float()
        y = torch.tensor(self.y, requires_grad=False).float()
        

            
        batch_size =32 #wandb.config.batch_size if wandb_verbose else 50
        batches = len(y) // batch_size
        print("batches: ",batches)

        data_loader = torch.utils.data.DataLoader(
            list(zip(X, y)), batch_size=batch_size)

            
        for epoch in tqdm(range(wandb.config.epochs if wandb_verbose else 100)):
        
            running_loss = 0
        
            for xs, ys in data_loader:

                opt.zero_grad()
                res=self.model(xs)

                loss_evaluated = loss(res, ys)
            
                loss_evaluated.backward()
        
                opt.step()
            
                running_loss += loss_evaluated

        
            avg_loss = running_loss# / 1 if batches==0 else batches
            print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

            y_pred = self.model(X)
 
            predictions = torch.argmax(y_pred, axis=1).detach().numpy()
            truelabel=torch.argmax(y, axis=1).detach().numpy()

            correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, truelabel)]
            accuracy = sum(correct) / len(correct)
            print(f"Accuracy: {accuracy * 100}%")

            

dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6, "retinamnist": 7}




def stereo_pj(X):
    n,m=np.shape(X)
    newX=np.zeros((n,m+1))
    for rowindex,x in enumerate(X):
        s=np.sum(pow(x,2))
        for index in range(m):
            newX[rowindex,index]=2*x[index]/(s+1)
        newX[rowindex,m]=(s-1)/(s+1)
    return newX


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
            pca_2 = decomposition.PCA(n_components=8)
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

dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6, "retinamnist": 7}

target_dataset="iris"
gpu=False

x,xval,y,yval=get_dataset(dataset_list[target_dataset],split=True)
net=QPNN([10],x,y,xval,yval)#,
if not gpu:
    net.device="cpu"
net.train(wandb_verbose=False,verbose=True)