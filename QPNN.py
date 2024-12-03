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


s=0.05
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
        #print(input_var)
        filt = [2**i for i in range(self.size_q_in)]
        #print(filt)
        #print(input_var[:, filt])
        return input_var[:, filt]*12-6
        
class QPNN:
    def __init__(self,structure,X,y,xval=None,yval=None):
        self.X=X
        self.y=y
        self.xval=xval
        self.yval=yval
        self.architecture=[np.shape(X)[1]]+structure+[int((np.shape(y)[1]))]
        self.nqubits=np.max(self.architecture)
        if torch.cuda.is_available():
            self.device="cuda"
            print("YES ")
        else:
            self.device="cpu"
            print("NO")
  
        self.devs=[]
        self.qnodes=[]
        self.qlayers=[]
        self.model_architecture=[]
        self.model=None
        
        
    def init_model(self,mod_arch=None):
        #print("Initializing Model, using device: ",self.device)
       
        if mod_arch != None:
            self.architecture=[self.architecture[0]]+[mod_arch]+[self.architecture[-1]]
        for layer in range(len(self.architecture)-1):
            n_layers = 1
            n_pars = int((2*max(self.architecture[layer],self.architecture[layer+1])-1-min(self.architecture[layer],self.architecture[layer+1]))*(min(self.architecture[layer+1],self.architecture[layer]))/2)
            if self.device == "cuda":
                dev = qml.device("lightning.gpu", wires=max(self.architecture[layer],self.architecture[layer+1])) #TODO: ADD CUDA

            else:

                dev = qml.device("default.qubit", wires=max(self.architecture[layer],self.architecture[layer+1])) #TODO: ADD CUDA
            qnode = qml.QNode(self.probs_single, dev)
            weight_shapes = {"weights": (n_layers, n_pars),"weights_aux":(self.architecture[layer],self.architecture[layer+1])}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes,init_method=init_method)#
            self.devs.append(dev)
            self.qnodes.append(qnode)
            self.qlayers.append(qlayer)
            self.model_architecture.append(qlayer)
            self.model_architecture.append(ProbsToUnaryLayer(self.architecture[layer+1]))
            print("adding qlayer")
            print("adding Prob to unitary")

                #print("Layer[",layer,"]: arch:",self.architecture[layer] ," pars: ", n_pars)
            #if layer!=len(self.architecture)-2:
            print("adding Sigmoid")
            
            self.model_architecture.append(torch.nn.Sigmoid())
            #print(". Added probsToUnitary. Added Softmax ")
 
    
       #self.model_architecture.append(torch.nn.Linear(10,3))
        self.model= torch.nn.Sequential(*self.model_architecture)
        print("Architecture: ",self.architecture)
        
        #input()
       # for index,x in  enumerate(reversed(list(self.model.parameters()))):
        #    if index%2==0:
        #        x.requires_grad =False    
    
    
        
        
        
    def probs_single(self,inputs, weights,weights_aux):
        shape=np.shape(weights_aux)
        max_q=np.max(shape)
        
        q_base=max_q-shape[0]
        #print("shape:", shape)
        #print("qbase: ",q_base)
        #print(weights)
        qml.PauliX(wires=q_base)
        prd_fact=1.0
        for qi_idx, qi in enumerate(range(max_q-shape[0],max_q-1)):

            theta_i=2*torch.arccos((inputs[...,qi_idx])/prd_fact)
            #prd_fact=prd_fact*torch.sin(theta_i)
            #print(theta_i)
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
        if wandb_verbose:
            run= wandb.init()
            self.init_model(wandb.config.arch_elements)
        else:
            self.init_model()
        if wandb_verbose:
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
            opt = torch.optim.Adam(self.model.parameters(), lr=0.2)

        loss = torch.nn.CrossEntropyLoss()#.to("cuda")
        
        X = torch.tensor(self.X, requires_grad=False).float()
        y = torch.tensor(self.y, requires_grad=False).float()
        if self.xval is not None:
            Xval = torch.tensor(self.xval, requires_grad=False).float()
            yval = torch.tensor(self.yval, requires_grad=False).float()

        if self.device=="cuda":
            X.to('cuda')
            y.to("cuda")

            if self.xval is not None:
                Xval.to('cuda')
                yval.to("cuda")

            
        batch_size =len(y) #wandb.config.batch_size if wandb_verbose else 50
        batches = 1
        #print("batches: ",batches)

        data_loader = torch.utils.data.DataLoader(
            list(zip(X, y)), batch_size=batch_size)
        if self.device=="cuda":
            self.model.to('cuda')
        accuracy=0
        avg_loss=0
        pbar=tqdm(range(wandb.config.epochs if wandb_verbose else 500))
        for epoch in pbar:
        
            running_loss = 0
            
            for xs, ys in data_loader:
               #print(xs)
                #input()
                if self.device=="cuda":
                    xs.to("cuda")
                    ys.to("cuda")
                    
                opt.zero_grad()
                if self.device=="cuda":
                    res=self.model(xs).to("cpu")
                    ys.to("cpu")
                else:
                    res=self.model(xs)
                #print(res)   
                loss_evaluated = loss(res, ys)
                #print(loss_evaluated)
                #print(res)
                #input()
                loss_evaluated.backward()
        
                opt.step()
        
                running_loss += loss_evaluated
        
            avg_loss = running_loss# /batch_size# / 1 if batches==0 else batches
           # if verbose: 
           #     print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
            if wandb_verbose:
                wandb.log({"loss":avg_loss })
            #VALIDATION
            if self.xval is not None:
                y_pred_val = self.model(Xval)
                if self.device=="cuda":
                    y_pred_val.to("cpu")
                predictions_val = torch.argmax(y_pred_val, axis=1).detach().numpy()
                loss_evaluated_val = loss(y_pred_val, yval)
                y_val_pos=torch.argmax(yval, axis=1).detach().numpy()

                correct_val = [1 if p == p_true else 0 for p, p_true in zip(predictions_val, y_val_pos)]
                accuracy_val = sum(correct_val) / len(correct_val)
                #if verbose:
                #    print(f"Validation Accuracy: {accuracy_val * 100}%")
                #    print(f"Validation Loss: {loss_evaluated_val}")
                if wandb_verbose:
                    wandb.log({"accuracy_validation":accuracy_val * 100,"validation_loss":loss_evaluated_val})
                    
            if self.device=="cuda":
                y_pred = self.model(X).to("cpu")
            else:
                y_pred = self.model(X)
 
            predictions = torch.argmax(y_pred, axis=1).detach().numpy()
            truelabel=torch.argmax(y, axis=1).detach().numpy()
            #print(predictions)
            #print(truelabel)
            ##input()
            correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, truelabel)]
            accuracy = sum(correct) / len(correct)
            pbar.set_postfix({'ACC': accuracy*100, 'loss': avg_loss.detach().item() })
       	    wandb.log({"Accuracy":accuracy*100})
	# if verbose:
        #    print(f"Accuracy: {accuracy * 100}%")
       # if wandb_verbose:
       #     wandb.log({"accuracy":accuracy * 100})
        
        
        
