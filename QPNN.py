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
class NonLinearPostsel(torch.nn.Module):

    def __init__(self, size_out,alpha=1.0,shift=0.0):
        super(NonLinearPostsel, self).__init__()
        self.size_out=size_out
        self.alpha=alpha
        self.shift=shift

    def forward(self, input_var):
        #print(input_var.shape)
        #print(torch.sum(input_var[...,self.size_out:],1).shape)
        return self.alpha*torch.log(input_var[...,:self.size_out]+1e-8+self.shift) #/(1.0-torch.sum(input_var[...,self.size_out:],1))


# usage:

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
        return input_var[:, filt]#*12-6#*3-1.5
        
class QPNN:
    def __init__(self,structure,X,y,xval=None,yval=None):
        self.X=X
        self.y=y
        self.xval=xval
        self.yval=yval
        self.architecture=[np.shape(X)[1]]+structure+[int((np.shape(y)[1]))]
        self.nqubits=np.max(self.architecture)
        self.devs=[]
        self.qnodes=[]
        self.qlayers=[]
        self.model_architecture=[]
        self.model=None
        
        if torch.cuda.is_available():
            self.device="cuda"
            print("YES ")
        else:
            self.device="cpu"
            print("NO")

    def init_model(self,mod_arch=None):
        #print("Initializing Model, using device: ",self.device)
       
        if mod_arch != None:
            if type(mod_arch) is list:
                 self.architecture=[self.architecture[0]]+mod_arch+[self.architecture[-1]]
            else:
                self.architecture=[self.architecture[0]]+[mod_arch]+[self.architecture[-1]]
         
        for layer in range(len(self.architecture)-1):
            n_layers = 1
            n_pars = int((2*max(self.architecture[layer],self.architecture[layer+1])-1-min(self.architecture[layer],self.architecture[layer+1]))*(min(self.architecture[layer+1],self.architecture[layer]))/2)
            
            if self.device == "cuda":
                dev = qml.device("lightning.gpu", wires=max(self.architecture[layer],self.architecture[layer+1]))  

            else:

                dev = qml.device("default.qubit", wires=max(self.architecture[layer],self.architecture[layer+1]))  
            
            qnode = qml.QNode(self.probs_single, dev)
            #qml.partial(qml.transforms.insert, op=qml.DepolarizingChannel, op_args=0.2, position="all")
            weight_shapes = {"weights": (n_layers, n_pars),"weights_aux":(self.architecture[layer],self.architecture[layer+1])}

            qlayer = qml.qnn.TorchLayer(qnode, weight_shapes,init_method=init_method)
            self.devs.append(dev)
            self.qnodes.append(qnode)
            self.qlayers.append(qlayer)
            if layer==0:
                self.model_architecture.append(NonLinearPostsel(self.architecture[layer],2.0,15.0))#per rinormalizzare
                self.model_architecture.append(torch.nn.Softmax(dim=-1))
            self.model_architecture.append(qlayer)
            self.model_architecture.append(ProbsToUnaryLayer(self.architecture[layer+1]))
            print("adding qlayer")
            print("adding Prob to unitary")
            print("adding Sigmoid")
            
            #self.model_architecture.append(torch.nn.Tanh())
            self.model_architecture.append(NonLinearPostsel(self.architecture[layer+1],2.0))#per rinormalizzare
            self.model_architecture.append(torch.nn.Softmax(dim=-1))

 
        self.model= torch.nn.Sequential(*self.model_architecture)
        print("Architecture: ",self.architecture)
 
    
        
        
        
    def probs_single(self,inputs, weights,weights_aux):
        
        shape=np.shape(weights_aux)
        max_q=np.max(shape)
        
        q_base=max_q-shape[0]
        qml.PauliX(wires=q_base)
        prd_fact=1.0
        #print("HERE")
        for qi_idx, qi in enumerate(range(max_q-shape[0],max_q-1)):
            
            #print(inputs[...,qi_idx])
            if np.any(inputs.detach().numpy()<0):
                print(inputs)

            theta_i=2*torch.arccos(torch.sqrt(inputs[...,qi_idx])/prd_fact)
            prd_fact=prd_fact*torch.sin(theta_i/2)+1e-8
            #print(theta_i)
            #RBSGate(theta_i,wires=[qi,qi+1],id=f"$\\alpha1_{{{{{qi}}}}}$")
            qml.Hadamard(wires=qi),
            qml.Hadamard(wires=qi+1),
            qml.CZ(wires=[qi,qi+1]),
            qml.RY(theta_i/2.,wires=qi),
            qml.RY(-theta_i/2.,wires=qi+1),
            qml.CZ(wires=[qi,qi+1]),
            qml.Hadamard(wires=qi),
            qml.Hadamard(wires=qi+1)
        ctr=0

        for ji,j in enumerate(range(max_q,max_q-shape[1],-1)):
            for i in range(q_base-1-ji,q_base-1-ji+shape[0]):
                if i<0:
                    continue
                #RBSGate(weights[0][ctr],[i,i+1],id=f"$\\theta1_{{{{{ctr}}}}}$")
                qml.Hadamard(wires=i),
                qml.Hadamard(wires=i+1),
                qml.CZ(wires=[i,i+1]),
                qml.RY(weights[0][ctr],wires=i),
                qml.RY(-weights[0][ctr],wires=i+1),
                qml.CZ(wires=[i,i+1]),
                qml.Hadamard(wires=i),
                qml.Hadamard(wires=i+1)
                ctr+=1
        #print("HERE 2")
        res =qml.probs(wires=range(max_q-shape[1],max_q))
        #print("HERE 2.5")
        
        return  res


    def predict(self, X):
        #print("HERE3")
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

        loss = torch.nn.CrossEntropyLoss() 
        
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

        data_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=batch_size)

        if self.device=="cuda":
            self.model.to('cuda')

        accuracy=0
        avg_loss=0

        pbar=tqdm(range(wandb.config.epochs if wandb_verbose else 500))

        for epoch in pbar:
        
            running_loss = 0
            
            for xs, ys in data_loader:
      
                if self.device=="cuda":
                    xs.to("cuda")
                    ys.to("cuda")
                #print("LOOOK")
                #print(xs)    
                opt.zero_grad()
                if self.device=="cuda":
                    res=self.model(xs).to("cpu")
                    ys.to("cpu")
                else:
                    #print("HEREs")
                    res=self.model(xs)
                    #print("res")
                    #print(res)
                loss_evaluated = loss(res, ys)
                #print("LOSS")
                #print(loss_evaluated)
               # if np.isnan(loss_evaluated.detach().numpy()):
                #    input()
                loss_evaluated.backward()
        
                opt.step()
        
                running_loss += loss_evaluated
                #input()
        
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
            correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, truelabel)]
            accuracy = sum(correct) / len(correct)
            pbar.set_postfix({'ACC': accuracy*100, 'loss': avg_loss.detach().item() })
       	    wandb.log({"Accuracy":accuracy*100})
 
        
        
        
