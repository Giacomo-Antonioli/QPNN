import wandb
from QPNN import *
from dataset import *

dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6, "retinamnist": 7}

target_dataset="iris"
gpu=False

x,xval,y,yval=get_dataset(dataset_list[target_dataset],split=True)
net=QPNN([],x,y,xval,yval)#,
if not gpu:
    net.device="cpu"
net.train(wandb_verbose=False,verbose=True)