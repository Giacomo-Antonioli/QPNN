import wandb
from QPNN import *
from dataset import *
import warnings
warnings.filterwarnings("ignore")
import torch
print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
dataset_list={"iris": 1,"digits": 2,"wine": 3,"cancer": 4, "iris_linear": 5, "moon": 6, "retinamnist": 7, "pca_digits":8}

#import os
#os.environ["OMP_NUM_THREADS"]="1"

target_dataset="pca_digits"
gpu=False
connectivity='full'
hot_qubits=[0,1] # unimplemented yet for "1d" (default) connectivity
num_params=80
boundary_arch=[5,5]

def generate_sweep_config():
    wandb.login()
    sweep_config = {'method': 'bayes'}
    metric = {'name': 'loss','goal': 'minimize'}
    sweep_config['metric'] = metric
    parameters_dict = { 'epochs': {
                            'values': [300,400,500]},
                       'optimizer':{'values':["ADAM","ADAMW"]},
                        'arch_elements':{'values':[[]]}
                        }
    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
          },
       # 'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
         #   'distribution': 'q_log_uniform_values',
        #    'q': 2,
        #    'min': 8,
        #    'max': 100,
         # }
        })
    #pprint.pprint(sweep_config)
    return sweep_config

def main(gpu=False):
    x,xval,y,yval=get_dataset(dataset_list[target_dataset],split=True)
    net=QPNN([5],x,y,xval,yval,connectivity=connectivity,hot_qubits=hot_qubits,num_params=num_params,boundary_arch=boundary_arch)#,
    if not gpu:
        net.device="cpu"
    net.train(wandb_verbose=True,verbose=True)

sweep_id = wandb.sweep(sweep=generate_sweep_config(), entity='quantum_kets', project='QPNN_TRAIN_SWEEP_'+target_dataset)
wandb.agent(sweep_id, function=main, count=200)
