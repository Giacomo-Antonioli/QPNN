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

target_dataset="pca_digits"
gpu=False

def generate_sweep_config():
    wandb.login()
    sweep_config = {'method': 'bayes'}
    metric = {'name': 'loss','goal': 'minimize'}
    sweep_config['metric'] = metric
    parameters_dict = { 'epochs': {
                            'values': [300,400,500]},
                       'optimizer':{'values':["ADAM","ADAMW"]},
                        'arch_elements':{'values':[8,9,10,11]}
                        }
    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
          },
        })
    return sweep_config


def main(gpu=False):
    x,xval,y,yval=get_dataset(dataset_list[target_dataset],split=True)

    net=QPNN([],x,y,xval,yval)

    if not gpu:
        net.device="cpu"
    net.train(wandb_verbose=True,verbose=True)

sweep_id = wandb.sweep(sweep=generate_sweep_config(), entity='quantum_kets', project='QPNN_TRAIN_SWEEP_'+target_dataset)
wandb.agent(sweep_id, function=main, count=200)
