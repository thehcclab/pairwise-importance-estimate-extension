import matplotlib
matplotlib.use("Agg")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml, io
with open("./config/MNIST_config.yaml", 'r') as stream:
        config= yaml.safe_load(stream)

config["default_path"]= str(config["default_path"])
config["cuda"]= str(config["cuda"])
config["epoch"]= int(config["epoch"])
config["batch_size"]=int(config["batch_size"])


import os
os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda"]

default_path=config["default_path"]

from utilities.script_func import init, post, MNIST_pre
import pickle
import logging
logging.info("")

feat= "MNIST-NFS"
def main():

    import Models.models as Models
    import Models.model_func as Model_Func
    from Models.MNISTBaseline_Model import MNISTBaseline_Model

    input_dim=dataset_val.X.shape[1]
    baseline_model= MNISTBaseline_Model(device=device, input_dim=input_dim, classes=classes).to(device)
    non_linear_func= torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim),
        torch.nn.BatchNorm1d(input_dim)
    )
    model= Models.Univariate_NeuralFS(device=device, input_dim=input_dim,
                                      nonlinear_func=non_linear_func, decision_net=baseline_model).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # for i in model.parameters():
    #     print(i.shape)


    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train)

    dir_path=default_path+"NFS/"
    if not os.path.exists(dir_path):
             os.makedirs(dir_path)

    pickle.dump( model.return_pairwise_weights(), open(dir_path+feat+"-w-"+str(config["epoch"])+".pkl", "wb") )

    return model

import argparse
if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to produce the baselines and store the training-validation split of the data for subsequent training.')
    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")

    args= parser.parse_args()
    args.dir= str(args.dir)

    default_path= init(args, default_path, config)
    train_dataloader, val_dataloader, dataset_val, classes= MNIST_pre(config)
    model= main()
    post(default_path, config, feat, model, dataset_val.y, classes, val_dataloader, "NFS")
