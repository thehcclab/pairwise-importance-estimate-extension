import matplotlib
matplotlib.use("Agg")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml, io
with open("./config/DryBean_config.yaml", 'r') as stream:
        config= yaml.safe_load(stream)

config["default_path"]= str(config["default_path"])
config["split"]= float(config["split"])
config["cuda"]= str(config["cuda"])
config["epoch"]= int(config["epoch"])
config["batch_size"]=int(config["batch_size"])

import os
os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda"]

default_path=config["default_path"]

from utilities.script_func import init, post, DryBean_pre
import pickle
import logging
logging.info("")


feat= "DryBean-Baseline"

# def main(default_path, config, train_dataloader, val_dataloader, data, classes):
def main():
    import Models.model_func as Model_Func
    from Models.DryBeanBaseline_Model import DryBeanBaseline_Model

    model= DryBeanBaseline_Model(device=device, input_dim=data.X.shape[1], classes=classes).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# for i in model.parameters():
#     print(i.shape)
    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train)


    return model


import argparse
if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to produce the baselines and store the training-validation split of the data for subsequent training.')
    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")

    args= parser.parse_args()
    args.dir= str(args.dir)

    default_path= init(args, default_path, config)
    train_dataloader, val_dataloader, data, classes= DryBean_pre(config, default_path)
    model= main()
    post(default_path, config, feat, model, data.y_val, classes, val_dataloader, "Baseline")
