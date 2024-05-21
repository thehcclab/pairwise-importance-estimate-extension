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

from utilities.script_func import init, post, rs_init
import numpy as np
import pickle
import logging
logging.info("")

def main():
    logging.info("Loading Data: ...")

    from utilities.MNISTDataset import MNISTDataset

    dataset_train=MNISTDataset("TRAIN")
    dataset_val=MNISTDataset("TEST")
    # classes=dataset_train.classes
    logging.info("Done")

    logging.info(f"X_train: {dataset_train.X.shape}, X_val: {dataset_val.X.shape}")
    array= rs_init("MNIST/", args.rs, args.percentile, dataset_train.X)
    #baseline_model= DryBeanBaseline_Model(device=device, input_dim=dataset_train.X.shape[1], classes=classes)
    #model= Models.Univariate_RS(device=device, input_dim=dataset_train.X.shape[1], model=baseline_model,rs_path="MNIST/",rs_selection=args.rs).to(device)

    train_dataloader= torch.utils.data.DataLoader(dataset_train.return_subset_dataset(array), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(dataset_val.return_subset_dataset(array), batch_size=config["batch_size"])

    import Models.models as Models
    import Models.model_func as Model_Func
    from Models.MNISTBaseline_Model import MNISTBaseline_Model

    model= MNISTBaseline_Model(device=device, input_dim=np.count_nonzero(array), classes=dataset_train.classes).to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train)

    dir_path=default_path+f"RS/RS{args.rs}/"
    if not os.path.exists(dir_path):
             os.makedirs(dir_path)

    return model, dataset_train.classes, dataset_val, val_dataloader
# pickle.dump( model.return_layer_weights(), open(default_path+"RS/"+feat+"-w-"+str(config["epoch"])+".pkl", "wb") )
import argparse
if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to produce the baselines and store the training-validation split of the data for subsequent training.')
    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")
    parser.add_argument("percentile", help="Percetile of feature to select. Default: 10", default=10)
    parser.add_argument("rs", help="Selection of rs. Default: 0", default=0)

    args= parser.parse_args()
    args.dir= str(args.dir)
    args.percentile=int(args.percentile)
    args.rs= int(args.rs)

    feat= f"MNIST-subset-RS{args.rs}-{args.percentile}"

    default_path= init(args, default_path, config)
    model, classes, dataset_val, val_dataloader= main()
    post(default_path, config, feat, model, dataset_val.y, classes, val_dataloader, f"RS/RS{args.rs}")
