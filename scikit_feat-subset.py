import matplotlib
matplotlib.use("Agg")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utilities.Log import define_root_logger
import logging
logging.info("")
import pickle

# from utilities.script_func import init
from utilities.subset_func import Grad_AUC, Grad_ROC, Grad_COV, Grad_STD, Fisher_Score, FScore, Percentile_Subset, subset_post
import numpy as np

def main():

    global args

    logging.info("Loading Data: "+args.data+"...")

    from utilities.ScikitFeatDataset import ScikitFeatDataset

    data=ScikitFeatDataset(dataset=args.data, split=config["split"], default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    import math

    logging.info(f"X_train: {len(data.train_index)}, X_val: {len(data.val_index)}")


    if args.model=="Grad-AUC":

        args.model="Grad"

        weights= Grad_AUC(load_w, config["epoch"], args.subset, data.X)

    elif args.model=="Grad-ROC":

        args.model="Grad"

        weights= Grad_ROC(load_w, config["epoch"], args.subset, data.X)

    elif args.model=="Grad-COV":

        args.model="Grad"

        weights= Grad_COV(load_w, config["epoch"], args.subset, data.X)

    elif args.model=="Grad-STD":

        args.model="Grad"

        weights= Grad_STD(load_w, config["epoch"], args.subset, data.X)

    elif args.model=="Fisher":

        weights= Fisher_Score(default_path, load_w, args.model, args.subset, data.X.shape[1], *data.return_training_data())

    elif args.model=="FScore":

        weights= FScore(default_path, load_w, args.model, args.subset, data.X.shape[1], *data.return_training_data())

    else:
        weights= Percentile_Subset(load_w, args.subset)


    train_dataloader= torch.utils.data.DataLoader(data.return_training_subset_dataset(weights),batch_size=config["batch_size"] )
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_subset_dataset(weights), batch_size=config["batch_size"])

    import Models.models as Models
    import Models.model_func as Model_Func
    from Models.ScikitFeatBaseline_Model import ScikitFeatBaseline_Model

    print("original dim: ", data.X_train.shape)
    print("dim: ", data.X_train[:, weights].shape)

    model= ScikitFeatBaseline_Model(device=device, input_dim=np.count_nonzero(weights), classes=data.classes, dataset=data.dataset)
    # model= Models.Univariate_Subset(device=device, input_dim=data.X.shape[1], model=baseline_model)

    # model.subset_layer_weights(weights.astype(float))
    model.to(device)


    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # for i in model.parameters():
    #     print(i.shape)


    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train)

    if args.model=="ThresholdedWeight":
        args.model="Weight"

    return model, data, val_dataloader

if __name__=="__main__":
    import argparse

    parser= argparse.ArgumentParser(description='This script is used to produce the baselines and store the training-validation split of the data for subsequent training.')


    parser.add_argument("-data", help="Dataset to load. Default: \"USPS\"", default="USPS")
    parser.add_argument('dir', help='default path of where experiment is saved. Default: \"./experiment/Benchmark/exp_log0\"', default="None")
    parser.add_argument("subset", help="Percentage of subset. Default: \"10\"", default="10")
    parser.add_argument("model", help="method for selection", default= "None")


    args= parser.parse_args()
    args.data= str(args.data)
    args.dir= str(args.dir)
    args.model= str(args.model)

    import yaml, io

    with open("./config/DL_"+args.data+"_config.yaml", 'r') as stream:
            config= yaml.safe_load(stream)

    config["default_path"]= str(config["default_path"])
    config["split"]= float(config["split"])
    config["cuda"]= str(config["cuda"])
    config["epoch"]= int(config["epoch"])
    config["batch_size"]=int(config["batch_size"])


    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda"]

    default_path=config["default_path"]

    feat= f"scikit_feat-subset-{args.model}-{args.data}-{args.subset}"

    default_path= default_path+f"/{args.dir}"
    file_name="log"+args.dir[-1]
    define_root_logger(default_path+f"/{file_name}.log")
    default_path= default_path+"/"

    w_path={
        "DF":default_path+f"DF/scikit_feat-DF-{args.data}-w-{config['epoch']}.pkl",
        "Weight":default_path+f"Weight/scikit_feat-Weight-{args.data}-w-{config['epoch']}.pkl",
        "Fisher":default_path+f"Fisher/fisher_score_idx-{args.data}.pkl",
        "FScore":default_path+f"FScore/fscore_idx-{args.data}.pkl",
        "ThresholdedWeight":default_path+f"Weight/scikit_feat-ThresholdedWeight-{args.data}-w-{config['epoch']}.pkl",
        "Grad-STD": default_path+f"Grad/list/scikit_feat-Grad-list-{args.data}-{config['epoch']}.pkl",
        "Grad-AUC": default_path+f"Grad/list/scikit_feat-Grad-list-{args.data}-{config['epoch']}.pkl",
        "Grad-ROC": default_path+f"Grad/list/scikit_feat-Grad-list-{args.data}-{config['epoch']}.pkl",
        "Grad-COV": default_path+f"Grad/list/scikit_feat-Grad-list-{args.data}-{config['epoch']}.pkl",
        "NFS":default_path+f"NFS/scikit_feat-NFS-{args.data}-w-{config['epoch']}.pkl"
    }

    load_w=w_path[args.model]

    model, data, val_dataloader= main()
    subset_post(default_path, config, feat, model, data.y_val, data.classes, val_dataloader, f"{default_path}{args.model}")
