import matplotlib
matplotlib.use("Agg")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utilities.script_func import init, post, rs_init
import numpy as np
import pickle
import logging
logging.info("")

def main():
    logging.info("Loading Data: "+args.data+"...")

    from utilities.ScikitFeatDataset import ScikitFeatDataset

    data=ScikitFeatDataset(dataset=args.data, split=config["split"], default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {len(data.train_index)}, X_val: {len(data.val_index)}")
    array= rs_init(f"scikit_feature/{args.data}-", args.rs, args.percentile, data.X)
    train_dataloader= torch.utils.data.DataLoader(data.return_training_subset_dataset(array), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_subset_dataset(array), batch_size=config["batch_size"])

    import Models.models as Models
    import Models.model_func as Model_Func
    from Models.ScikitFeatBaseline_Model import ScikitFeatBaseline_Model

    model= ScikitFeatBaseline_Model(device=device, input_dim=np.count_nonzero(array), classes=data.classes, dataset=data.dataset)
    # model= Models.Univariate_RS(device=device, input_dim=data.X.shape[1], model=baseline_model,rs_path=f"scikit_feature/args.data", rs_selection=args.rs)

    model.to(device)

    logging.info("Percentile: "+str(args.percentile))
    # model.select_percentile(args.percentile)

    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # for i in model.parameters():
    #     print(i.shape)


    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train)

    dir_path=default_path+f"RS/RS{args.rs}/"
    if not os.path.exists(dir_path):
             os.makedirs(dir_path)

    return model, data.classes, data, val_dataloader

# pickle.dump( model.return_layer_weights(), open(default_path+"RS/"+feat+"-w-"+str(config["epoch"])+".pkl", "wb") )
if __name__=="__main__":

    import argparse

    parser= argparse.ArgumentParser(description='This script is used to produce the baselines and store the training-validation split of the data for subsequent training.')

    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")
    parser.add_argument("-data", help="Dataset to load. Default: \"USPS\"", default="USPS")
    parser.add_argument("percentile", help="Percetile of feature to select. Default: 10", default=10)
    parser.add_argument("rs", help="Selection of rs. Default: 0", default=0)

    args= parser.parse_args()
    args.data= str(args.data)
    args.dir= str(args.dir)
    args.percentile=int(args.percentile)
    args.rs= int(args.rs)

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


    feat= f"scikit_feat-subset-RS{args.rs}-"+args.data+"-"+str(args.percentile)

    default_path= init(args, default_path, config)
    model, classes, dataset, val_dataloader= main()
    post(default_path, config, feat, model, dataset.y_val, classes, val_dataloader, f"RS/RS{args.rs}")
