import torch
import pandas as pd
import numpy as np

import math

from utilities.preproc_func import *

def create_simulation_dataset(instances=3000):
    dictionary= {
            "x0":[],
            "x1":[],
            "x2":[],
            "x3":[],
            "x4":[],
            "x5":[],
            "x6":[],
            "x7":[],
            "x8":[],
            "x9":[],
            "x10":[],
            "x11":[],
            "x12":[],
            "x13":[],
            "y":[]
        }

    for _ in range(instances):
            x0= np.random.uniform()
            x1= np.random.uniform()
            x2= np.random.uniform()
            x3= np.random.uniform()
            x4= np.random.uniform()

            y= 10*math.sin(math.pi*x0*x1)-20*(x2-0.5)**2+10*x3+5*x4+np.random.uniform()

            x10= x0+np.random.normal(0,0.01)
            x11= x1+np.random.normal(0,0.01)
            x12= x2+np.random.normal(0,0.01)
            x13= x3+np.random.normal(0,0.01)

            x5= np.random.uniform()*0.1
            x6= np.random.uniform()*0.1
            x7= np.random.uniform()*0.1
            x8= np.random.uniform()*0.1
            x9= np.random.uniform()*0.1


            dictionary["x0"].append(x0)
            dictionary["x1"].append(x1)
            dictionary["x2"].append(x2)
            dictionary["x3"].append(x3)
            dictionary["x4"].append(x4)
            dictionary["x5"].append(x5)
            dictionary["x6"].append(x6)
            dictionary["x7"].append(x7)
            dictionary["x8"].append(x8)
            dictionary["x9"].append(x9)
            dictionary["x10"].append(x10)
            dictionary["x11"].append(x11)
            dictionary["x12"].append(x12)
            dictionary["x13"].append(x13)
            dictionary["y"].append(y)

    return pd.DataFrame(dictionary)

from utilities.Dataset import *
# class SubsetDataset(torch.utils.data.Dataset):
#     def __init__(self, X:pd.DataFrame, y:pd.DataFrame):
#
#         super().__init__()
#
#         self.X= X
#         self.y= y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, index):
#         return self.X[index], self.y[index]

class SimulationStudyDataset(Dataset):

    def __init__(self,default_path="./experiments/simul_study/", split=0.3): #init or load
        # assert protocol=="init" or protocol=="load", "\"load\" or \"init\""
        super().__init__()

        try:
            data= pd.read_csv(f"{default_path}simulation_study_data.csv", index_col=0)
        except Exception:
            data= create_simulation_dataset()
            data.to_csv(f"{default_path}simulation_study_data.csv")

        self.y= np.array(data["y"])
        self.init_y()
# X= np.array(data[["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]])
        self.X= np.array(data[["x0", "x1", "x2", "x7", "x8", "x9", "x10", "x11", "x12", "x13"]])


        try:
            self.train_indices, self.val_indices= pickle.load(open(default_path+"indices.pkl", "rb"))
        except FileNotFoundError:
            self.train_indices, self.val_indices= create_indices(self.y, split=split)
            pickle.dump([self.train_indices, self.val_indices], open(default_path+"indices.pkl", "wb"))

        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]
        self.classes= len( np.unique(self.y) )

    def init_y(self):
        binary_y= self.y>=self.y.mean()
        self.y= binary_y.astype(float)
