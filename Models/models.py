import torch
from typing import List
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm
import yaml

class Univariate_Model(torch.nn.Module):
    def __init__(self, device:torch.device):

        super().__init__()

        assert isinstance(device, torch.device), "device is nto a torch.device"

        self.device= device

        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None

    def set_optimiser(self, optimiser):

        self.optimiser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

    def training_procedure(self, iteration, train_dataloader, val_dataloader, path, loss_func, optimiser, train_func, print_cycle):
        device= self.device

        if self.optimiser==None:
            self.optimiser= optimiser

        path= path+"-dictionary"
        if not os.path.exists(path):
            os.makedirs(path)

        for i in tqdm(range(iteration), desc="Iterations"):
            t_loss, t_dict= Model_Func.training(self, train_dataloader, train_func, loss_func, device)

            self.t_dicts += [t_dict]
            # pickle.dump(t_dict, open(path+"/t_dict-"+str(self.epoch)+".pkl", "wb"))

            print("Epoch ", i, ", loss", t_loss)

            if i % print_cycle==0:
                v_loss, v_dict= Model_Func.validation(self, val_dataloader, loss_func, device)
                pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

                self.v_dicts += [v_dict]

                t_acc, v_acc= t_dict["accuracy"], v_dict["accuracy"]
                t_recall, v_recall= t_dict["macro avg"]["recall"], v_dict["macro avg"]["recall"]
                t_prec, v_prec= t_dict["macro avg"]["precision"], v_dict["macro avg"]["precision"]
                t_f1, v_f1= t_dict["macro avg"]["f1-score"], v_dict["macro avg"]["f1-score"]

#                 print("Epoch: ", i)
#                 print("t_loss: ", t_loss,", v_loss: ", v_loss)
#                 print("t_acc: ", t_acc,", v_acc: ", v_acc)
#                 print("t_recall: ", t_recall,", v_recall: ", v_recall)
#                 print("t_prec: ", t_prec, ", v_prec: ", v_prec)
#                 print("t_f: ", t_f1,", v_f: ", v_f1)
#                 print("////////")

                logging.info("Epoch: "+str(i))
                logging.info("t_loss: "+str(t_loss)+", v_loss: "+str(v_loss))
                logging.info("t_acc: "+str(t_acc)+", v_acc: "+str(v_acc))
                logging.info("t_recall: "+str(t_recall)+", v_recall: "+str(v_recall))
                logging.info("t_prec: "+str(t_prec)+", v_prec: "+str(v_prec))
                logging.info("t_f: "+str(t_f1)+", v_f: "+str(v_f1))
                logging.info("//////////")

            self.epoch += 1


class Univariate_Subset(Univariate_Model):
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module):

        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not a torch.nn.Module"

        self.model= model
        self.layer_weights= torch.ones(input_dim, dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, x):
        new_input= x * self.layer_weights

        return self.model(new_input)

    def return_layer_weights(self):
        return self.layer_weights.detach().cpu().numpy()

    def subset_layer_weights(self, array):
        self.layer_weights.copy_( torch.tensor(array, dtype=torch.float32) )

class Univariate_RS(Univariate_Subset):
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module, rs_path:str, rs_selection=0):

        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not a torch.nn.Module"

        super().__init__(device, input_dim, model)

        assert isinstance(rs_path, str), "dataset is nto str"

        if rs_path[:6] == "scikit":
            rs_path += "-"

        with open(f"./experiments/{rs_path}rs{rs_selection}.yaml", 'r') as stream:
            self.dictionary= yaml.safe_load(stream)

#         self.layer_weights= torch.ones(input_dim, dtype=torch.float32, requires_grad=False, device=device)

#         self.epoch=1
#         self.t_dicts=[]
#         self.v_dicts=[]

#     def forward(self, x):
#         new_input= x * self.layer_weights

#         return self.model(new_input)

#     def return_layer_weights(self):
#         return self.layer_weights.detach().cpu().numpy()

    def select_percentile(self, percentile:int):

        array=[ 1. for i in range(len(self.layer_weights)) ]
        for i in self.dictionary[percentile]:
                array[i]= 0

        # self.subset_layer_weights(array)
        return array


class Univariate_IELayer(Univariate_Model):
    """Simultaneous method"""
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module):

        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not torch.nn.Module"

        self.model= model

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))


    def return_layer_weights(self):

        return self.IE_weights.detach().cpu().numpy()

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights

        return self.model(new_input)


class Univariate_IEGradient(Univariate_IELayer):
    def __init__(self, device: torch.device, input_dim: int, model: torch.nn.Module):

        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not a torch.nn.Module"
        super().__init__(device, input_dim, model)

        self.IE_grad=[]

    def return_IE_gradient(self):
        return self.IE_weights.grad

    def store_IE_gradient(self):
        if self.return_IE_gradient()!=None:
            self.IE_grad.append( self.return_IE_gradient().clone().detach().cpu().numpy()  )

    def IE_grad_setting(self, boolean):
        self.IE_weights.requires_grad_(boolean)
        self.IE_weights.grad= None

    def return_IE_grad(self):
        return self.IE_grad


class Univariate_ThresholdedIE(Univariate_IELayer):
    def __init__(self, device: torch.device, input_dim: int, model: torch.nn.Module):

        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not a torch.nn.Module"
        super().__init__(device, input_dim, model)

#         self.IE_weights= torch.nn.Parameter(torch.nn.init.uniform_(self.IE_weights))
#         self.IE_weights= torch.nn.Parameter(torch.nn.init.normal_(self.IE_weights, mean=0.0, std=0.5))
        self.IE_weights_cached= torch.empty(input_dim, dtype=torch.float32, requires_grad=False, device=device)

    def cache_IE_weights(self):
        self.IE_weights_cached.copy_(self.IE_weights)

#     def return_IE_weights(self):
#         return self.IE_weights.detach().cpu().numpy()

    def threshold_layer(self, upper_bound=1, lower_bound=0):
        selection_bool= ( ((self.IE_weights<lower_bound)|(self.IE_weights>upper_bound)).type(torch.float))
#         if self.IE_weights > upper_bound:
        diff= self.IE_weights_cached-self.IE_weights

#         print("selection_bool", grad_bool[:5] )
#         print("diff", diff[:5])
#         print()
#         print("+", self.IE_weights + (grad_bool*diff) )
#         print("-", self.IE_weights - (grad_bool*diff))

        return self.IE_weights + (selection_bool*diff)


# ( (self.IE_weights-self.return_IE_gradient() > 1).type(torch.float) * -self.return_IE_gradient() )
#         tmp= (self.IE_weights > 1).type(torch.float) + ((self.IE_weights < -1).type(torch.float) * -1) + torch.where((self.IE_weights<=1)&(self.IE_weights>=-1), self.IE_weights, torch.tensor(0.))

#     def compute_l1_loss(self):
#         return self.IE_weights.sum()

#     def compute_l2_loss(self):
#         return (self.IE_weights * self.IE_weights).sum()

    def forward(self, x):
#         print("before", self.IE_weights)
#         print("thresholded: ", self.threshold_layer())
        new_input= x * self.IE_weights

        return self.model(new_input)


class Univariate_DF(Univariate_Model):
    def __init__(self, device: torch.device, input_dim: int, model: torch.nn.Module, l1_lambda=1.0, l2_lambda=0.1):
        """
        input_dim: dimensions of input data, usually in the form of [time_domain_dim, feature_domain_dim]
        model: Existing Pytorch torch.nn.Module model
        l1_lambda, l2_lambda: variables for the parameter of the Elastic Net regularisation
        mode: 'dc', data channels, or 'ts', time steps.
            'dc'-> measures importance of feature_domain data channels
            'ts'-> measures importance of time_domain channels
        """
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(model, torch.nn.Module), "Expecting torch.nn.module for the model"
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting flaot for l2_lambda"
        assert isinstance(input_dim, int), "Expecting int for input_dim"

        self.model= model
        self.l1_lambda= l1_lambda
        self.l2_lambda= l2_lambda
        self.DF_weights=torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.DF_weights= torch.nn.Parameter( torch.nn.init.uniform_(self.DF_weights) )

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights * self.DF_weights).sum()

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def forward(self,x):
#         print(x.shape, self.DF_ts.shape, self.DF_dc.shape)
        new_input= x * self.DF_weights
#         print("new input", new_input.shape)

        output= self.model(new_input)
        return output


class Univariate_NeuralFS(Univariate_Model):
    def __init__(self, device: torch.device, input_dim:int, nonlinear_func: torch.nn.Module, decision_net: torch.nn.Module):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for nonlinear_func"
        assert isinstance(decision_net, torch.nn.Module), "Expecting torch.nn.Module for decision_net"
        assert isinstance(input_dim, int), "Expecting an int for input_dim"

        self.nonlinear_func= nonlinear_func
        self.decision_net= decision_net
        self.pairwise_var=torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.pairwise_var= torch.nn.init.uniform_(self.pairwise_var)

    def return_pairwise_weights(self):
        return self.pairwise_var.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.009):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x, threshold=0.009):
        nonlinear_output= self.nonlinear_func(x)

        pairwise_connected_output= self.Thresholded_Linear(nonlinear_output * self.pairwise_var, threshold)

#         print("pairwise",pairwise_connected_output.size())
        selected_input= x * pairwise_connected_output
#         print("selected",selected_input.size(), "x", x.size())

        output= self.decision_net(selected_input)
        return output
