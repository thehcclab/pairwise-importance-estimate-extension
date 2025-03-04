import torch
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm

class MNISTBaseline_Model(torch.nn.Module):
    def __init__(self, device:torch.device, input_dim:int, classes:int):
        super().__init__()

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device

        # version 1
        self.layer_0= torch.nn.Linear(input_dim, 800)
        self.layer_1= torch.nn.Linear(800,classes)

        # version 2
        # self.layer_0= torch.nn.Linear(input_dim, 2500)
        # self.layer_1= torch.nn.Linear(2500, 2000)
        # self.layer_2= torch.nn.Linear(2000, 1500)
        # self.layer_3= torch.nn.Linear(1500, 1000)
        # self.layer_4= torch.nn.Linear(1000, 500)
        # self.layer_5= torch.nn.Linear(500,classes)

        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None
    def forward(self,x):
        # version 1
        return self.layer_1( torch.tanh( self.layer_0(x) ) ).softmax(dim=1)

        # version 2
        # return self.layer_5( torch.tanh(self.layer_4( torch.tanh(self.layer_3( torch.tanh(self.layer_2( torch.tanh(self.layer_1( torch.tanh(self.layer_0(x)) )) )) )) )) ).softmax(dim=1)

    def set_optimiser(self, optimiser):
        self.optimser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

    def training_procedure(self, iteration, train_dataloader, val_dataloader, path, loss_func, optimiser, train_func, print_cycle):
        device= self.device

        path= path+"-dictionary"
        if not os.path.exists(path):
            os.makedirs(path)

        if self.optimiser==None:
            self.optimiser= optimiser

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
