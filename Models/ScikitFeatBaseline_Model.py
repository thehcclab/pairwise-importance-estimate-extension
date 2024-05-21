import torch
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm

class ScikitFeatBaseline_Model(torch.nn.Module):
    def __init__(self, dataset:str, device:torch.device, input_dim:int, classes:int, lr=0.001):
        super().__init__()

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device
        self.lr= lr

#         self.layer_0= torch.nn.Linear(input_dim, 100)
#         self.layer_1= torch.nn.Linear(100,classes)

#         self.layer_0= torch.nn.Linear(input_dim, 117) 3/6 0.96666
#         self.layer_1= torch.nn.Linear(117,138)
#         self.layer_2= torch.nn.Linear(138, classes)

        if dataset=="USPS":
#             self.layer_0= torch.nn.Linear(input_dim, 197) # 3/6 0.96666 worse
#             self.layer_1= torch.nn.Linear(197,113)
#             self.layer_2= torch.nn.Linear(113, classes)
#             self.output= self.layer_2( self.tanh( self.layer_1( self.relu( self.layer_0() ) ) ) ).softmax(dim=1)
            self.model= torch.nn.Sequential(
                torch.nn.Linear(input_dim, 197),
                torch.nn.ReLU(),
                torch.nn.Linear(197,113),
                torch.nn.Tanh(),
                torch.nn.Linear(113,classes),
                torch.nn.Softmax(dim=1)
            )
        elif dataset=="BASEHOCK":
#             self.layer_0= torch.nn.Linear(input_dim, 151)
#             self.layer_1= torch.nn.Linear(151,184)
#             self.layer_2= torch.nn.Linear(184,168)
#             self.layer_3= torch.nn.Linear(168,classes)
#             self.output= self.layer_3( self.layer_2( self.layer_1( self.layer_0() ).sigmoid() ) ).softmax(dim=1)
            self.model= torch.nn.Sequential(
                torch.nn.Linear(input_dim, 151),
                torch.nn.Linear(151,184),
                torch.nn.Sigmoid(),
                torch.nn.Linear(184,168),
                torch.nn.Linear(168,classes),
                torch.nn.Softmax(dim=1)
            )
        elif dataset=="PCMAC":
            self.model= torch.nn.Sequential(
                torch.nn.Linear(input_dim, 58),
                torch.nn.ReLU(),
                torch.nn.Linear(58,97),
                torch.nn.ReLU(),
                torch.nn.Linear(97,101),
                torch.nn.Linear(101,4),
                torch.nn.Linear(4,classes),
                torch.nn.Softmax(dim=1)
            )
        elif dataset=="RELATHE":
            self.model= torch.nn.Sequential(
                torch.nn.Linear(input_dim, 99),
                torch.nn.Tanh(),
                torch.nn.Linear(99,113),
                torch.nn.ReLU(),
                torch.nn.Linear(113,66),
                torch.nn.ReLU(),
                torch.nn.Linear(66,classes),
                torch.nn.Softmax(dim=1)
            )
        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None

    def forward(self,x):
        return self.model(x)
#         return self.layer_1( self.layer_0(x) ).softmax(dim=1)

    def set_optimiser(self,optimiser):
        self.optimiser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

    def training_procedure(self, iteration, train_dataloader, val_dataloader, path, loss_func, optimiser, train_func, print_cycle):
        device= self.device

        path= path+"-dictionary"
        if not os.path.exists(path):
            os.makedirs(path)

        if self.optimiser==None:
            self.optimiser=optimiser

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
