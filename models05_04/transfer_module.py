import torch
from torch import nn
from utils import BasicModule

class TransferModule(BasicModule):
    def __init__(self, cate_number,bit,opt):
        super(TransferModule, self).__init__()
        self.module_name = "transfer_model"


        self.transfer_1=nn.Linear(in_features=cate_number,out_features=bit)
        self.transfer_1.weight.data = torch.randn(bit,cate_number) * 0.01
        self.transfer_1.bias.data = torch.randn(bit) * 0.01

        if opt.activation=='ReLU':
            self.activation=nn.ReLU()
        elif opt.activation=='LeakyReLU':
            self.activation=nn.LeakyReLU()
        elif opt.activation=='ELU':
            self.activation=nn.ELU()
        elif opt.activation=='Sigmoid':
            self.activation=nn.Sigmoid()


        self.transfer_2=nn.Linear(in_features=bit,out_features=bit)
        self.transfer_2.weight.data = torch.randn(bit,bit) * 0.01
        self.transfer_2.bias.data = torch.randn(bit) * 0.01



    def forward(self, x):

        x=self.transfer_1(x)
        x=self.activation(x)
        x=self.transfer_2(x)

        return  x


