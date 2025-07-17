import sys
sys.path[0]='/home/phd-zhang.ziyang/python/Federated-Cross-Modal-Hashing'
from .models05_04 import ImgModule, TxtModule, TransferModule
import copy
import random
from .options05_04 import args_parser
import os
from .task05_04 import *
from .data_handler import *
from .utils import *
import datetime
import logging
import matplotlib.pyplot as plt

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        record_time = datetime.datetime.fromtimestamp(record.created) + datetime.timedelta(hours=8)  # 加8小时
        if datefmt:
            return record_time.strftime(datefmt)
        else:
            return record_time.strftime("%Y-%m-%d %H:%M:%S")
        
def train(opt,time_str,logger,filename):
    # loading and splitting data
    images, tags, labels = load_data(opt)
    X, Y, L = split_data(images, tags, labels, opt)
    if opt.backbone=='DCMH':
        user_groups = iid(opt, L['train'])
    elif opt.backbone=='SHDCH':
        S12 = np.zeros([opt.num_class1, opt.num_class2])
        index1 = 0
        index2 = 0
        for i in range(0, len(labels)):
            for j in range(0, len(labels[0])):
                if (labels[i][j] == 1 and j < opt.num_class1):
                    index1 = j
                elif labels[i][j] == 1 and j >= opt.num_class1:
                    index2 = j
            S12[index1][index2 - opt.num_class1] = 1
        S12=torch.from_numpy(S12).float().cuda()
        user_groups = hetero_dir_partition(opt, L['train'][:,opt.num_class1:],opt.beta)

    print('...loading and splitting data finish')

    # build models
    y_dim = Y['train'].shape[1]
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    if opt.backbone=='DCMH':
        server_img_model = ImgModule(opt.bit, pretrain_model, L['train'].shape[1]).cuda()
        server_txt_model = TxtModule(y_dim, opt.bit, L['train'].shape[1]).cuda()
        server_transfer_model=TransferModule(L['train'].shape[1],opt.bit,opt).cuda()
    elif opt.backbone=='SHDCH':
        server_img_model = ImgModule(opt.bit, pretrain_model, opt.num_class2).cuda()
        server_txt_model = TxtModule(y_dim, opt.bit, opt.num_class2).cuda()
        server_transfer_model=TransferModule(opt.num_class2,opt.bit,opt).cuda()

    print('...Structure initialization is completed...')

    F_buffer = {}
    G_buffer = {}
    F_buffer_transfer={}
    G_buffer_transfer={}
    img_cls_loss = {}
    txt_cls_loss = {}
    B = {}
    Y1={}
    Y2={}

    models_img, models_txt, models_transfer = [], [], []

    for client in range(opt.num_users):
        model_img = copy.deepcopy(server_img_model)
        model_txt = copy.deepcopy(server_txt_model)
        model_transfer=copy.deepcopy(server_transfer_model)

        models_img.append(model_img)
        models_txt.append(model_txt)
        models_transfer.append(model_transfer)
        num_train = len(user_groups[client])
        F_buffer[client] = torch.randn(num_train, opt.bit)
        G_buffer[client] = torch.randn(num_train, opt.bit)
        F_buffer_transfer[client] = torch.randn(num_train, opt.bit)
        G_buffer_transfer[client] = torch.randn(num_train, opt.bit)
        B[client] = torch.sign(F_buffer[client] + G_buffer[client])

        
        F_buffer[client] = F_buffer[client].cuda()
        G_buffer[client] = G_buffer[client].cuda()
        F_buffer_transfer[client] = F_buffer_transfer[client].cuda()
        G_buffer_transfer[client] = G_buffer_transfer[client].cuda()
        B[client] = B[client].cuda()

        if opt.backbone=='SHDCH':
            Y1[client] = torch.randn(opt.num_class1,opt.bit)
            Y2[client] = torch.randn(opt.num_class2,opt.bit)
            Y1[client] = Y1[client].cuda()
            Y2[client] = Y2[client].cuda()
            
    
    # training start
    if opt.backbone=='DCMH':       
        Fed_taskheter_DCMH(models_img, models_txt, models_transfer,opt, X, Y, L, user_groups, server_img_model, server_txt_model, server_transfer_model, F_buffer, G_buffer, F_buffer_transfer, G_buffer_transfer, img_cls_loss, txt_cls_loss, B,logger,filename,time_str)
    elif opt.backbone=='SHDCH':
        Fed_taskheter_SHDCH(models_img, models_txt, models_transfer,opt, X, Y, L, user_groups, S12,server_img_model, server_txt_model, server_transfer_model, F_buffer, G_buffer, F_buffer_transfer, G_buffer_transfer, img_cls_loss, txt_cls_loss, B,Y1,Y2,logger,filename,time_str)



if __name__ == '__main__':
    # default parameter settings
    now = datetime.datetime.now()
    current_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    opt = args_parser()
    filename=time_str+'_FedSEH_'+opt.dataset
    if not os.path.exists('./log'):
        os.makedirs('./log')
    logging.basicConfig(filename='./log/'+filename+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.setFormatter(formatter)


    
    logger.info(f"Arguments: {vars(opt)}")
    

    # set random seeds
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.device == 'cuda':
        torch.cuda.set_device(int(opt.gpu))
        torch.cuda.manual_seed(opt.seed)
        torch.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    #training
    train(opt,time_str,logger,filename)

