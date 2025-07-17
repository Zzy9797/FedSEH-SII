import torch
from tqdm import tqdm
from .utils import *
from torch.autograd import Variable
import numpy as np
import gc
import copy
import os
from fast_pytorch_kmeans import KMeans
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import cvxpy as cp




def valid(img_model, txt_model, server_transfer_model,query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, opt):
    qBX = generate_image_code(img_model, server_transfer_model,query_x, opt)
    qBY = generate_text_code(txt_model, server_transfer_model,query_y, opt,)
    rBX = generate_image_code(img_model, server_transfer_model,retrieval_x, opt)
    rBY = generate_text_code(txt_model, server_transfer_model,retrieval_y, opt)
    query_L = query_L.cuda()
    retrieval_L = retrieval_L.cuda()
    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i


def generate_image_code(img_model, transfer_model,X, opt):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, opt.bit, dtype=torch.float).cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float).cuda()
        l_predict,cur_f= img_model(image)
        hash_transfer=transfer_model(l_predict)
        cur_f=cur_f+opt.e*hash_transfer
        B[ind, :] = cur_f.data
    B = torch.sign(B)    
    return B


def generate_text_code(txt_model, transfer_model,Y, opt):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, opt.bit, dtype=torch.float).cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float).cuda()
        l_predict,cur_g= txt_model(text)
        hash_transfer=transfer_model(l_predict)
        cur_g=cur_g+opt.e*hash_transfer
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def weight_fed(opt,agg_number,local_F_buffer_transfer_dict,local_G_buffer_transfer_dict):
    n = len((agg_number))
    p1 = np.array(agg_number)/np.sum(agg_number)  
    sim=[]

    for j in range(opt.num_users):
        sim.append(torch.sum(torch.pow(local_F_buffer_transfer_dict[j]-local_G_buffer_transfer_dict[j], 2)).detach().cpu().numpy()/opt.batch_size)
    sim=np.array(sim)
    e_x = np.exp(-sim/opt.T2) 
    sim=e_x/e_x.sum()
    P = np.identity(n)                    
    P = cp.atoms.affine.wraps.psd_wrap(P)         
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)

    q = 2*(opt.wfe)*p1+2*(1-opt.wfe)*sim
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x),     
                    [G @ x <= h,
                    A @ x == b]
                    )
    prob.solve()

    w=torch.from_numpy(x.value).unsqueeze(0)


    return w

def weight_fed_encoder(opt,agg_number,cls_loss_dict):

    n = len((agg_number))    
    p1 = np.array(agg_number)/np.sum(agg_number)   
    p2 = np.array(list(cls_loss_dict.values())) 

    e_x = np.exp(p2/opt.T) 
    p2=e_x / e_x.sum()  
    P = np.identity(n)                   
    P = cp.atoms.affine.wraps.psd_wrap(P)         
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)

    q = 2*(opt.wfe)*p1+2*(1-opt.wfe)*p2
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x),     
                    [G @ x <= h,
                    A @ x == b]
                    )
    prob.solve()

    w=torch.from_numpy(x.value).unsqueeze(0)

    return w

def weighted_agg_fed(w, modal_w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * modal_w[0][0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * modal_w[0][i]

    return w_avg

class LocalImgUpdate_DCMH(object):
    def __init__(self, opt, F_buffer, G_buffer, F_buffer_transfer, B, train_img, train_L, train_txt):
        self.opt = opt
        self.F_buffer = F_buffer
        self.G_buffer = G_buffer
        self.F_buffer_transfer = F_buffer_transfer
        self.B = B
        self.train_image = train_img
        self.train_L = train_L
        self.train_text = train_txt
        self.Sim = calc_neighbor(opt, train_L, train_L)
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, model,model_transfer):
        # set mode to train model
        num_train = self.train_image.shape[0]
        batch_size = self.opt.batch_size
        ones_ = torch.ones(num_train - batch_size, 1).cuda()
        optimizer_img = torch.optim.SGD([{'params': model.parameters(), 'lr': self.opt.lr}, {'params': model_transfer.parameters(), 'lr': self.opt.lr2}])

        model.train()
        model_transfer.train()
        loss_value=[]

        # train image net
        for i in tqdm(range(num_train // batch_size)):  
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(self.train_L[ind, :]).cuda()
            image = Variable(self.train_image[ind].type(torch.float)).cuda()
            ones = self.ones.cuda()

            S = calc_neighbor(self.opt, sample_L, self.train_L)

            l_predict,cur_f = model(image)
            hash_transfer=model_transfer(l_predict)
            cur_f=self.opt.e*hash_transfer+cur_f
            self.F_buffer[ind, :] = cur_f.data    
            self.F_buffer_transfer[ind,:] = hash_transfer.data
            F = Variable(self.F_buffer)
            G = Variable(self.G_buffer)

            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L/torch.sum(sample_L,dim=1).unsqueeze(-1), reduction='mean')
            loss_value.append(-predict_loss)
        

            # hash loss
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(self.B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
            loss_x = logloss_x + self.opt.gamma * quantization_x + self.opt.alphard * balance_x
            loss_x /= (num_train * batch_size)
            loss_x = (1.0 / self.opt.num_users)*loss_x

            loss_x=loss_x+self.opt.mu*predict_loss
            if (torch.isnan(loss_x).any()):
                continue

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

            mean_loss= torch.mean(torch.stack(loss_value))

        return self.F_buffer, self.F_buffer_transfer, mean_loss.item()

class LocalTxtUpdate_DCMH(object):
    def __init__(self, opt, F_buffer, G_buffer, G_buffer_transfer, B, train_txt, train_L, train_img):
        self.opt = opt
        self.F_buffer = F_buffer
        self.G_buffer = G_buffer
        self.G_buffer_transfer = G_buffer_transfer
        self.B = B
        self.train_text = train_txt
        self.train_image = train_img
        self.train_L = train_L
        self.Sim = calc_neighbor(opt, train_L, train_L)
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, model,model_transfer):
        # set mode to train model
        num_train = self.train_text.shape[0]
        batch_size = self.opt.batch_size
        ones_ = torch.ones(num_train - batch_size, 1).cuda()
        optimizer_txt = torch.optim.SGD([{'params': model.parameters(), 'lr': self.opt.lr}, {'params': model_transfer.parameters(), 'lr': self.opt.lr2}])
        model.train()
        model_transfer.train()
        loss_value=[]

        # train text net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(self.train_L[ind, :]).cuda()
            text = self.train_text[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text).cuda()
            ones = self.ones.cuda()


            # similar matrix
            S = calc_neighbor(self.opt, sample_L, self.train_L)

            l_predict,cur_g= model(text)
            hash_transfer=model_transfer(l_predict)
            cur_g=cur_g+self.opt.e*hash_transfer
            self.G_buffer[ind, :] = cur_g.data
            self.G_buffer_transfer[ind, :] = hash_transfer.data
            F = Variable(self.F_buffer)
            G = Variable(self.G_buffer)

            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L/torch.sum(sample_L,dim=1).unsqueeze(-1), reduction='mean')
            loss_value.append(-predict_loss)
         
            # hash loss
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(self.B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + self.opt.gamma * quantization_y + self.opt.alphard * balance_y
            loss_y /= (num_train * batch_size)
            loss_y = (1.0 / self.opt.num_users) * loss_y
            
            loss_y=loss_y+self.opt.mu*predict_loss

            if (torch.isnan(loss_y).any()):
                continue

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

            mean_loss= torch.mean(torch.stack(loss_value))

        return self.G_buffer, self.G_buffer_transfer, mean_loss.item()

class LocalImgUpdate_SHDCH(object):
    def __init__(self, opt, F_buffer, F_buffer_transfer, B, Y1,Y2,train_img, train_L, train_txt):
        self.opt = opt
        self.F_buffer = F_buffer
        self.F_buffer_transfer = F_buffer_transfer
        self.B = B
        self.Y1=Y1
        self.Y2=Y2
        self.train_image = train_img
        self.train_L = train_L
        self.train_text = train_txt
        self.Sim = calc_neighbor(opt, train_L, train_L)    
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, model,model_transfer):
        # set mode to train model
        num_train = self.train_image.shape[0]
        batch_size = self.opt.batch_size
        optimizer_img = torch.optim.Adam([{'params': model.parameters(), 'lr': self.opt.lr}, {'params': model_transfer.parameters(), 'lr': self.opt.lr2}])
        model.train()
        model_transfer.train()
        loss_value=[]

        # train image net
        for i in tqdm(range(num_train // batch_size)):   
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            
            sample_L = Variable(self.train_L[ind, :]).cuda()
            image = Variable(self.train_image[ind].type(torch.float)).cuda()

            l_predict,cur_f = model(image)
            hash_transfer=model_transfer(l_predict)
            cur_f=cur_f+self.opt.e*hash_transfer
            self.F_buffer[ind, :] = cur_f.data
            self.F_buffer_transfer[ind, :] =hash_transfer.data

    
            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L[:,self.opt.num_class1:], reduction='mean')
            loss_value.append(-predict_loss)
          

            # hash loss
            l1_x = torch.sum(torch.pow(self.opt.bit * sample_L[:,:self.opt.num_class1] - torch.matmul(cur_f, self.Y1.t()), 2))
            l2_x = torch.sum(torch.pow(self.opt.bit * sample_L[:,self.opt.num_class1:] - torch.matmul(cur_f, self.Y2.t()), 2))
            quantization_x = torch.sum(torch.pow(self.B[ind, :] - cur_f, 2))
            loss_x = self.opt.hash_l1*l1_x + (1-self.opt.hash_l1)*l2_x+self.opt.gamma * quantization_x 
            loss_x /= (num_train * batch_size)
            loss_x = (1.0 / self.opt.num_users)*loss_x

            loss_x=loss_x+self.opt.mu*predict_loss

            if (torch.isnan(loss_x).any()):
                continue

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

            mean_loss= torch.mean(torch.stack(loss_value))

        return self.F_buffer, self.F_buffer_transfer, mean_loss.item()

class LocalTxtUpdate_SHDCH(object):
    def __init__(self, opt, G_buffer, G_buffer_transfer, B, Y1, Y2, train_txt, train_L, train_img):
        self.opt = opt
        self.G_buffer = G_buffer
        self.G_buffer_transfer = G_buffer_transfer
        self.B = B
        self.Y1=Y1
        self.Y2=Y2
        self.train_text = train_txt
        self.train_image = train_img
        self.train_L = train_L
        self.Sim = calc_neighbor(opt, train_L, train_L)
        self.ones = torch.ones(opt.batch_size, 1).cuda()

    def update_weights_het(self, model,model_transfer):
        # set mode to train model
        num_train = self.train_text.shape[0]
        batch_size = self.opt.batch_size
        optimizer_txt = torch.optim.Adam([{'params': model.parameters(), 'lr': self.opt.lr}, {'params': model_transfer.parameters(), 'lr': self.opt.lr2}])
        model.train()
        model_transfer.train()
        loss_value=[]

        # train text net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]

            sample_L = Variable(self.train_L[ind, :]).cuda()
            text = self.train_text[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text).cuda()


            l_predict,cur_g= model(text)
            hash_transfer=model_transfer(l_predict)
            cur_g=cur_g+self.opt.e*hash_transfer
            self.G_buffer[ind, :] = cur_g.data
            self.G_buffer_transfer[ind, :] = hash_transfer.data

            # classification loss
            predict_loss = nn.functional.kl_div(l_predict.log(), sample_L[:,self.opt.num_class1:], reduction='mean')
            loss_value.append(-predict_loss)


            # hash loss
            l1_y = torch.sum(torch.pow(self.opt.bit * sample_L[:,:self.opt.num_class1] - torch.matmul(cur_g, self.Y1.t()), 2))
            l2_y = torch.sum(torch.pow(self.opt.bit * sample_L[:,self.opt.num_class1:] - torch.matmul(cur_g, self.Y2.t()), 2))
            quantization_y = torch.sum(torch.pow(self.B[ind, :] - cur_g, 2))
            loss_y = self.opt.hash_l1*l1_y + (1-self.opt.hash_l1)*l2_y+self.opt.gamma * quantization_y
            loss_y /= (num_train * batch_size)
            loss_y = (1.0 / self.opt.num_users) * loss_y

            loss_y=loss_y+self.opt.mu*predict_loss


            if (torch.isnan(loss_y).any()):
                continue

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

            mean_loss= torch.mean(torch.stack(loss_value))

        return self.G_buffer, self.G_buffer_transfer, mean_loss.item()

def Fed_taskheter_DCMH(models_img, models_txt, models_transfer,opt, X, Y, L, user_groups, server_img_model, server_txt_model, server_transfer_model,local_F_buffer_dict, local_G_buffer_dict, local_F_buffer_transfer_dict, local_G_buffer_transfer_dict, img_cls_loss_dict, txt_cls_loss_dict, local_B_dict,logger,filename,time_str):
    train_L = torch.from_numpy(L['train']).cuda().float()
    train_x = torch.from_numpy(X['train']).cuda()
    train_y = torch.from_numpy(Y['train']).cuda()

    query_L = torch.from_numpy(L['query']).float()
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval']).float()
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    max_mapi2t = max_mapt2i = 0.


    rounds = []
    i_t = []
    t_i = []
    rounds.append(0)
    i_t.append(0)
    t_i.append(0)

    for r in range(opt.rounds):
        w_img_glob, w_txt_glob,w_transfer_glob = [], [], []
        agg_number=[]
        
        for j in range(opt.num_users):
            # update local weights every round
            models_img[j] = copy.deepcopy(server_img_model)
            models_txt[j] = copy.deepcopy(server_txt_model)
            models_transfer[j]=copy.deepcopy(server_transfer_model)

            # pick up the certain client's training image, text and label dataset
            idxs = user_groups[j]
            local_train_x = train_x[idxs]
            local_train_y = train_y[idxs]
            local_train_L = train_L[idxs]
            
            agg_number.append(len(idxs))

            Sim = calc_neighbor(opt, local_train_L, local_train_L)

            for epoch in range(opt.max_epoch):
                # train image net
                local_img_model = LocalImgUpdate_DCMH(opt=opt, F_buffer=local_F_buffer_dict[j], G_buffer=local_G_buffer_dict[j], F_buffer_transfer=local_F_buffer_transfer_dict[j],
                                                 B=local_B_dict[j], train_img=local_train_x, train_L=local_train_L, train_txt=local_train_y)
                local_F_buffer, local_F_buffer_transfer, img_cls_loss = local_img_model.update_weights_het(models_img[j],models_transfer[j])
                local_F_buffer_dict[j] = local_F_buffer
                local_F_buffer_transfer_dict[j]=local_F_buffer_transfer
                img_cls_loss_dict[j] = img_cls_loss

                # train text net
                local_txt_model = LocalTxtUpdate_DCMH(opt=opt, F_buffer=local_F_buffer_dict[j], G_buffer=local_G_buffer_dict[j], G_buffer_transfer=local_G_buffer_transfer_dict[j],
                                                 B=local_B_dict[j], train_txt=local_train_y, train_L=local_train_L, train_img=local_train_x)
                local_G_buffer, local_G_buffer_transfer, txt_cls_loss = local_txt_model.update_weights_het(models_txt[j],models_transfer[j])
                local_G_buffer_dict[j] = local_G_buffer
                local_G_buffer_transfer_dict[j] = local_G_buffer_transfer
                txt_cls_loss_dict[j] = txt_cls_loss

                # update B 
                local_B = torch.sign(local_F_buffer + local_G_buffer)
                local_B_dict[j] = local_B

                # calculate total loss 
                loss = calc_loss_DCMH(local_B, local_F_buffer, local_G_buffer, Variable(Sim), opt)
                print('...round: %3d, client: %3d, epoch: %3d, loss: %3.3f' % (r + 1, j + 1, epoch + 1, loss.data))
                logger.info('...round: %3d, client: %3d, epoch: %3d, loss: %3.3f' % (r + 1, j + 1, epoch + 1, loss.data))
            w_img = models_img[j].state_dict()
            w_img_glob.append(w_img)
            w_txt = models_txt[j].state_dict()
            w_txt_glob.append(w_txt)
            w_transfer=models_transfer[j].state_dict()
            w_transfer_glob.append(w_transfer)


        modal_w = weight_fed(opt,agg_number,local_F_buffer_transfer_dict,local_G_buffer_transfer_dict)
        modal_img = weight_fed_encoder(opt,agg_number,img_cls_loss_dict)
        modal_txt = weight_fed_encoder(opt,agg_number,txt_cls_loss_dict)

        # calculate updated global weights
        w_img_new = weighted_agg_fed(w_img_glob, modal_img)
        w_txt_new = weighted_agg_fed(w_txt_glob, modal_txt)
        w_transfer_new=weighted_agg_fed(w_transfer_glob,modal_w)
        server_img_model.load_state_dict(w_img_new)
        server_txt_model.load_state_dict(w_txt_new)
        server_transfer_model.load_state_dict(w_transfer_new)

        # send updated global weights to every client
        for i in range(opt.num_users):
            models_img[i] = copy.deepcopy(server_img_model)
            models_txt[i] = copy.deepcopy(server_txt_model)
            models_transfer[i]=copy.deepcopy(server_transfer_model)

        
        mapi2t, mapt2i = valid(server_img_model, server_txt_model, server_transfer_model,query_x, retrieval_x, query_y, retrieval_y,
                                       query_L, retrieval_L, opt)
        rounds.append(r+1)
        i_t.append((mapi2t*100).cpu().numpy())
        t_i.append((mapt2i*100).cpu().numpy())
        print('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (r + 1, mapi2t, mapt2i))
        logger.info('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (r + 1, mapi2t, mapt2i))
        plt.plot(rounds, i_t, label='mapi2t', linestyle='-', color='b')
        plt.plot(rounds, t_i, label='mapt2i', linestyle='-', color='r')

        plt.xlabel('round')
        plt.ylabel('map')
        plt.legend()
        

        plt.xlim(0, opt.rounds)
        plt.ylim(0, 100)
        

        if not os.path.exists('./plt/'):
            os.makedirs('./plt/')
        plt.savefig('./plt/'+filename+'.png')  
        plt.clf() 

        max_mapi2t = max(mapi2t,max_mapi2t)
        max_mapt2i = max(mapt2i,max_mapt2i)


                
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        logger.info('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))


    print('...training procedure finish')
    logger.info('...training procedure finish')


def Fed_taskheter_SHDCH(models_img, models_txt,models_transfer ,opt, X, Y, L, user_groups, S12,server_img_model, server_txt_model, server_transfer_model,local_F_buffer_dict, local_G_buffer_dict, local_F_buffer_transfer_dict, local_G_buffer_transfer_dict, img_cls_loss_dict, txt_cls_loss_dict, local_B_dict,local_Y1_dict,local_Y2_dict,logger,filename,time_str):
    train_L = torch.from_numpy(L['train']).cuda().float()
    train_x = torch.from_numpy(X['train']).cuda()
    train_y = torch.from_numpy(Y['train']).cuda()

    query_L = torch.from_numpy(L['query']).float()
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval']).float()   
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    max_mapi2t = max_mapt2i = 0.


    rounds = []
    i_t = []
    t_i = []
    rounds.append(0)
    i_t.append(0)
    t_i.append(0)

    for r in range(opt.rounds):
        agg_number=[]

        w_img_glob, w_txt_glob,w_transfer_glob = [], [],[]
        for j in range(opt.num_users):
            # update local weights every round
            models_img[j] = copy.deepcopy(server_img_model)
            models_txt[j] = copy.deepcopy(server_txt_model)
            models_transfer[j]=copy.deepcopy(server_transfer_model)

            # pick up the certain client's training image, text and label dataset
            idxs = user_groups[j]   
            local_train_x = train_x[idxs]
            local_train_y = train_y[idxs]
            local_train_L = train_L[idxs] #(num_train,C)
            agg_number.append(len(idxs))

            for epoch in range(opt.max_epoch):
                # train image net
                local_img_model = LocalImgUpdate_SHDCH(opt=opt, F_buffer=local_F_buffer_dict[j], F_buffer_transfer=local_F_buffer_transfer_dict[j],
                                                 B=local_B_dict[j], Y1=local_Y1_dict[j],Y2=local_Y2_dict[j],train_img=local_train_x, train_L=local_train_L, train_txt=local_train_y)
                local_F_buffer, local_F_buffer_transfer, img_cls_loss = local_img_model.update_weights_het(models_img[j],models_transfer[j])
                local_F_buffer_dict[j] = local_F_buffer #(num_train,bit)
                local_F_buffer_transfer_dict[j] = local_F_buffer_transfer

                img_cls_loss_dict[j] = img_cls_loss

                # train text net
                local_txt_model = LocalTxtUpdate_SHDCH(opt=opt, G_buffer=local_G_buffer_dict[j], G_buffer_transfer=local_G_buffer_transfer_dict[j],
                                                 B=local_B_dict[j], Y1=local_Y1_dict[j],Y2=local_Y2_dict[j], train_txt=local_train_y, train_L=local_train_L, train_img=local_train_x)
                local_G_buffer, local_G_buffer_transfer, txt_cls_loss = local_txt_model.update_weights_het(models_txt[j],models_transfer[j])
                local_G_buffer_dict[j] = local_G_buffer
                local_G_buffer_transfer_dict[j] = local_G_buffer_transfer
                txt_cls_loss_dict[j] = txt_cls_loss

                # update B 
                local_B = torch.sign(local_F_buffer + local_G_buffer)
                local_B_dict[j] = local_B

                # update Y1 and Y2

                Q1 = opt.bit * (opt.hash_l1 * (torch.matmul(local_F_buffer_dict[j].t(), local_train_L[:,0:opt.num_class1]) + torch.matmul(local_G_buffer_dict[j].t(), local_train_L[:,0:opt.num_class1])) +
                        opt.hash2 * torch.matmul(local_Y2_dict[j].t(), S12.t()))
                Q2 = opt.bit * ((1-opt.hash_l1) * (torch.matmul(local_F_buffer_dict[j].t(), local_train_L[:,opt.num_class1:]) + torch.matmul(local_G_buffer_dict[j].t(), local_train_L[:,opt.num_class1:])) +
                        opt.hash2 * torch.matmul(local_Y1_dict[j].t(), S12))



                for i in range(3):
                    F = local_F_buffer_dict[j].t()
                    G = local_G_buffer_dict[j].t()
                    Y1 = local_Y1_dict[j].t()   
                    Y2 = local_Y2_dict[j].t()
                    for k in range(opt.bit):

                        sel_ind = torch.isin(torch.arange(opt.bit), torch.tensor([k]), invert=True)
                        Y1_ = Y1[sel_ind, :]
                        y1k = Y1[k, :].t()
                        Y2_ = Y2[sel_ind, :]
                        y2k = Y2[k, :].t()
                        Fk = F[k, :].t( )
                        F_ = F[sel_ind, :]
                        Gk = G[k, :].t()
                        G_ = G[sel_ind, :]

                        y1 = torch.sign(Q1[k, :].t() - opt.hash2 * Y1_.t().matmul(Y2_.matmul(y2k))
                                    - opt.hash_l1* (Y1_.t().matmul(F_.matmul(Fk)) + Y1_.t().matmul(G_.matmul(Gk))))

                        local_Y1_dict[j][:, k] = y1
                    if torch.norm(local_Y1_dict[j]-Y1.t()) < 1e-6 * torch.norm(Y1.t()):
                        break
                for i in range(3):
                    F = local_F_buffer_dict[j].t()
                    G = local_G_buffer_dict[j].t()
                    Y1 = local_Y1_dict[j].t()
                    Y2 = local_Y2_dict[j].t()
                    for k in range(opt.bit):
                        sel_ind = torch.isin(torch.arange(opt.bit), torch.tensor([k]), invert=True)
                        Y1_ = Y1[sel_ind, :]
                        y1k = Y1[k, :].t()
                        Y2_ = Y2[sel_ind, :]
                        y2k = Y2[k, :].t()
                        Fk = F[k, :].t()
                        F_ = F[sel_ind, :]
                        Gk = G[k, :].t()
                        G_ = G[sel_ind, :]
                        q1 = Q1[k, :].t()
                        q2 = Q2[k, :].t()

                        y2 = torch.sign(Q2[k, :].t() - opt.hash2 * Y2_.t().matmul(Y1_.matmul(y1k))
                                    - (1-opt.hash_l1) * (Y2_.t().matmul(F_.matmul(Fk)) + Y2_.t().matmul(G_.matmul(Gk))))
                        local_Y2_dict[j][:, k] = y2
                    if torch.norm(local_Y2_dict[j] - Y2.t()) < 1e-6 * torch.norm(Y2.t()):
                        break


                # calculate total loss 
                loss = calc_loss_SHDCH(local_B, local_F_buffer, local_G_buffer, local_Y1_dict[j],local_Y2_dict[j],Variable(S12),local_train_L[:,0:opt.num_class1], local_train_L[:,opt.num_class1:],opt)
                print('...round: %3d, client: %3d, epoch: %3d, loss: %3.3f' % (r + 1, j + 1, epoch + 1, loss.data))
                logger.info('...round: %3d, client: %3d, epoch: %3d, loss: %3.3f' % (r + 1, j + 1, epoch + 1, loss.data))

            w_img = models_img[j].state_dict()
            w_img_glob.append(w_img)
            w_txt = models_txt[j].state_dict()
            w_txt_glob.append(w_txt)
            w_transfer = models_transfer[j].state_dict()
            w_transfer_glob.append(w_transfer)

    

        modal_w = weight_fed(opt,agg_number,local_F_buffer_transfer_dict,local_G_buffer_transfer_dict)
        modal_img = weight_fed_encoder(opt,agg_number,img_cls_loss_dict)
        modal_txt = weight_fed_encoder(opt,agg_number,txt_cls_loss_dict)

        # calculate updated global weights
        w_img_new = weighted_agg_fed(w_img_glob, modal_img)
        w_txt_new = weighted_agg_fed(w_txt_glob, modal_txt)
        w_transfer_new = weighted_agg_fed(w_transfer_glob, modal_w)

        server_img_model.load_state_dict(w_img_new)
        server_txt_model.load_state_dict(w_txt_new)
        server_transfer_model.load_state_dict(w_transfer_new)

        # send updated global weights to every client
        for i in range(opt.num_users):
            models_img[i] = copy.deepcopy(server_img_model)
            models_txt[i] = copy.deepcopy(server_txt_model)
            models_transfer[i] = copy.deepcopy(server_transfer_model)

        
        mapi2t, mapt2i = valid(server_img_model, server_txt_model, server_transfer_model,query_x, retrieval_x, query_y, retrieval_y,
                                       query_L[:,opt.num_class1:], retrieval_L[:,opt.num_class1:], opt)
        rounds.append(r+1)
        i_t.append((mapi2t*100).cpu().numpy())
        t_i.append((mapt2i*100).cpu().numpy())
        print('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (r + 1, mapi2t, mapt2i))
        logger.info('...round: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (r + 1, mapi2t, mapt2i))
        plt.plot(rounds, i_t, label='mapi2t', linestyle='-', color='b')
        plt.plot(rounds, t_i, label='mapt2i', linestyle='-', color='r')

        plt.xlabel('round')
        plt.ylabel('map')
        plt.legend()
        

        plt.xlim(0, opt.rounds)
        plt.ylim(0, 100)
        
        if not os.path.exists('./plt/'):
            os.makedirs('./plt/')
        plt.savefig('./plt/'+filename+'.png') 

        plt.clf() 


        max_mapi2t = max(mapi2t,max_mapi2t)
        max_mapt2i = max(mapt2i,max_mapt2i)



        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        logger.info('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))

    print('...training procedure finish')
    logger.info('...training procedure finish')
  


