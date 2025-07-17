import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time

class BasicModule(torch.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        pass

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_loss_DCMH(B, F, G, Sim, opt):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + opt.gamma * term2 + opt.alphard * term3
    loss = loss / (opt.batch_size * B.shape[0])
    return loss

def calc_loss_SHDCH(B,F,G,Y1,Y2,Sim12,L1,L2,opt):
    term1 = opt.hash_l1 * torch.sum(torch.pow((opt.bit * L1 - torch.matmul(F, Y1.t())), 2) + torch.pow(
        (opt.bit * L1 - torch.matmul(G, Y1.t())), 2))
    term2 = (1-opt.hash_l1) * torch.sum(torch.pow((opt.bit * L2 - torch.matmul(F, Y2.t())), 2) + torch.pow(
        (opt.bit * L2 - torch.matmul(G, Y2.t())), 2))
    term3 = opt.hash2 * torch.sum(torch.pow((opt.bit * Sim12 - torch.matmul(Y1, Y2.t())), 2))
    term4 = opt.gamma * torch.sum(torch.pow((B - F), 2) + torch.pow((B - G), 2))

    loss = term1 + term2 + term3 + term4
    loss = loss / (opt.batch_size * B.shape[0])
    return loss
    

def calc_neighbor(opt, label1, label2):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    return Sim

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)   #计算当前查询样本标签和每一个样本标签的内积,返回bool，表示标签相似性
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]   #根据哈希码的汉明距离来对内积进行排序
        total = min(k, int(tsum))   
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)   #torch.mean(count / tindex)把相似样本全部检索出来的平均准确率
    map = map / num_query
    return map



