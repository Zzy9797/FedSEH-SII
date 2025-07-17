import warnings
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument('--dataset', type=str, default='Ssense',choices=['FLICKR-25K', 'NUS-WIDE','FashionVC','Ssense'],
                        help="name of dataset")
    parser.add_argument('--data_path', type=str,
                        help="path of dataset")
    parser.add_argument('--backbone',type=str, choices=['DCMH','SHDCH'],
                        help='backbone method')
    parser.add_argument('--pretrain_model_path', type=str, default='./DCMH_imagenet-vgg-f.mat',
                        help="directory of pretrain_model")
    parser.add_argument('--database_size', type=int,
                        help="database_size")
    parser.add_argument('--query_size', type=int,
                        help="query_size")
    parser.add_argument('--training_size', type=int,
                        help="training_size")
    parser.add_argument('--num_class1', type=int,
                        help="num_class1")
    parser.add_argument('--num_class2', type=int,
                        help="num_class2")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch_size"),
   

    # federated arguments
    parser.add_argument('--rounds', type=int, default=150,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")

    # hyper-parameters
    parser.add_argument('--max_epoch', type=int,
                        help="the number of local episodes: E")
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--lr2', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float,
                        help='gamma')
    parser.add_argument('--activation',type=str, default='ReLU',choices=['ReLU','LeakyReLU','Sigmoid','ELU'],
                        help='activation method')
    parser.add_argument('--wfe', type=float, default=0.1,
                        help='wfe') 
    parser.add_argument('--T', type=float, default=0.09,
                        help='temperature') 
    parser.add_argument('--T2', type=float, default=0.1,
                        help='temperature2') 
    parser.add_argument('--e', type=float, default=0.5,
                        help='e')          
    parser.add_argument('--alphard', type=float, default=1,
                        help='alphard')
    parser.add_argument('--hash_l1', type=float, default=0.3,
                        help='hash_l1')
    parser.add_argument('--hash2', type=float, default=100,
                        help='hash2')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='mu')     
    parser.add_argument('--bit', type=int, default=32,
                        help='final binary code length')
    parser.add_argument('--gpu', default=1,
                        help="Set to a specific GPU ID.")
    parser.add_argument('--beta',type=float,default=0.2)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    opt = parser.parse_args()
    if opt.dataset=='FLICKR-25K':
        opt.data_path='./MIRFlickr-25K'
        opt.backbone='DCMH'
        opt.database_size=18015
        opt.query_size=2000
        opt.training_size=10000
        opt.max_epoch=20
        opt.gamma=1

    elif opt.dataset=='NUS-WIDE':
        opt.data_path='./NUS-WIDE-TC21'
        opt.backbone='DCMH'
        opt.database_size=193734
        opt.query_size=2100
        opt.training_size=10500
        opt.max_epoch=20
        opt.gamma=2.5
        
    elif opt.dataset=='FashionVC':
        opt.data_path='./FashionVC'
        opt.backbone='SHDCH'
        opt.database_size=16862
        opt.query_size=3000
        opt.training_size=16862
        opt.num_class1=8
        opt.num_class2=27
        opt.batch_size=128
        opt.max_epoch=10
        opt.lr=0.0001
        opt.lr2=0.0001
        opt.gamma=7
        opt.mu=1.0
        opt.hash2=150



    elif opt.dataset=='Ssense':
        opt.data_path='./Ssense'
        opt.backbone='SHDCH'
        opt.database_size=13696
        opt.query_size=2000
        opt.training_size=13696
        opt.num_class1=4
        opt.num_class2=28
        opt.batch_size=128
        opt.max_epoch=10
        opt.lr=0.0001
        opt.lr2=0.0001
        opt.gamma=5
        opt.hash2=100
    return opt



