import h5py
import scipy.io as scio
import os
import numpy as np

def load_data(opt):
    if opt.dataset=='FLICKR-25K':
        file = h5py.File(os.path.join(opt.data_path,'FLICKR-25K.mat'))
        images = file['IAll'][:].transpose(0,1,3,2) #(20015,3,224,224)
        labels = file['LAll'][:].transpose(1,0)  #(20015, 24)
        tags = file['YAll'][:].transpose(1,0) #(20015, 1386)
        file.close()

    elif opt.dataset=='NUS-WIDE':
        imgfile = h5py.File(os.path.join(opt.data_path,'nus-wide-tc21-iall.mat'))
        images = imgfile['IAll'][:] #(195834,3,224,224)
        imgfile.close()
        labelfile=scio.loadmat(os.path.join(opt.data_path,'nus-wide-tc21-lall.mat'))
        labels = labelfile['LAll'][:]  #(195834, 21)
        tagfile=scio.loadmat(os.path.join(opt.data_path,'nus-wide-tc21-yall.mat'))
        tags = tagfile['YAll'][:] #(195834, 1000)

    elif opt.dataset=='FashionVC' or 'Ssense':
        imgfile=scio.loadmat(os.path.join(opt.data_path,'image.mat'))
        images=imgfile['Image'].transpose(0,3,1,2)
        labelfile = scio.loadmat(os.path.join(opt.data_path,'label.mat'))
        labels = labelfile['Label'][:]
        tagfile=scio.loadmat(os.path.join(opt.data_path,'tag.mat'))
        tags=tagfile['Tag'][:]

    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)

def split_data(images, tags, labels, opt):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L

def iid(opt, train_label):
    """
    Sample I.I.D. client data from dataset
    """
    num_items = int(len(train_label) / opt.num_users)
    dict_users, all_idxs = {}, [i for i in range(len(train_label))]
    for i in range(opt.num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

 
def hetero_dir_partition(opt, train_label, dir_alpha, min_require_size=None):   
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if min_require_size is None:
        min_require_size = train_label.shape[1]
    targets=np.argmax(train_label, axis=1)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = train_label.shape[0]

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(opt.num_users)]
        # for each class in the dataset
        for k in range(train_label.shape[1]):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, opt.num_users))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / opt.num_users) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(opt.num_users):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    return client_dict

