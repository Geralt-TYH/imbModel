
import time
import random

import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from loss import ClassificationLoss
from models import GCIM
from utils import accuracy, load_data

# Training settings
#random seed
seed = 12345
#Number of epochs to train.
epochs = 200
#Initial learning rate.
lr = 0.1
#Weight decay (L2 loss on parameters).
weight_decay = 5e-4
#Number of hidden units.
hidden = 64
#Dropout rate (1 - keep probability).
dropout = 0.5

def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

set_seed(seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../data/cora/")

# Model and optimizer
model = GCIM(ninput=features.shape[1],
             nhid=hidden,
             nout=labels.max().item() + 1,
             dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

#初始化E
E = torch.ones(len(idx_train))
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj,labels)
    z = model.encoder(features,adj)
    recon_loss = model.recon_loss(adj,z)
    kl_clusting_loss = model.kl_clusting_loss(features,adj,labels)
    loss_train = ClassificationLoss(output[idx_train],labels[idx_train],nclass=labels.max().item() + 1,weight=E).forward()
    loss_train += recon_loss
    loss_train += kl_clusting_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj,labels)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
