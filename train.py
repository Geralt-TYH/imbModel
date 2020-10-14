from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from layers import Encoder
from layers import InnerProductDecoder as Decoder

from utils import load_data, accuracy
from models import GCIM
from sklearn.cluster import KMeans

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)



# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
decoder = Decoder()
model = GCIM(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#if args.cuda:
#    model.cuda()
#    features = features.cuda()
#    adj = adj.cuda()
#    labels = labels.cuda()
#    idx_train = idx_train.cuda()
#    idx_val = idx_val.cuda()
#    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj,labels)
    with torch.no_grad():
        z = model.encoder(features,adj)
    recon_loss = model.recon_loss(adj,z)
    kl_clusting_loss = model.kl_clusting_loss(features,adj,labels)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train += recon_loss
    loss_train += kl_clusting_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj,labels)

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
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
