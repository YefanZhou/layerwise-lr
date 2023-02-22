import os
import torch
from torch.nn import functional as F
import argparse
import random
from torch.utils.data import DataLoader
import torchvision.datasets as dts
import torchvision.transforms as T
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from scipy.spatial.distance import cosine
from functools import partial
import collections
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from resnet import resnet18
import torch.nn as nn
import weightwatcher as ww

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cuda', type=int, help='cuda number')
parser.add_argument('--model', type=str, help='network')
parser.add_argument('--downsample', action='store_true', default=False)
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=200)
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--save_dir', type=str, default="save dir")


args = parser.parse_args()

def set_seed(seed = 1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def test(model,loader):
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    loss    = 0
    total   = 0
    for i,(x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat    = model(x)
            _,pred  = yhat.max(1)
        correct += pred.eq(y).sum().item()
        loss += F.cross_entropy(yhat,y)*len(x)
        total += len(x)
    acc     = correct/total * 100.0
    loss    = loss/total
    
    model.train()
    
    return acc,loss


def compute_layer_rotation(current_model, initial_w):
    '''
    for each layer, computes cosine distance between current weights and initial weights
    initial_w is a list of tuples containing layer name and corresponding initial numpy weights
    '''
    s = []
    for w_idx, w in enumerate(initial_w):
        s.append(cosine( current_model[w_idx].flatten(), w.flatten()))
    return s


use_cuda=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = args.cuda

batch_size=100
if args.model == 'resnet18':
    net = resnet18(num_classes=100, norm_layer=nn.BatchNorm2d).to(DEVICE)

net = net.to(device)
checkpoint = torch.load(args.resume, map_location='cpu')
remove_module_dict = {}
for key in checkpoint['net']:
    remove_module_dict['.'.join(key.split('.')[1:])] = checkpoint['net'][key]
net.load_state_dict(remove_module_dict)


Path(args.save_dir).mkdir(parents=True, exist_ok=True)
watcher = ww.WeightWatcher(model=net)
details_pl = watcher.analyze(vectors=False, randomize=True, mp_fit=True, fit='PL')
details_pl.to_csv(os.path.join(args.save_dir, f'dataframe_pl.csv'))

details_pl_fixfinger = watcher.analyze(vectors=False, randomize=True, mp_fit=True, fit='PL', fix_fingers='xmin_peak')
details_pl_fixfinger.to_csv( os.path.join(args.save_dir, f'dataframe_pl_fixfinger.csv'))

details_etpl = watcher.analyze(vectors=False, randomize=True, mp_fit=True, fit='E_TPL', fix_fingers='xmin_peak') 
details_etpl.to_csv( os.path.join(args.save_dir,  f'dataframe_etpl_epoch.csv'))