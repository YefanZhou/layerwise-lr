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

set_seed(args.seed)
criterion = nn.CrossEntropyLoss()
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

cifar_nm    = T.Normalize(mean,std)
tfm_test = T.Compose([T.ToTensor(),cifar_nm])
# Training set init
trainset = torchvision.datasets.CIFAR100(root="/data/yefan0726/data/cv/cifar100", train=True, download=True, transform=tfm_test)
subset_list = list(range(0, args.train_size))
trainset_subset = torch.utils.data.Subset(trainset, subset_list)
trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=128, shuffle=False, num_workers=4)

# Test set init
testset = torchvision.datasets.CIFAR100(root="/data/yefan0726/data/cv/cifar100", train=False, download=True, transform=tfm_test)
subset_list = list(range(0, args.test_size))
testset_subset = torch.utils.data.Subset(testset, subset_list)
testloader = torch.utils.data.DataLoader(testset_subset, batch_size=100, shuffle=False, num_workers=4)



# evaluate
raw_test_acc, raw_test_loss = test(net, torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4))
print(f"raw_test_acc: {raw_test_acc:.3f},  raw_test_loss: {raw_test_loss:.3f}")

train_activations = collections.defaultdict(list)
train_targets = []
train_hook_handles= []
def save_train_activation(name, mod, inp, out):
    train_activations[name].append(out.cpu())

test_activations = collections.defaultdict(list)
test_targets = []
def save_test_activation(name, mod, inp, out):
    test_activations[name].append(out.cpu())

# forward to get train activations
for name, m in net.named_modules():
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        # partial to assign the layer name to each hook
        handle = m.register_forward_hook(partial(save_train_activation, name))   
        train_hook_handles.append(handle)  
for inputs, target in trainloader:
    inputs = inputs.to(device)
    out = net(inputs)
    train_targets.append(target)
# remove train hook
for handle in train_hook_handles:
    handle.remove()
    
# forward to get test activations
for name, m in net.named_modules():
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        # partial to assign the layer name to each hook
        handle = m.register_forward_hook(partial(save_test_activation, name))

for inputs, target in testloader:
    inputs = inputs.to(device)
    out = net(inputs)
    test_targets.append(target)

train_activations = {name: torch.cat(outputs, 0) for name, outputs in train_activations.items()}
train_targets = torch.cat(train_targets, 0)

test_activations = {name: torch.cat(outputs, 0) for name, outputs in test_activations.items()}
test_targets = torch.cat(test_targets, 0)
#name == 'resnet18':
#        assert num_classes > 0
#        return resnet18(num_classes=num_classes, norm_layer=bn_layer)

train_acc_lst = []
test_acc_lst = []
name_lst = []
dimension_kernel = {32: 4, 16: 4, 8: 2, 4:2, 2:1}
target_dim = 2048
for name in train_activations:
    sc_X = StandardScaler()
    if len(train_activations[name].shape) > 2 and args.downsample:
        X_Train = F.max_pool2d(train_activations[name], kernel_size=dimension_kernel[train_activations[name].shape[-1]]).flatten(start_dim=1)#.detach().numpy()
        X_Train = F.max_pool1d(X_Train, int(X_Train.shape[-1] / target_dim))
        X_Train = X_Train.detach().numpy()
        X_Test = F.max_pool2d(test_activations[name], kernel_size=dimension_kernel[test_activations[name].shape[-1]]).flatten(start_dim=1)#.detach().numpy()
        X_Test = F.max_pool1d(X_Test, int(X_Test.shape[-1] / target_dim))
        X_Test = X_Test.detach().numpy()
    else:
        X_Train = train_activations[name].detach().flatten(start_dim=1).numpy() #
        X_Test = test_activations[name].detach().flatten(start_dim=1).numpy() #
    
    #print(X_Train.shape, X_Test.shape)
    Y_Train = train_targets.numpy()
    Y_Test = test_targets.numpy()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    print(f"Starting to SVM fit the layer: {name}")
    classifier = SVC(kernel = 'linear', random_state = 0) #, max_iter=200
    classifier.fit(X_Train, Y_Train)
    
    Y_Train_pred = classifier.predict(X_Train)
    Y_Test_pred = classifier.predict(X_Test)

    train_acc = accuracy_score(Y_Train_pred, Y_Train)
    test_acc = accuracy_score(Y_Test_pred, Y_Test)

    train_acc_lst.append(train_acc)
    test_acc_lst.append(test_acc)
    name_lst.append(name)
    print(train_acc_lst, test_acc_lst)

train_acc_lst = np.array(train_acc_lst)
test_acc_lst = np.array(test_acc_lst)
name_lst = np.array(name_lst)

print("test_acc_lst", test_acc_lst)

Path(os.path.join(args.save_dir)).mkdir(parents=True, exist_ok=True)
print(f"Saving to {os.path.join(args.save_dir, f'svm_acc_downsample_{args.downsample}_{args.train_size}_{args.test_size}.npy')}")
np.save(os.path.join(args.save_dir, 
        f'svm_acc_downsample_{args.downsample}_{args.train_size}_{args.test_size}.npy'),  
        {'train_acc': train_acc_lst,'test_acc': test_acc_lst, 'name': name_lst, 
         'raw_test_acc':raw_test_acc, 'raw_test_loss':raw_test_loss })
