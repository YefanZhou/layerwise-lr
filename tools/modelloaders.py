import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from tools.models import *
from tools.pruners import prune_weights_reparam

def model_and_opt_loader(model_string, target_spar, retrain_budget, DEVICE):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = target_spar
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.SGD, lr=0.1),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD, lr=0.1),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)
        amount = target_spar
        batch_size = 100
        pre_steps = 50000
        post_steps = int(pre_steps * retrain_budget)
        opt_pre = {
            "optimizer": partial(optim.SGD, lr=0.1, weight_decay=5e-4),
            "steps": pre_steps,
            "scheduler": partial(optim.lr_scheduler.MultiStepLR, milestones=[int(0.5 * pre_steps), int(0.75 * pre_steps)], gamma=0.1),
            "warmup": False
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.1, weight_decay=5e-4),
            "steps": post_steps,
            "scheduler": partial(optim.lr_scheduler.LinearLR, start_factor=1, end_factor=0, total_iters= int(0.9 * post_steps)),
            "warmup":True
        }
    elif model_string == 'densenet':
        # model = DenseNet121().to(DEVICE)
        # amount = target_spar
        # batch_size = 100
        # opt_pre = {
        #     "optimizer": partial(optim.AdamW,lr=0.0003),
        #     "steps": 80000,
        #     "scheduler": None
        # }
        # opt_post = {
        #     "optimizer": partial(optim.AdamW,lr=0.0003),
        #     "steps": 60000,
        #     "scheduler": None
        # }
        model = DenseNet121().to(DEVICE)
        amount = target_spar
        batch_size = 100
        pre_steps = 80000
        post_steps = int(pre_steps * retrain_budget)
        opt_pre = {
            "optimizer": partial(optim.SGD, lr=0.1, weight_decay=5e-4),
            "steps": pre_steps,
            "scheduler": partial(optim.lr_scheduler.MultiStepLR, milestones=[int(0.5 * pre_steps), int(0.75 * pre_steps)], gamma=0.1),
            "warmup": False
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.1, weight_decay=5e-4),
            "steps": post_steps,
            "scheduler": partial(optim.lr_scheduler.LinearLR, start_factor=1, end_factor=0, total_iters= int(0.9 * post_steps)),
            "warmup":True
        }
    elif model_string == 'effnet':
        # model = EfficientNetB0().to(DEVICE)
        # amount = target_spar
        # batch_size = 100
        # opt_pre = {
        #     "optimizer": partial(optim.AdamW,lr=0.0003),
        #     "steps": 50000,
        #     "scheduler": None
        # }
        # opt_post = {
        #     "optimizer": partial(optim.AdamW,lr=0.0003),
        #     "steps": 40000,
        #     "scheduler": None
        # }
        model = EfficientNetB0().to(DEVICE)
        amount = target_spar
        batch_size = 100
        pre_steps = 50000
        post_steps = int(pre_steps * retrain_budget)
        opt_pre = {
            "optimizer": partial(optim.SGD, lr=0.1, weight_decay=5e-4),
            "steps": pre_steps,
            "scheduler": partial(optim.lr_scheduler.MultiStepLR, milestones=[int(0.5 * pre_steps), int(0.75 * pre_steps)], gamma=0.1),
            "warmup": False
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.1, weight_decay=5e-4),
            "steps": post_steps,
            "scheduler": partial(optim.lr_scheduler.LinearLR, start_factor=1, end_factor=0, total_iters= int(0.9 * post_steps)),
            "warmup":True
        }
    else:
        raise ValueError('Unknown model')
    prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post