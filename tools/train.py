import os
import torch
import numpy as np
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
import tqdm
def trainer_loader():
    return train

def initialize_weight(model,loader):
    batch = next(iter(loader))
    device = next(model.parameters()).device
    with torch.no_grad():
        model(batch[0].to(device))

def train(model,optpack,train_loader,test_loader,print_steps=-1, log_results=False, save_model=False, log_path='log.txt', log_dir=''):
    model.train()
    opt = optpack["optimizer"](model.parameters())
    if optpack["scheduler"] is not None:
        sched = optpack["scheduler"](opt)
        if optpack['warmup']:
            sched = GradualWarmupScheduler(opt, multiplier=1, total_epoch=int(0.1 * sched.total_iters), after_scheduler=sched)
    else:
        sched = None
    num_steps = optpack["steps"]
    device = next(model.parameters()).device
    lr_log = []
    results_log = []
    results_log.append(["training_step", "lr", "test_acc", "test_loss", "train_acc", "train_loss"])
    training_step = 0
    pbar = tqdm.tqdm(total=num_steps, desc='Training....')
    if sched is not None:
        while True:
            for i,(x,y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat,y)
                loss.backward()
                opt.step()
                sched.step()
                pbar.update(1)
                lr_log.append(opt.param_groups[0]['lr'])
                np.save(os.path.join(log_dir, f'lr_log.npy'), np.array(lr_log))
                if print_steps != -1 and training_step%print_steps == 0:
                    train_acc,train_loss    = test(model,train_loader)
                    test_acc,test_loss      = test(model,test_loader)
                    train_loss = train_loss.item()
                    test_loss = test_loss.item()
                    lr = opt.param_groups[0]['lr']
                    print(f'\n Steps: {training_step}/{num_steps} \t LR: {lr:.5f} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f}')
                    if log_results:
                        results_log.append([training_step, lr, test_acc, test_loss, train_acc, train_loss])
                        np.save(log_path, results_log)
                    if save_model:
                        torch.save({'net':model.state_dict(),
                                    'training_step':training_step,
                                    'lr':lr,
                                    'test_acc':test_acc,
                                    'test_loss':test_loss,
                                    'train_acc':train_acc,
                                    'train_loss':train_loss
                                    }, 
                                    os.path.join(log_dir, f'model_iter_{training_step}.pth'))

                if training_step >= num_steps:
                    break
            if training_step >= num_steps:
                break
    else:
        while True:
            for i,(x,y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)
        
                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat,y)
                loss.backward()
                opt.step()
                
                if print_steps != -1 and training_step%print_steps == 0:
                    train_acc,train_loss    = test(model,train_loader)
                    test_acc,test_loss      = test(model,test_loader)
                    print(f'Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f}', end='\r')
                    if log_results:
                        results_log.append([test_acc,test_loss,train_acc,train_loss])
                        np.savetxt(log_path,results_log)

                if training_step >= num_steps:
                    break
            if training_step >= num_steps:
                break
    train_acc,train_loss    = test(model,train_loader)
    test_acc,test_loss      = test(model,test_loader)
    train_loss = train_loss.item()
    test_loss = test_loss.item()
    print(f'Train acc: {train_acc:.2f}\t Test acc: {test_acc:.2f}')
    return [test_acc,test_loss,train_acc,train_loss]

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