import torch,argparse,random,os
import numpy as np
from pathlib import Path
from tools import *

""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--cuda', type=int, help='cuda number')
parser.add_argument('--model', type=str, help='network')
parser.add_argument('--pruner', type=str, help='pruning method')
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=20, help='start iteration for pruning')
parser.add_argument('--target_spar', type=float, default=0.2, help='start iteration for pruning')
parser.add_argument('--retrain_budget', type=float, default=0.05, help='budget to retrain')
parser.add_argument('--print_steps', type=int, default=250, help='budget to retrain')

args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

DEVICE = args.cuda

""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model,amount_per_it,batch_size,opt_pre,opt_post = model_and_opt_loader(args.model, args.target_spar, args.retrain_budget, DEVICE)
train_loader, test_loader = dataset_loader(args.model,batch_size=batch_size)
pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader()

""" SET SAVE PATHS """
DICT_PATH = f'/data/yefan0726/checkpoints/retrain_lr/dict/{args.model}/{args.seed}/spar_{args.target_spar}'
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)
BASE_PATH = f'/data/yefan0726/checkpoints/retrain_lr/results/iterate/{args.model}/{args.seed}'
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

""" PRETRAIN (IF NEEDED) """
if args.iter_start == 1:
    filename_string = 'unpruned.pth'
else:
    filename_string = args.pruner+str(args.iter_start-1)+'.pth'
if os.path.exists(os.path.join(DICT_PATH,filename_string)):
    print(f"LOADING PRE-TRAINED MODEL: SEED: {args.seed}, MODEL: {args.model}, ITER: {args.iter_start - 1}")
    state_dict = torch.load(os.path.join(DICT_PATH,filename_string),map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
else:
    if args.iter_start == 1:
        print(f"PRE-TRAINING A MODEL: SEED: {args.seed}, MODEL: {args.model}")
        Path(os.path.join(DICT_PATH, 'unpruned')).mkdir(parents=True, exist_ok=True)
        pretrain_results = trainer(model,opt_pre, train_loader, test_loader, 
                                    int(args.print_steps * 2), log_results=True, save_model=False, 
                                    log_path = os.path.join(DICT_PATH, 'unpruned', 'log.npy'), 
                                    log_dir=os.path.join(DICT_PATH, 'unpruned'))
        
        torch.save(pretrain_results, DICT_PATH+'/unpruned_loss.dtx')
        torch.save(model.state_dict(),os.path.join(DICT_PATH,'unpruned.pth'))
    else:
        raise ValueError('No (iteratively pruned/trained) model found!')

""" PRUNE AND RETRAIN """
results_to_save = []
for it in range(args.iter_start,args.iter_end+1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(model,amount_per_it)   
    Path(os.path.join(DICT_PATH, f'pruned_{it}')).mkdir(parents=True, exist_ok=True) 
    torch.save({'net':model.state_dict()}, 
                os.path.join(os.path.join(DICT_PATH, 
                f'pruned_{it}', f'pruned.pth')))

    result_log = trainer(model,opt_post,train_loader,test_loader, 
                         args.print_steps, log_results=True, save_model=True, 
                         log_path = os.path.join(DICT_PATH, f'pruned_{it}', f'log.npy'), 
                         log_dir=os.path.join(DICT_PATH, f'pruned_{it}'))
    
    result_log.append(get_model_sparsity(model))
    results_to_save.append(result_log)
    np.save(DICT_PATH+f'/{args.pruner}.npy', results_to_save)