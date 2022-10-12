import numpy as np
import torch
import random 
import os

SEED =0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_reconstr_score =None
        
        self.early_stop = False
        self.acc_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, acc_score, reconst_score, model):

        score = acc_score

        if self.best_score is None:
            self.best_score = score
            self.best_reconstr_score =reconst_score
            self.save_checkpoint(acc_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_reconstr_score =reconst_score
            self.save_checkpoint(acc_score, model)
            self.counter = 0

    def save_checkpoint(self, acc_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Accuracy score inscreased ({self.acc_score_max:.6f} --> {acc_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'models/checkpoint_{acc_score:.6f}.pt')
        self.acc_score_max = acc_score