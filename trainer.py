import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from early_stopping import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold,GroupShuffleSplit
from torch.utils.data import DataLoader, ConcatDataset
from models import EEGAutoencoder
from torch import nn
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


class Trainer():

    def __init__(self, num_epochs, experiment_name,
                 model, rloss, closs, optimizer, scheduler,device ='cuda:0',patience=15,seed=0):
        self.num_epochs =num_epochs
        self.writer =SummaryWriter(experiment_name)
        self.model =model
        self.criterion =rloss
        self.cr_loss =closs
        self.optimizer =optimizer
        self.scheduler =scheduler
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.device =device
        self.seed =seed
        self.set_seed()
        
    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
    
    def plot_batch_and_outputs(self, batch, outputs, show_type='img'):
        Y_ticks = np.linspace(0, 128, num=98)
        X_ticks = np.linspace(0, 45, num=24)
        random_channel = np.random.randint(batch.shape[1])
        levels = 45

        if show_type == 'img':
            print('True: ')
            spectrum = plt.imshow(batch[0, 0, random_channel, :, :].detach().cpu().numpy(), cmap='jet')
            plt.show()
            print('Predicted: ')
            spectrum = plt.imshow(outputs[0, 0, random_channel, :, :].detach().cpu().numpy(), cmap='jet')
            plt.show()
        else:
            spectrum = plt.contourf(Y_ticks, X_ticks, batch[0, 0, random_channel, :, :].detach().cpu().numpy(), levels,
                                    cmap='jet')
            plt.show()
            print('Predicted: ')
            spectrum = plt.contourf(Y_ticks, X_ticks, outputs[0, 0, random_channel, :, :].detach().cpu().numpy(),
                                    levels, cmap='jet')
            plt.show()


    def __iner_loop__(self, loader, is_Train =True):
        loss,reconstr_loss,correct,total,labels_all,softmax_preds = 0,0,0,0, [],[]
        for data in loader:
            img, label = data
            img = img.float().to(self.device)
            label = label.long().to(self.device)
            # ===================forward=====================
            codes, output, preds = self.model(img)
            soft_preds = F.softmax(preds.data)
            _, predicted = torch.max(soft_preds, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            l2_loss = self.criterion(output, img)
            cr_loss1 = self.cr_loss(preds, label)
            loss = l2_loss + cr_loss1
            # ===================backward====================
            if is_Train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()


            loss += loss.data
            reconstr_loss +=l2_loss.data
            labels_all.append(label.cpu().detach().numpy())
            softmax_preds.append(soft_preds.cpu().detach().numpy()[:, 1])
        labels_all = np.concatenate(labels_all)
        softmax_preds = np.concatenate(softmax_preds)
        return loss,reconstr_loss,correct,total,labels_all,softmax_preds,img,output


    def train(self, train_loader,val_loader,vebrose=False):
        for epoch in range(self.num_epochs):
            train_loss,train_reconstr_loss,train_correct,train_total,_,_,_,_ =self.__iner_loop__(train_loader, is_Train=True)
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_loss,val_reconstr_loss, val_correct, val_total, labels_all, softmax_preds,img,output = self.__iner_loop__(val_loader, is_Train=False)
                    self.writer.add_scalar('training total loss',
                                      train_loss / train_total,
                                      epoch)
                    self.writer.add_scalar('training reconstraction loss',
                                      train_reconstr_loss / val_total,
                                     epoch)
                    
                    self.writer.add_scalar('validation total loss',
                                      val_loss / val_total,
                                      epoch)
                    self.writer.add_scalar('validation reconstraction loss',
                                      val_reconstr_loss / val_total,
                                      epoch)
                    self.writer.add_scalar('train accuracy',
                                      train_correct / train_total,
                                      epoch)
                    self.writer.add_scalar('val accuracy',
                                      val_correct / val_total,
                                      epoch)
                    self.writer.add_scalar('val roc auc',
                                      roc_auc_score(labels_all, softmax_preds),
                                      epoch)
                    if vebrose:
                        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch + 1, self.num_epochs, train_loss / train_total))
                        print('epoch [{}/{}], val loss:{:.4f}'.format(epoch + 1, self.num_epochs, val_loss / val_total))
                        print('epoch [{}/{}], train accuracy:{:.4f}'.format(epoch + 1, self.num_epochs,
                                                                            train_correct / train_total))
                        print('epoch [{}/{}], val accuracy:{:.4f}'.format(epoch + 1, self.num_epochs, val_correct / val_total))
                        print('epoch [{}/{}], val roc auc:{:.4f}'.format(epoch + 1, self.num_epochs,
                                                                         roc_auc_score(labels_all, softmax_preds)))
                        self.plot_batch_and_outputs(img, output, show_type='c')
                    self.early_stopping(val_correct / val_total,val_reconstr_loss/val_total, self.model)

                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
        return  self.early_stopping.best_score, self.early_stopping.best_reconstr_score