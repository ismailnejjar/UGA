import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
import pickle
torch.set_num_threads(1)
import math
from read_data import ImageList_r as ImageList
import torchvision
from utils import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='n', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='s', metavar='T',
                    help='target dataset')
parser.add_argument('--uncertainty_alignment', type=str, default='feature',
                        help='where to align the uncertainty at the output or at the feature level')
parser.add_argument('--cmixup', type=int, default=1,
                        help='cmixup True (1) or False (0)')
parser.add_argument('--lr', type=float, default= 2e-5,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--batch', type=int, default=36,
                        help='batch size')
parser.add_argument('--num_iter', type = int, default=20000,
                        help='number of iteration')
parser.add_argument('--test_interval', type = int, default=500,
                        help='number of iteration before testing')
parser.add_argument('--dropout_rate', type = float, default=0.1,
                        help='drop out rate in resnet')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}

# set dataset
batch_size = {"source_train": args.batch, "target_train": args.batch,"target_test": args.batch, "source_test": args.batch}
c="color.txt"
n="noisy.txt"
s="scream.txt"

c_t="color_test.txt"
n_t="noisy_test.txt"
s_t="scream_test.txt"

if args.src =='c':
    source_path = c
elif args.src =='n':
    source_path = n
elif args.src =='s':
    source_path = s

if args.src =='c':
    source_path_t = c_t
elif args.src =='n':
    source_path_t = n_t
elif args.src =='s':
    source_path_t = s_t

if args.tgt =='c':
    target_path = c
elif args.tgt =='n':
    target_path = n
elif args.tgt =='s':
    target_path = s
    
if args.tgt =='c':
    target_path_t = c_t
elif args.tgt =='n':
    target_path_t = n_t
elif args.tgt =='s':
    target_path_t = s_t


dsets = {"source_train": ImageList(open(source_path).readlines(), transform=data_transforms["train"]),
         "target_train": ImageList(open(target_path).readlines(),transform=data_transforms["train"]),
         "target_test": ImageList(open(target_path_t).readlines(),transform=data_transforms["test"]),
         "source_test": ImageList(open(source_path_t).readlines(),transform=data_transforms["test"])}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],shuffle=True, num_workers=8)
                for x in ['source_train', 'target_train','target_test','source_test']}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Regression_test(loader, model,mode='test'):
    MSE = [0, 0, 0,0]
    MAE = [0, 0, 0,0]
    number = 0
    with torch.no_grad():
        for (imgs, labels) in loader[mode]:
            imgs = imgs.to(device)
            labels_source = labels.to(device)
            label1 = labels_source[:, 0]
            label3 = labels_source[:, 2]
            label4 = labels_source[:, 3]
            label1 = label1.unsqueeze(1)
            label3 = label3.unsqueeze(1)
            label4 = label4.unsqueeze(1)
            labels_source = torch.cat((label1,label3,label4),dim=1)
            labels = labels_source.float()

            pred,ob,alpha,beta = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred[:, 2], labels[:, 2])
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred[:, 2], labels[:, 2])
            MSE[3] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[3] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
    for j in range(4):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1},{2}\n".format(MSE[0],MSE[1],MSE[2]))
    print("\tMAE : {0},{1},{2}\n".format(MAE[0], MAE[1], MAE[2]))
    print("\tMSEall : {0}\n".format(MSE[3]))
    print("\tMAEall : {0}\n".format(MAE[3]))
    return MAE[3]

def get_mixup_sample_rate(data_packet, device='cuda', use_kde = False):
    
    mix_idx = []
    _, y_list = data_packet['x_train'], data_packet['y_train'] 
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        data_i = data_list[i]
        data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D

        ######### get kde sample rate ##########
        kd = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data_i)  # should be 2D
        each_rate = np.exp(kd.score_samples(data_list))
        each_rate /= np.sum(each_rate)  # norm
        
        ####### visualization: observe relative rate distribution shot #######
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]
    
    return mix_idx


def get_batch_kde_mixup_idx(Batch_X, Batch_Y, device):
    assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(Batch_packet, device, use_kde=True) # batch -> kde

    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx]) 
            for sel_idx in np.arange(Batch_X.shape[0]//2)]
    return idx2

def get_batch_kde_mixup_batch(Batch_X1, Batch_X2, Batch_Y1, Batch_Y2, device):
    Batch_X = torch.cat([Batch_X1, Batch_X2], dim = 0)
    Batch_Y = torch.cat([Batch_Y1, Batch_Y2], dim = 0)

    idx2 = get_batch_kde_mixup_idx(Batch_X,Batch_Y,device)

    New_Batch_X2 = Batch_X[idx2]
    New_Batch_Y2 = Batch_Y[idx2]
    return New_Batch_X2, New_Batch_Y2,idx2

class Generator(nn.Module):
    def __init__(self,dropout=0.1):
        super(Generator,self).__init__()
        self.model_fc = model.Resnet18Fc(dropout = dropout)
        
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(Generator, self).train(mode)
        for m in self.model_fc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False

    def forward(self,x):
        feature = self.model_fc(x)
        return feature

    
class Regressor(nn.Module):
    def __init__(self,dimention = 512):
        super(Regressor,self).__init__()
        self.uncertainty = nn.Sequential()
        self.uncertainty.add_module('d_fc1', nn.Linear(dimention, 4*3))
        self.uncertainty.d_fc1.bias.data.fill_(0.0)

        self.evidence = nn.Softplus()
    
        self.sigm = nn.Sigmoid()
        
        
    def forward(self,x):
        pred = self.uncertainty((x))
        pred = pred.view(x.shape[0], -1, 4)

        mu,logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]

        v = self.evidence(logv)+1e-5
        alpha = self.evidence(logalpha) + 1 +1e-5
        beta = self.evidence(logbeta) +1e-5
        return self.sigm(mu), v, alpha, beta

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        x = self.modelA(x)
        mu, v, alpha, beta = self.modelB(x)
        return mu, v, alpha, beta
    
    
G = Generator(dropout = args.dropout_rate)
R = Regressor(dimention = 512)

G = G.to(device)
R = R.to(device)

mmd_loss = MMD_loss()

opt_g = optim.Adam(G.parameters(), lr= args.lr)
opt_r = optim.Adam(R.parameters(), lr= args.lr)

param_lr_g = []
param_lr_r = []

for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
    
for param_group in opt_r.param_groups:
    param_lr_r.append(param_group["lr"])
    
def reset_grad():
    opt_g.zero_grad()
    opt_r.zero_grad()
    
def opti_step():
    opt_g.step()
    opt_r.step()
    
def train_models():
    G.train()
    R.train()
    
def eval_models():
    G.eval()
    R.eval()
    
len_source = len(dset_loaders["source_train"]) - 1
len_target = len(dset_loaders["target_train"]) - 1

iter_source = iter(dset_loaders["source_train"])
iter_target = iter(dset_loaders["target_train"])

train_uncertainty_loss = train_source_loss = train_total_loss = 0.0
print(args)

for iter_num in range(1, args.num_iter + 1):
    p = (iter_num)/(20000)
    alpha_ = 2. / (1. + np.exp(-10 * p)) - 1
    opt_g = inv_lr_scheduler(param_lr_g, opt_g, iter_num, init_lr=1, gamma=args.gamma, power=0.75,weight_decay=0.0005)
    opt_r = inv_lr_scheduler(param_lr_r, opt_r, iter_num, init_lr=1, gamma=args.gamma, power=0.75,weight_decay=0.0005)

    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["source_train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["target_train"])
        
    data_source = iter_source.next()
    data_target = iter_target.next()

    inputs_target, labels_target = data_target    
    inputs_source, labels_source = data_source

    labels1 = labels_source[:, 0]
    labels3 = labels_source[:, 2]
    labels4 = labels_source[:, 3]
    labels1 = labels1.unsqueeze(1)
    labels3 = labels3.unsqueeze(1)
    labels4 = labels4.unsqueeze(1)
    labels_source = torch.cat((labels1,labels3,labels4),dim=1)
    labels_source = labels_source.float()
    
    train_models()
    reset_grad()
    
    feat_s = G(inputs_source.float().to(device))
    gamma_s,v_s,alpha_s,beta_s = R(feat_s)
    loss_s = EvidentialRegression(labels_source.to(device),gamma_s, v_s, alpha_s, beta_s, device, coef=1)
    
    feat_t = G(inputs_target.to(device))
    gamma_t, v_t, alpha_t, beta_t = R(feat_t)
    
    uncertainty_s = (beta_s)/(v_s*(alpha_s-1))
    uncertainty_t = (beta_t)/(v_t*(alpha_t-1))
    
    evi_s = torch.cat((v_s,alpha_s,beta_s),dim=1)
    evi_t = torch.cat((v_t, alpha_t, beta_t),dim=1)

    
    if(args.cmixup):
        X1 = inputs_target
        Y1 = gamma_t.detach().cpu()
        X2 = inputs_source
        Y2 = labels_source
        X2, Y2, idx2= get_batch_kde_mixup_batch(X1,X2,Y1,Y2,device)

        batch_size = X1.shape[0]

        X1 = X1.to(device)
        X2 = X2.to(device)
        Y1 = Y1.to(device)
        Y2 = Y2.to(device)

        lambd = np.random.beta(2,2)
        #mixup_Y = (Y1 * lambd + Y2 * (1 - lambd))
        #mixup_X = (X1 * lambd + X2 * (1 - lambd))

        #feat_mix = G(mixup_X.float())
        feat_mix = feat_t*lambd + torch.cat((feat_t,feat_s))[idx2]*(1-lambd)
        gamma_mix,v_mix,alpha_mix,beta_mix = R(feat_mix)
        uncertainty_mix = (beta_mix)/(v_mix*(alpha_mix-1))
        evi_mix = torch.cat((v_mix,alpha_mix,beta_mix),dim=1)
        if(args.uncertainty_alignment == 'output'):
            uncertainty_loss = mmd_loss(evi_s,evi_t) + mmd_loss(evi_s,evi_mix)
        else:
            uncertainty_loss = mmd_loss(feat_s,feat_t) + mmd_loss(feat_s,feat_mix)
    else:
        if(args.uncertainty_alignment == 'output'):
            uncertainty_loss = mmd_loss(evi_s,evi_t)
        else:
            uncertainty_loss = mmd_loss(feat_s,feat_t)

    
    loss =  loss_s + alpha_*(uncertainty_loss)

    loss.backward()
    opti_step()
    
    train_total_loss += loss.detach().item()
    train_uncertainty_loss += uncertainty_loss.detach().item()
    train_source_loss += loss_s.detach().item()

    if (iter_num % args.test_interval) == 0:
        
        print('Iter number : ',iter_num ,', Total loss : ' , train_total_loss/args.test_interval ,
              ', Source loss : ' , train_source_loss/args.test_interval ,', Uncertainty loss : ', train_uncertainty_loss/args.test_interval)
        
        eval_models()
        Model_R = MyEnsemble(G, R)
        mae = Regression_test(dset_loaders, Model_R,mode = 'target_test')
        train_uncertainty_loss = train_source_loss = train_total_loss = 0.0