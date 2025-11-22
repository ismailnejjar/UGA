import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity

def NIG_NLL(y, gamma, v, alpha, beta, device, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(torch.tensor(np.pi).to(device)/v)  \
        - alpha*torch.log(twoBlambda+1e-5)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda+1e-5)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return nll.mean() if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, device, omega=0.01, reduce=True, kl=False):
    error = torch.abs(y-gamma).detach()
    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, device, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha) #+ 1/beta
        reg = error*evi

    return (reg).mean() if reduce else reg

def EvidentialRegression(y_true, gamma, v, alpha, beta, device ,coef=1.0,reduce=True):
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, device, reduce = reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, device, reduce = reduce)
    return loss_nll + coef * loss_reg

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


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
        kd = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_i)  # should be 2D
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
