import torch
import math

PI_ = math.pi

def Gaussian_likelihood(preds,log_devs2,y):
    fn = (y - preds) ** 2
    return 0.5*(torch.exp(-log_devs2) * fn + log_devs2).mean(),fn.mean()



def Student_t_likelihood(preds,log_devs2,nu,y):
    eps = 1e-8
    nu = nu.exp() + eps 
    fn = (y - preds) ** 2
    numerator_ = torch.lgamma((nu + 1.) / 2.)
    denominator_ = torch.lgamma(nu / 2.) + torch.log(torch.sqrt(nu*PI_*log_devs2.exp()))
    lkld = numerator_ - denominator_ -(nu + 1) * 0.5 * torch.log(1 + fn / (nu*log_devs2.exp()))
    return -lkld.mean() , fn.mean()