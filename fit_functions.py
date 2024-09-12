from data_utils.functions import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import pkbar
from models.Bayes_ReLUKAN import Bayes_ReLUKAN#,Bayes_ReLUKAN_EpiOnly
from loss_utils import Gaussian_likelihood, Student_t_likelihood
import os
from matplotlib.colors import LogNorm
import argparse
import random

def set_seeds_(seed=752022):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

class Experiment():
    def __init__(self,generator,width,grid,k,method=None,lkld="Gauss",verbose=False,device='cuda',num_samples=100000,kl_scale=1.0,lr=1e-3):
        self.generator = generator
        self.width = width
        self.grid = grid
        self.k = k
        self.num_samples = num_samples
        self.method = method
        self.x_range = [0,1]
        self.verbose = verbose
        y_ranges = [[-3,4],[-4,4],[-3,6]]

        if "f1" in method:
            self.y_range = y_ranges[0]
        elif "f2" in method:
            self.y_range = y_ranges[1]
        else:
            self.y_range = y_ranges[-1]

        if "noise" in method:
            print("Using aleatoric and epistemic uncertainty.")
            self.bkan = Bayes_ReLUKAN(width,grid,k,aleatoric=True)
        else:
            print("Using only epistemic uncertainty.")
            self.bkan =Bayes_ReLUKAN(width,grid,k,aleatoric=False)

        self.kl_scale = kl_scale
        self.lr = lr
        self.device = device
        self.lkld = lkld
        self.out_path = "data_"+str(self.lkld)

        if self.verbose:
            if not os.path.exists(self.out_path):
                print("Outputs will be placed in ", str(self.out_path))
                os.makedirs(self.out_path)
            else:
                print("Found existing directory at ",self.out_path,". Overwriting outputs.")

        print("Experiment on function: ",str(self.method))

        self.input_size = width[0]

        self.train_xs = np.random.random([self.num_samples,self.input_size,1])
        self.train_ys = self.generator(self.train_xs)

        mu_ = self.train_ys.mean()
 
        self.val_xs = np.random.random([self.num_samples // 10,self.input_size,1])
        self.val_ys = self.generator(self.val_xs)

        self.train_xs = torch.tensor(self.train_xs)
        self.train_ys = torch.tensor(self.train_ys)
        self.val_xs = torch.tensor(self.val_xs)
        self.val_ys = torch.tensor(self.val_ys)

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.opt = torch.optim.Adam(lr=self.lr,params=self.bkan.parameters())
        if self.lkld == "Gauss":
            self.loss_fun = Gaussian_likelihood
        elif self.lkld == "Student":
            self.loss_fun = Student_t_likelihood
        else:
            raise ValueError("Likelihood not implemented. Select: Gauss or Student")


    def trainer(self,num_epochs=1000):
        epoch = 0
        kbar = pkbar.Kbar(target=num_epochs, epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        for epoch in range(num_epochs):
            loss,mse,kl_div = self.train()
            val_loss,val_mse,val_kl_div = self.validate()

            kbar.update(epoch, values=[("loss", loss.item()),("mse",mse.item()),("kl_loss",kl_div.item()),("val_loss", val_loss.item()),("val_mse",val_mse.item()),("val_kl_loss",val_kl_div.item())])


    def train(self,):
        self.bkan.to(self.device)
        self.bkan.train()
        self.opt.zero_grad()

    
        if "noise" in self.method:
            with torch.set_grad_enabled(True):
                pred,log_devs2 = self.bkan(self.train_xs.to(self.device).float())

            if self.lkld == "Student":
                bnn_loss,mse = self.loss_fun(pred, log_devs2,self.bkan.nu, self.train_ys.to(self.device).float())
            else:
                bnn_loss,mse = self.loss_fun(pred, log_devs2, self.train_ys.to(self.device).float())
        
        else:
            with torch.set_grad_enabled(True):
                pred = self.bkan(self.train_xs.to(self.device).float())

            bnn_loss = torch.mean((pred - self.train_ys.to(self.device).float())**2)
            mse = bnn_loss

        kl_div = self.kl_scale * self.bkan.kl_div() / len(self.train_xs)
        loss = bnn_loss + kl_div
        loss.backward()
        self.opt.step()
        self.train_loss.append(loss.item())
        return loss,mse,kl_div

    def validate(self,):
        self.bkan.eval()
        
        if "noise" in self.method:
            with torch.no_grad():
                pred,log_devs2 = self.bkan(self.val_xs.to(self.device).float())
            
            if self.lkld == "Student":
                bnn_loss,mse = self.loss_fun(pred, log_devs2,self.bkan.nu,self.val_ys.to(self.device).float())
            else:
                bnn_loss,mse = self.loss_fun(pred, log_devs2,self.val_ys.to(self.device).float())

        else:
            with torch.no_grad():
                pred = self.bkan(self.val_xs.to(self.device).float())
            bnn_loss = torch.mean((pred - self.val_ys.to(self.device).float())**2)
            mse = bnn_loss

        kl_div = self.kl_scale * self.bkan.kl_div() / len(self.val_xs)
        loss = bnn_loss + kl_div
        self.test_loss.append(loss.item())
        return loss,mse,kl_div    


    def plot_loss(self,name):
        fig = plt.figure(figsize=(8,8))
        #plt.title(f'${name}$ training process')
        plt.xlabel('iterations')
        plt.ylabel('MSE loss')
        plt.plot(self.train_loss, '-', color='black', label='train')
        plt.plot(self.test_loss, '--', color='black', label='test')
        plt.legend()
        plt.savefig(os.path.join(self.out_path,f'process_{name}.pdf'), dpi=600)
        plt.close()

    def plot_results(self, name, mode=1,num_test_samples=20000):
        self.bkan.eval()
        plt.xlabel('$x$',fontsize=24)
        plt.ylabel('$f(x)$',fontsize=24)
        xs = np.array([np.arange(0, num_test_samples) / num_test_samples]).T
        sigma_=0.1
        if 'noise' in name:
            ncol = 2
            ys,clean_ys = self.generator(xs,return_clean=True,sigma=sigma_)
            plt.hist2d(xs.flatten(),ys.flatten(),bins=100,density=True,norm=LogNorm(),range=[self.x_range,self.y_range])
        else:
            ncol=1
            ys = self.generator(xs)
            plt.plot(xs,ys,color='k',label='Truth',lw=2)

        xs = torch.tensor(xs).to('cuda').float()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            if "noise" in self.method:
                avg_pred,epistemic,aleatoric = self.bkan.sample(xs.to(self.device).float(),num_samples=10000)
            else:
                avg_pred,epistemic = self.bkan.sample(xs.to(self.device).float(),num_samples=10000)
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        plt.plot(xs.detach().cpu().numpy(),avg_pred,'--',color='red',label='Avg. Prediction')
        plt.fill_between(xs.detach().cpu().numpy().flatten(),avg_pred - epistemic,avg_pred + epistemic, color='blue',alpha=0.3,label='Epistemic')

        if "noise" in self.method:
            print(" ")
            lower = np.percentile(aleatoric,2.5)
            upper = np.percentile(aleatoric,97.5)
            print("Average Aleatoric: ",np.average(aleatoric)," - 95% Quantiles: ",upper,lower," Interval Length: ",upper - lower)
            if self.lkld == "Student":
                print("DOF: ",self.bkan.nu.exp().detach().cpu())
            print(" ")
            quad = np.sqrt(epistemic ** 2 + aleatoric ** 2)
            plt.fill_between(xs.detach().cpu().numpy().flatten(),avg_pred - quad,avg_pred + quad, color='red',alpha=0.3,label='Quadrature')
            plt.fill_between(xs.detach().cpu().numpy().flatten(),avg_pred - aleatoric,avg_pred + aleatoric, color='k',alpha=0.3,label='Aleatoric')

        plt.legend(loc='best',fontsize=14,ncol=ncol)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(self.y_range)
        plt.xlim(self.x_range)
        plt.savefig(os.path.join(self.out_path,f'effect_{name}.pdf'), bbox_inches="tight")
        plt.close()


train_plan = {
    'f1': (f1, [1, 1], 5, 3),
    'f1_noise': (f1_noise, [1, 1], 5, 3),
    'f2': (f2, [1, 1], 5, 3),
    'f2_noise': (f2_noise, [1, 1], 5, 3),
    'f3': (f3, [1, 1], 5, 3),
    'f3_noise': (f3_noise, [1, 1], 5, 3),
}

if __name__=='__main__':
	# PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Swin Training')
    parser.add_argument('-l', '--lkd', default='Gauss',type=str,
                        help='Choice of likelihood. Gauss or Student.')

    args = parser.parse_args()

    LKLD_ = args.lkd
    set_seeds_()

    for i,f_name in enumerate(train_plan):

        if i == 0:
            verbose = True
        else:
            verbose = False

        train = Experiment(*train_plan[f_name],method=f_name,lkld=LKLD_,verbose=verbose)

        if "noise" in f_name:
            num_epochs = 15000
        else:
            num_epochs = 10000

        train.trainer(num_epochs)
        train.plot_loss(f_name)
        train.plot_results(f_name)
        print(" ")