import numpy as np
import torch
import torch.nn as nn
from models.torch_mnf.layers.mnf_conv import MNFConv2d
from models.torch_mnf.layers.mnf_linear import MNFLinear


# Developed based on the code from: 
class Bayes_ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True ,imin: float = 0.,imax: float = 1., order: int = 1, device = 'cuda',multi_dim=False):
        super().__init__()
        self.device = device
        if order == 1:
            self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1)*(imax-imin)*(imax-imin))
        else:
            self.g, self.k, self.r = g, k, 2*g / ((k+1)*(imax-imin))

        self.input_size, self.output_size = input_size, output_size
        self.order = order
        self.multi_dim = multi_dim
        phase_low = (imax-imin) * torch.arange(-k, g) / g - (-imin)
        phase_height = phase_low + (k+1) / g * (imax - imin)


        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                     requires_grad=train_ab)

        self.phase_low_log_var = nn.Parameter(-9. + 0.1 * torch.randn_like(self.phase_low),
                                      requires_grad=train_ab)

        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)

        self.phase_height_log_var = nn.Parameter(-9. + 0.1 * torch.randn_like(self.phase_height),
                                         requires_grad=train_ab)

        self.equal_size_conv = MNFConv2d(1,output_size,kernel_size=(int(g+k),int(input_size)))

    def forward(self, x):
        if self.multi_dim:
            x = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        x1 = torch.relu(x - (self.phase_low + self.phase_low_log_var.exp().sqrt() * torch.randn_like(x)))
        x2 = torch.relu((self.phase_height + self.phase_height_log_var.exp().sqrt() * torch.randn_like(x)) - x)
        x = (x1 * x2 * self.r) ** self.order
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)

        if self.multi_dim:
            return x.reshape((len(x), self.output_size))
        else:
            return x.reshape((len(x), self.output_size, 1))

    def kl_div(self) -> float:
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))


class Bayes_ReLUKAN(nn.Module):
    def __init__(self, width, grid, k,order: int = 1, imin: float = 0., imax: float = 1.0,train_ab=True,aleatoric=False,multi_dim=False,aleatoric_width=None,aleatoric_grid=None,aleatoric_k=None):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        self.nu = nn.Parameter(torch.tensor(1.0),requires_grad=train_ab) # log(nu)
        self.aleatoric = aleatoric
        self.multi_dim = multi_dim
        if aleatoric_width is None:
            print("Aleatoric surrogate model using same structure as functional KAN.")
            aleatoric_width = width
            aleatoric_grid = grid
            aleatoric_k = k
            self.diff_call = False
        else:
            print("Aleatoric surrogate model structure will be set with the following parameters:")
            print("Width: ",aleatoric_width)
            print("Grid: ",aleatoric_grid)
            print("k: ",aleatoric_k)
            # Requires a different call, if not we can save a little bit of time.
            self.diff_call = True
        # Default order = 1, imax = 1, imin= 0 ---> Defaults to ReLU KAN implementation
        if order == 1:
            print("Defaulting to first order Bayesian ReLU-KAN.")
        elif order > 1:
            print("Using higher order Bayesian ReLU-KAN.")
        else:
            raise ValueError("Order must be non-negative.")

        # Create surrogate model to estimate aleatoric
        if self.aleatoric:
            self.alea_layers = []

            for i in range(len(width) - 1):
                self.rk_layers.append(Bayes_ReLUKANLayer(width[i], grid, k, width[i+1],imin=imin,imax=imax,order=order,multi_dim=self.multi_dim))   
            self.rk_layers = nn.ModuleList(self.rk_layers)
    
            for i in range(len(aleatoric_width) - 1):
                self.alea_layers.append(Bayes_ReLUKANLayer(aleatoric_width[i], aleatoric_grid, aleatoric_k, aleatoric_width[i+1],imin=imin,imax=imax,order=order,multi_dim=self.multi_dim))
            self.alea_layers = nn.ModuleList(self.alea_layers)

        else:
            for i in range(len(width) - 1):
                self.rk_layers.append(Bayes_ReLUKANLayer(width[i], grid, k, width[i+1],imin=imin,imax=imax,order=order,multi_dim=self.multi_dim))
            self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        if self.aleatoric:
            alea = x.detach().clone()

            if not self.diff_call:
                for rk_layer,alea_layer in zip(self.rk_layers,self.alea_layers):
                    x = rk_layer(x)
                    alea = alea_layer(alea)
            else:
                for rk_layer in self.rk_layers:
                    x = rk_layer(x)
                for alea_layer in self.alea_layers:
                    alea = alea_layer(alea)

            return x,alea
        else:
            for rk_layer in self.rk_layers:
                x = rk_layer(x)
            return x

    def kl_div(self) -> float:
        if self.aleatoric:
            reg_head = sum(lyr.equal_size_conv.kl_div() for lyr in self.rk_layers)
            alea_head = sum(lyr.equal_size_conv.kl_div() for lyr in self.alea_layers)
            return reg_head + alea_head
        else:
            return sum(lyr.equal_size_conv.kl_div() for lyr in self.rk_layers)


    def sample(self,x,num_samples=10000):
        
        inputs = x.detach().cpu().numpy()
        temp = []
        for j in range(len(inputs)):
            temp.append(np.expand_dims(inputs[j],0).repeat(num_samples,0))
        inputs = torch.tensor(np.concatenate(temp)).to(x.device)

        if self.aleatoric:
            pred,log_devs2 = self.forward(inputs)
            pred = pred.reshape(-1,num_samples,pred.shape[1],pred.shape[-1])
            epistemic = pred.std(1)
            avg_pred = pred.mean(1)
            aleatoric = log_devs2.exp().sqrt().reshape(-1,num_samples,log_devs2.shape[1],log_devs2.shape[-1]).mean(1)
            if self.multi_dim:
                return avg_pred,epistemic,aleatoric
            else:
                return avg_pred.cpu().numpy().flatten(),epistemic.cpu().numpy().flatten(),aleatoric.cpu().numpy().flatten()

        else:
            pred = self.forward(inputs)
            pred = pred.reshape(-1,num_samples,pred.shape[1],pred.shape[-1])
            epistemic = pred.std(1)
            avg_pred = pred.mean(1)
            if self.multi_dim:
                return avg_pred,epistemic
            else:
                return avg_pred.cpu().numpy().flatten(),epistemic.cpu().numpy().flatten()