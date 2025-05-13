from typing import Union
import torch.nn as nn
import torch
from traj_sec import SHA_POS_Sampler

def pad_t_like_x_pl(t, x,b_l):
    if isinstance(t, (float, int)):
        return t
    return t[b_l].unsqueeze(1)



class CFM_CD:
    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = sigma


    def compute_mu_t(self, x0, x1,t,b_l):
        t = pad_t_like_x_pl(t, x0,b_l)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon,b_l):
        mu_t = self.compute_mu_t(x0, x1, t,b_l)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x_pl(sigma_t, x0,b_l)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1,b_l, t=None, return_noise=False,pre_trian=False):
        if t is None:
            t = torch.rand(b_l.max().item()+1).type_as(x0)
        # assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps,b_l)



        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)




class CFM_CD_PERTURB(nn.Module):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__()
        self.x_perturb = nn.Parameter(torch.tensor(1.5))##notice!!
        self.h_perturb = nn.Parameter(torch.tensor(1.5))
        self.sigma = sigma


    def forward(self,x0, x1,x_bar,b_l, t=None, return_noise=False,pre_trian=False):
        x_bar=torch.cat([x_bar[:,:3]*self.x_perturb,x_bar[:,3:]*self.h_perturb],dim=1)
        return self.sample_location_and_conditional_flow(x0, x1,x_bar,b_l, t=None, return_noise=False,pre_trian=False)

    def compute_mu_t(self, x0, x1,x_bar,t,b_l):

        t = pad_t_like_x_pl(t, x0,b_l)
        return t * x1 + (1 - t) * x0+t*(1-t)*x_bar

    def compute_sigma_t(self, t):

        del t
        return self.sigma

    def sample_xt(self, x0, x1,x_bar, t, epsilon,b_l):

        mu_t = self.compute_mu_t(x0, x1, x_bar,t,b_l)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x_pl(sigma_t, x0,b_l)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1,x_bar, t, xt):

        return x1 - x0+x_bar*(1-2*t)

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1,x_bar,b_l, t=None, return_noise=False,pre_trian=False):

        if t is None:
            t = torch.rand(b_l.max().item()+1).type_as(x0)
        # assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, x_bar,t, eps,b_l)

        ut = self.compute_conditional_flow(x0, x1,x_bar, pad_t_like_x_pl(t, x0,b_l), xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):

        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class POS_HA(CFM_CD):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.Sha_sampler = SHA_POS_Sampler()

    def sample_location_and_conditional_flow(self, x0, x1, bl,t=None, return_noise=False,pre_train=False):

        x0, x1 = self.Sha_sampler.sample_plan(x0, x1,bl,pre_train)
        return x0
