import torch
from scipy.optimize import linear_sum_assignment


class SHA_POS_Sampler:
    def __init__(self):
        super().__init__()

    def sample_plan(self, x0, x1,bl,pre_train=False):
        x0_batch=[]
        for i in range(bl.max().item()+1):
            mask = (bl == i)  #
            x0_=x0[:,:3]
            x1_=x1[:,:3]
            x0_tmp=x0_[mask]
            x1_tmp=x1_[mask]#
            x0_expand = x0_tmp.unsqueeze(0)
            x1_expand = x1_tmp.unsqueeze(1)
            diff = x1_expand - x0_expand#
            cost = torch.sum(diff ** 2, dim=-1)
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            ex_x0_idx = torch.tensor(col_ind).to(x0.device)
            x0_batch.append((x0[mask])[ex_x0_idx])


        x0=torch.cat(x0_batch,dim=0)
        return x0,x1
