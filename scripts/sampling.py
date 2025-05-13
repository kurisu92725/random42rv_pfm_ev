import numpy as np
import torch
from utils.evaluation import atom_num
from torch_geometric.data import Batch
from datasets.pl_data import FOLLOW_BATCH
from torch_geometric.utils import add_self_loops, scatter
import torch.nn.functional as F
from tqdm import tqdm



def sample_ode(device,one_data,x_pre,h_pre,x_pare,h_para, prior, net_dynamics,sample_num_atoms='prior',num_mole=10):
    ligand_v=[]
    ligand_p=[]

    batch = Batch.from_data_list([one_data.clone() for _ in range(num_mole)], follow_batch=FOLLOW_BATCH).to(device)
    protein_pos = batch.protein_pos
    protein_v = batch.protein_atom_feature.float()
    b_p = batch.protein_element_batch
    p_mean = scatter(protein_pos, b_p, dim=0, reduce='mean')#

    protein_pos_input=protein_pos-p_mean[b_p]


    if sample_num_atoms == 'prior':
        pocket_size = atom_num.get_space_size(one_data.protein_pos.detach().cpu().numpy())
        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(num_mole)]
        b_l = torch.repeat_interleave(torch.arange(num_mole), torch.tensor(ligand_num_atoms)).to(device)



    p_xh=torch.cat([protein_pos_input,protein_v],dim=-1)



    net_dynamics.forward = net_dynamics.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)

    x_pre.forward = x_pre.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)
    h_pre.forward = h_pre.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)




    z_x0, z_h0 = prior.sample(ligand_num_atoms,device)
    const_k = 5.0
    norm_clip = 0.9
    num_steps = 20#
    z_h0=z_h0*const_k


    z_h0_prob=F.softmax(z_h0,dim=-1)
    z_h0_v=torch.argmax(z_h0_prob+1e-8,1)
    z_h0_v = F.one_hot(z_h0_v, num_classes=12).float()

    ts=torch.linspace(5.e-2, 1.0, num_steps).to(z_x0.device)#
    t_1=ts[0]
    z_x1=z_x0.clone()
    z_h1=z_h0.clone()
    z_h1v=z_h0_v.clone()

    #el
    for t_2 in tqdm(ts[1:]):

        pre_z1,_=net_dynamics(t_1,torch.cat([z_x1,z_h1v],dim=-1))
        z_htv = torch.argmax(F.softmax(pre_z1[:, 3:], dim=-1) + 1e-8, 1)
        h_t = F.one_hot(z_htv, num_classes=12).float() * 2 * const_k - const_k

        ##perturb
        pre_x1_bar,_=x_pre(t_1,torch.cat([z_x0,F.one_hot(z_htv, num_classes=12).float()],dim=-1))
        pre_x1_bar=pre_x1_bar[:,0:3]*x_pare
        pre_h1_bar,_=h_pre(t_1,torch.cat([pre_z1[:,0:3],z_h0_v],dim=-1))
        pre_h1_bar=pre_h1_bar[:,3:]
        pre_h1_bar = torch.argmax(F.softmax(pre_h1_bar, dim=-1) + 1e-8, 1)
        pre_h1_bar = (F.one_hot(pre_h1_bar, num_classes=12).float() * 2 * const_k - const_k)*h_para

        dt=(t_2-t_1)*(torch.ones(z_x1.size(0),1).to(z_x1.device))

        #update position
        z_x1=(pre_z1[:,0:3]-z_x0+(1-2*t_1)*pre_x1_bar)*dt+z_x1

        #update type in logit space
        z_h1=(h_t-z_h0+(1-2*t_1)*pre_h1_bar)*dt+z_h1
        z_h1v=torch.argmax(F.softmax(z_h1,dim=-1)+1e-8,1)
        z_h1v = F.one_hot(z_h1v, num_classes=12).float()

        t_1=t_2



    #Final step
    t_1 = ts[-1]
    pre_final_z,_=net_dynamics(t_1,torch.cat([z_x1,z_h1v],dim=-1))
    pre_pos=pre_final_z[:,0:3]+p_mean[b_l]
    pre_v=torch.argmax(F.softmax(pre_final_z[:,3:],dim=-1)+1e-8,1)+1


    #Output
    ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
    atom_type = pre_v.cpu().numpy()
    x_squeeze = pre_pos.cpu().detach().numpy().astype(np.float64)
    ligand_p+= [x_squeeze[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(num_mole)]  # num_samples * [num_atoms_i, 3]
    ligand_v+= [atom_type[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(num_mole)]  # num_samples * [num_atoms_i, 3]
    return ligand_p,ligand_v



def sample_sde(device,one_data,x_pre,h_pre,x_pare,h_para, prior, net_dynamics,sample_num_atoms='prior',num_mole=10):
    ligand_v=[]
    ligand_p=[]

    batch = Batch.from_data_list([one_data.clone() for _ in range(num_mole)], follow_batch=FOLLOW_BATCH).to(device)
    protein_pos = batch.protein_pos
    protein_v = batch.protein_atom_feature.float()
    b_p = batch.protein_element_batch


    p_mean = scatter(protein_pos, b_p, dim=0, reduce='mean')#
    protein_pos_input=protein_pos-p_mean[b_p]

    if sample_num_atoms == 'prior':
        pocket_size = atom_num.get_space_size(one_data.protein_pos.detach().cpu().numpy())
        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(num_mole)]
        b_l = torch.repeat_interleave(torch.arange(num_mole), torch.tensor(ligand_num_atoms)).to(device)



    p_xh=torch.cat([protein_pos_input,protein_v],dim=-1)



    net_dynamics.forward = net_dynamics.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)

    x_pre.forward = x_pre.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)
    h_pre.forward = h_pre.wrap_forward(b_l, p_xh, b_p, node_mask=None, edge_mask=None,
                                                     context=None, reverse=False, pre_train=False)




    z_x0, z_h0 = prior.sample(ligand_num_atoms,device)#

    const_k = 5.0
    norm_clip = 0.9
    num_steps = 100#500 |400 0.0025 0.05|100
    z_h0=z_h0*const_k##
    path_sigma=0.1##0.1 0.2 0.3


    z_h0_prob=F.softmax(z_h0,dim=-1)
    z_h0_v=torch.argmax(z_h0_prob+1e-8,1)
    z_h0_v = F.one_hot(z_h0_v, num_classes=12).float()


    #init

    ts=torch.linspace(1e-2, 1.0, num_steps).to(z_x0.device)#
    t_1=ts[0]
    z_x1=z_x0.clone()
    z_h1=z_h0.clone()
    z_h1v=z_h0_v.clone()

    #eluer
    for t_2 in tqdm(ts[1:]):

        pre_z1,_=net_dynamics(t_1,torch.cat([z_x1,z_h1v],dim=-1))
        z_htv = torch.argmax(F.softmax(pre_z1[:, 3:], dim=-1) + 1e-8, 1)
        h_t = F.one_hot(z_htv, num_classes=12).float() * 2 * const_k - const_k
        ##perturb
        pre_x1_bar,_=x_pre(t_1,torch.cat([z_x0,F.one_hot(z_htv, num_classes=12).float()],dim=-1))
        pre_x1_bar=pre_x1_bar[:,0:3]*x_pare
        pre_h1_bar,_=h_pre(t_1,torch.cat([pre_z1[:,0:3],z_h0_v],dim=-1))
        pre_h1_bar=pre_h1_bar[:,3:]##not sf
        pre_h1_bar = torch.argmax(F.softmax(pre_h1_bar, dim=-1) + 1e-8, 1)
        pre_h1_bar = (F.one_hot(pre_h1_bar, num_classes=12).float() * 2 * const_k - const_k)*h_para



        dt=(t_2-t_1)*(torch.ones(z_x1.size(0),1).to(z_x1.device))

        #pos
        x_pre_avg=t_1*pre_z1[:,0:3]+(1-t_1)*z_x0+t_1*(1-t_1)*pre_x1_bar
        x_pre_score_v=(x_pre_avg-z_x1)/2  #/2 *2 *4.5

        z_x1=(pre_z1[:,0:3]-z_x0+(1-2*t_1)*pre_x1_bar+x_pre_score_v)*dt+z_x1+path_sigma*(0.1)*torch.randn_like(z_x1)##sqrt_time

        #h
        h_pre_avg = t_1 * h_t + (1 - t_1) * z_h0 + t_1 * (1 - t_1) * pre_h1_bar
        h_pre_score_v = (h_pre_avg - z_h1)/2  ##/2 *2 *4.5



        z_h1=(h_t-z_h0+(1-2*t_1)*pre_h1_bar+h_pre_score_v)*dt+z_h1+path_sigma*(0.1)*torch.randn_like(z_h1)
        z_h1v=torch.argmax(F.softmax(z_h1,dim=-1)+1e-8,1)
        z_h1v = F.one_hot(z_h1v, num_classes=12).float()
        t_1=t_2



    ##final step
    t_1 = ts[-1]

    pre_final_z,_=net_dynamics(t_1,torch.cat([z_x1,z_h1v],dim=-1))

    pre_pos=pre_final_z[:,0:3]+p_mean[b_l]
    pre_v=torch.argmax(F.softmax(pre_final_z[:,3:],dim=-1)+1e-8,1)+1#


    ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)

    atom_type = pre_v.cpu().numpy()#
    x_squeeze = pre_pos.cpu().detach().numpy().astype(np.float64)

    ligand_p+= [x_squeeze[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(num_mole)]  # num_samples * [num_atoms_i, 3]
    ligand_v+= [atom_type[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(num_mole)]  # num_samples * [num_atoms_i, 3]

    return ligand_p,ligand_v
