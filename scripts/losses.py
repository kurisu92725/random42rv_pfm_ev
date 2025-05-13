import torch
from utils.remove_mean  import remove_mean_crossdock
from torch_geometric.utils import add_self_loops, scatter
import torch.nn.functional as F

def G_fn(protein_coords, x,sigma):##sigma 25 gama 10   2;4
    # protein_coords: (n,3) , x: (m,3), output: (m,)
    e = torch.exp(-torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n) m-l n-p
    return -sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(protein_coords, ligand_coords,sigma, surface_ct):
    loss = torch.mean(torch.clamp(surface_ct - G_fn(protein_coords, ligand_coords, sigma), min=0))
    return loss



def type2prob_traj(norm_k,x):
    return F.one_hot(x, 12).float()*2*norm_k-norm_k##13


def sample_v_from_softmax(p):
    return torch.multinomial(p+1e-8,1).squeeze(1)


def sym_t_sampler(batch_ligand,pos):
    part=4
    t_sample=torch.empty(0).type_as(pos)
    t_0=(torch.rand((batch_ligand.max().item() + 1)//part).type_as(pos))/part
    for i in range(part):
        t_sample=torch.cat([t_sample,t_0+i*(1/part)],dim=0)

    random_index=torch.randperm(t_sample.size(0))
    return t_sample[random_index]

def compute_loss_pfm(config,x_bar_predictor,h_bar_predictor,FM,FM_align, prior, net_dynamics,batch,test):
    ##params
    norm_clip = config.train.loss_hyper.norm_clip
    norm_k = config.train.loss_hyper.norm_k
    lambda_h = config.train.loss_hyper.lambda_h
    lambda_edgeL=config.train.loss_hyper.lambda_edgeL
    lambda_bond_tp=config.train.loss_hyper.lambda_bond_tp
    lambda_angle=config.train.loss_hyper.lambda_angle
    surf_sigma=config.train.loss_hyper.surf_sigma
    surf_gamma=config.train.loss_hyper.surf_gamma
    lambda_surf=config.train.loss_hyper.lambda_surf

    batch_size=config.train.batch_size
    protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std

    protein_pos = batch.protein_pos+protein_noise
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    ligand_pos = batch.ligand_pos
    ligand_v = batch.ligand_atom_feature_full-1
    batch_ligand = batch.ligand_element_batch

    ligand_bond_index = batch.ligand_bond_index
    ligand_bond = batch.ligand_bond_type-1
    ligand_bond_distance_square = (ligand_pos[ligand_bond_index[0, :]] - ligand_pos[ligand_bond_index[1, :]]).square().sum(dim=-1)

    protein_pos,ligand_pos = remove_mean_crossdock(protein_pos, ligand_pos, batch_protein, batch_ligand)
    p_xh = torch.cat([protein_pos, protein_v], dim=-1)
    net_dynamics.forward = net_dynamics.wrap_forward(batch_ligand, p_xh, batch_protein, node_mask=None, edge_mask=ligand_bond_index, context=None,reverse=False,pre_train=False)

    #sample noise
    pos_0,lh_0=prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0=lh_0*norm_k
    if batch_ligand.max().item() + 1<batch_size:
        t = torch.rand(batch_ligand.max().item() + 1).type_as(pos_0)
    else:
        t = sym_t_sampler(batch_ligand,pos_0)

    x0_align=FM_align.sample_location_and_conditional_flow(torch.cat((pos_0,lh_0),dim=-1), torch.cat((ligand_pos,type2prob_traj(norm_k,ligand_v)),dim=-1), batch_ligand, t=t)
    # x0_align=torch.cat((pos_0,lh_0),dim=-1)

    #perturb
    x_bar_input=torch.cat([x0_align[:,:3],F.one_hot(ligand_v, 12).float()],dim=-1)
    h_bar_input=torch.cat([ligand_pos,F.one_hot(sample_v_from_softmax(F.softmax(lh_0, -1)), 12).float()],dim=-1)

    with torch.no_grad():
        x_bar_predictor.forward = x_bar_predictor.wrap_forward(batch_ligand, p_xh, batch_protein, node_mask=None,edge_mask=None, context=None, reverse=False, pre_train=False)
        h_bar_predictor.forward = h_bar_predictor.wrap_forward(batch_ligand, p_xh, batch_protein, node_mask=None,edge_mask=None, context=None, reverse=False, pre_train=False)
        x_bar,_= x_bar_predictor(t,x_bar_input)
        x_bar=x_bar[:,:3]
        h_bar,_= h_bar_predictor(t,h_bar_input)
        h_bar=h_bar[:,3:]
        h_bar = torch.multinomial(F.softmax(h_bar, dim=-1) + 1e-8, 1).squeeze(1)
        h_bar = F.one_hot(h_bar, num_classes=12).float() * 2 * norm_k - norm_k
        total_bar=torch.cat([x_bar,h_bar],dim=-1)

    _, pos_h_t, u_p_h_t = FM(x0_align, torch.cat((ligand_pos,type2prob_traj(norm_k,ligand_v)),dim=-1), total_bar,batch_ligand, t=t)
    pos_t, h_t = pos_h_t[:,:3], pos_h_t[:,3:]

    h_v_t = sample_v_from_softmax(F.softmax(h_t,-1))
    h_v_t_input=F.one_hot(h_v_t, 12).float()


    zt=(torch.cat([pos_t,h_v_t_input],dim=-1))
    pre_ut,pre_bond=net_dynamics(t,zt)

    #position loss
    pos_loss=scatter((pre_ut[:,:3]-ligand_pos).pow(2).sum(dim=1,keepdim=True),batch_ligand,dim=0,reduce='mean')

    #bond length loss
    ligand_bond_distance_predict_square = (pre_ut[ligand_bond_index[0, :], :3] - pre_ut[ligand_bond_index[1, :], :3]).square().sum(dim=-1)
    pos_loss_distance = scatter((ligand_bond_distance_predict_square - ligand_bond_distance_square).abs().unsqueeze(1),batch.ligand_bond_type_batch, dim=0, reduce='mean')

    #surface loss
    pos_loss_surf=torch.empty(0, 1).to(ligand_pos.device)
    for i in range(batch_ligand.max().item()+1):
        mask_ligand = (batch_ligand == i)
        mask_protein = (batch_protein == i)
        pos_pre_mask = (pre_ut[:,:3])[mask_ligand]
        pos_protein_mask = (protein_pos)[mask_protein]
        pos_loss_surf=torch.cat((pos_loss_surf,compute_body_intersection_loss(pos_protein_mask,pos_pre_mask,surf_sigma,surf_gamma).unsqueeze(0).unsqueeze(1)),dim=0)


    pos_loss = pos_loss + lambda_edgeL*pos_loss_distance+lambda_surf*pos_loss_surf

    #atom type loss
    h_loss = F.cross_entropy(pre_ut[:, 3:], ligand_v, reduction='mean')

    #bond type loss | omit
    bond_type_loss = F.cross_entropy(pre_bond, ligand_bond, reduction='mean')

    cond_field_loss=pos_loss+h_loss*lambda_h+bond_type_loss*lambda_bond_tp
    total_loss=cond_field_loss .mean()
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        pre_v=F.softmax(pre_ut[:,3:],dim=-1)
        # pre_bond_real=F.softmax(pre_bond,dim=-1)

        return total_loss,cond_field_loss.mean() ,pos_loss.mean(),h_loss.mean(),bond_type_loss.mean(),pre_v,ligand_v

    return total_loss,cond_field_loss.mean(),pos_loss.mean(),h_loss.mean()


def compute_loss_x_predictor(config,x_bar_predictor,h_bar_predictor,FM,FM_align, prior, net_dynamics,batch,test):
    ##params
    norm_clip = config.train.loss_hyper.norm_clip
    norm_k = config.train.loss_hyper.norm_k
    lambda_h = config.train.loss_hyper.lambda_h
    lambda_edgeL = config.train.loss_hyper.lambda_edgeL
    lambda_bond_tp = config.train.loss_hyper.lambda_bond_tp
    lambda_angle = config.train.loss_hyper.lambda_angle
    surf_sigma = config.train.loss_hyper.surf_sigma
    surf_gamma = config.train.loss_hyper.surf_gamma
    lambda_surf = config.train.loss_hyper.lambda_surf
    bond_type_placeholder=None

    batch_size = config.train.batch_size
    protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std

    protein_pos = batch.protein_pos + protein_noise
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    ligand_pos = batch.ligand_pos
    ligand_v = batch.ligand_atom_feature_full - 1
    batch_ligand = batch.ligand_element_batch

    ligand_bond_index = batch.ligand_bond_index
    ligand_bond = batch.ligand_bond_type-1
    ligand_bond_distance_square = (
                ligand_pos[ligand_bond_index[0, :]] - ligand_pos[ligand_bond_index[1, :]]).square().sum(dim=-1)

    protein_pos, ligand_pos = remove_mean_crossdock(protein_pos, ligand_pos, batch_protein, batch_ligand)
    p_xh = torch.cat([protein_pos, protein_v], dim=-1)
    net_dynamics.forward = net_dynamics.wrap_forward(batch_ligand, p_xh, batch_protein, node_mask=None,edge_mask=ligand_bond_index, context=None, reverse=False,pre_train=False)


    pos_0, lh_0 = prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0 = lh_0 * norm_k
    if batch_ligand.max().item() + 1 < batch_size:
        t = torch.rand(batch_ligand.max().item() + 1).type_as(pos_0)
    else:
        t = sym_t_sampler(batch_ligand, pos_0)
    x0_align = FM_align.sample_location_and_conditional_flow(torch.cat((pos_0, lh_0), dim=-1),torch.cat((ligand_pos, type2prob_traj(norm_k, ligand_v)),dim=-1), batch_ligand, t=t)


    #add noise to atom type
    ligand_v_add_noise=type2prob_traj(norm_k,ligand_v)
    ligand_v_add_noise=ligand_v_add_noise+torch.randn_like(ligand_v_add_noise)*2.5
    ligand_v_add_noise = torch.multinomial(F.softmax(ligand_v_add_noise, dim=-1) + 1e-8, 1).squeeze(1)
    z_input=torch.cat([x0_align[:,:3],F.one_hot(ligand_v_add_noise, 12).float()],dim=-1)


    pre_ut,_=net_dynamics(t,z_input)#

    #loss
    pos_loss=scatter((pre_ut[:,:3]-ligand_pos).pow(2).sum(dim=1,keepdim=True),batch_ligand,dim=0,reduce='mean')
    ligand_bond_distance_predict_square = (pre_ut[ligand_bond_index[0, :], :3] - pre_ut[ligand_bond_index[1, :], :3]).square().sum(dim=-1)
    pos_loss_distance = scatter((ligand_bond_distance_predict_square - ligand_bond_distance_square).abs().unsqueeze(1),batch.ligand_bond_type_batch, dim=0, reduce='mean')

    pos_loss_surf = torch.empty(0, 1).to(ligand_pos.device)

    for i in range(batch_ligand.max().item() + 1):
        mask_ligand = (batch_ligand == i)
        mask_protein = (batch_protein == i)
        pos_pre_mask = (pre_ut[:, :3])[mask_ligand]
        pos_protein_mask = (protein_pos)[mask_protein]
        pos_loss_surf = torch.cat((pos_loss_surf,
                                   compute_body_intersection_loss(pos_protein_mask, pos_pre_mask, surf_sigma,
                                                                  surf_gamma).unsqueeze(0).unsqueeze(1)), dim=0)


    pos_loss = pos_loss + lambda_edgeL*pos_loss_distance+lambda_surf*pos_loss_surf
    h_loss=F.cross_entropy(pre_ut[:,3:],ligand_v,reduction='mean')
    cond_field_loss=pos_loss+h_loss*lambda_h
    total_loss=cond_field_loss .mean()
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        pre_v=F.softmax(pre_ut[:,3:],dim=-1)

        return total_loss,cond_field_loss.mean() ,pos_loss.mean(),h_loss.mean(),bond_type_placeholder,pre_v,ligand_v
    return total_loss,cond_field_loss.mean(),pos_loss.mean(),h_loss.mean()

def compute_loss_h_predictor(config,x_bar_predictor,h_bar_predictor,FM,FM_align, prior, net_dynamics,batch,test):
    ##params
    norm_clip = config.train.loss_hyper.norm_clip
    norm_k = config.train.loss_hyper.norm_k
    lambda_h = config.train.loss_hyper.lambda_h
    lambda_edgeL = config.train.loss_hyper.lambda_edgeL
    lambda_bond_tp = config.train.loss_hyper.lambda_bond_tp
    lambda_angle = config.train.loss_hyper.lambda_angle
    surf_sigma = config.train.loss_hyper.surf_sigma
    surf_gamma = config.train.loss_hyper.surf_gamma
    lambda_surf = config.train.loss_hyper.lambda_surf
    bond_type_placeholder = None

    batch_size = config.train.batch_size
    protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std

    protein_pos = batch.protein_pos + protein_noise
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    ligand_pos = batch.ligand_pos
    ligand_v = batch.ligand_atom_feature_full - 1
    batch_ligand = batch.ligand_element_batch

    ligand_bond_index = batch.ligand_bond_index
    ligand_bond = batch.ligand_bond_type - 1
    ligand_bond_distance_square = (
            ligand_pos[ligand_bond_index[0, :]] - ligand_pos[ligand_bond_index[1, :]]).square().sum(dim=-1)

    protein_pos, ligand_pos = remove_mean_crossdock(protein_pos, ligand_pos, batch_protein, batch_ligand)
    p_xh = torch.cat([protein_pos, protein_v], dim=-1)
    net_dynamics.forward = net_dynamics.wrap_forward(batch_ligand, p_xh, batch_protein, node_mask=None,
                                                     edge_mask=ligand_bond_index, context=None, reverse=False,
                                                     pre_train=False)
    pos_0, lh_0 = prior.sample(ligand_pos.size(0), ligand_pos.device)
    lh_0 = lh_0 * norm_k
    if batch_ligand.max().item() + 1 < batch_size:
        t = torch.rand(batch_ligand.max().item() + 1).type_as(pos_0)
    else:
        t = sym_t_sampler(batch_ligand, pos_0)

    #add noise to atom position
    ligand_noise = torch.randn_like(batch.ligand_pos) * 0.5
    h_v_t = sample_v_from_softmax(F.softmax(lh_0, -1))
    z_input=torch.cat([ligand_pos+ligand_noise,F.one_hot(h_v_t, 12).float()],dim=-1)
    pre_ut,_=net_dynamics(t,z_input)#t(bs,)

    #field loss
    pos_loss=scatter((pre_ut[:,:3]-ligand_pos).pow(2).sum(dim=1,keepdim=True),batch_ligand,dim=0,reduce='mean')
    ligand_bond_distance_predict_square = (
            pre_ut[ligand_bond_index[0, :], :3] - pre_ut[ligand_bond_index[1, :], :3]).square().sum(dim=-1)
    pos_loss_distance = scatter((ligand_bond_distance_predict_square - ligand_bond_distance_square).abs().unsqueeze(1),
                                batch.ligand_bond_type_batch, dim=0, reduce='mean')

    pos_loss_surf = torch.empty(0, 1).to(ligand_pos.device)

    for i in range(batch_ligand.max().item() + 1):
        mask_ligand = (batch_ligand == i)
        mask_protein = (batch_protein == i)
        pos_pre_mask = (pre_ut[:, :3])[mask_ligand]
        pos_protein_mask = (protein_pos)[mask_protein]
        pos_loss_surf = torch.cat((pos_loss_surf,
                                   compute_body_intersection_loss(pos_protein_mask, pos_pre_mask, surf_sigma,
                                                                  surf_gamma).unsqueeze(0).unsqueeze(1)), dim=0)

    pos_loss = pos_loss + lambda_edgeL * pos_loss_distance + lambda_surf * pos_loss_surf
    h_loss=F.cross_entropy(pre_ut[:,3:],ligand_v,reduction='mean')

    cond_field_loss=pos_loss+lambda_h*h_loss
    total_loss=cond_field_loss .mean()
    if test:
        net_dynamics.forward = net_dynamics.unwrap_forward()
        pre_v=F.softmax(pre_ut[:,3:],dim=-1)

        return total_loss,cond_field_loss.mean() ,pos_loss.mean(),h_loss.mean(),bond_type_placeholder,pre_v,ligand_v#

    return total_loss,cond_field_loss.mean(),pos_loss.mean(),h_loss.mean()
