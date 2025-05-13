import os
import shutil
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset

import argparse

import torch
import numpy as np

from sampling import sample_ode,sample_sde
from prior_distributions import PositionFeaturePrior_PFM
from flow_matching_utils import CFM_CD_PERTURB

from egnn.DotGat import DotPre_FM


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    parser.add_argument('--trace', type=str, default='hutch', help='hutch | exact')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    training_config=misc.load_config(config.training_yml)
    logger.info(config)
    misc.seed_all(config.sample.seed)
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = training_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=training_config.data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')


    net_dynamics = DotPre_FM(training_config.model, 27, 12)
    x_bar_predictor = DotPre_FM(training_config.modelp, 27, 12)
    h_bar_predictor = DotPre_FM(training_config.modelp, 27, 12)
    prior = PositionFeaturePrior_PFM(n_dim=3, in_node_nf=12)
    FM = CFM_CD_PERTURB(training_config.train.sigma_pos).to(args.device)



    x_pre_dict = torch.load(training_config.data.x_predictor)
    h_pre_dict = torch.load(training_config.data.h_predictor)
    x_bar_predictor.load_state_dict(x_pre_dict['net_dynamics_state_dict'])
    h_bar_predictor.load_state_dict(h_pre_dict['net_dynamics_state_dict'])
    net_dynamics.load_state_dict(ckpt['net_dynamics_state_dict'])
    FM.load_state_dict(ckpt['FM_state_dict'])
    x_bar_predictor = x_bar_predictor.to(args.device)
    h_bar_predictor = h_bar_predictor.to(args.device)
    net_dynamics = net_dynamics.to(args.device)

    x_para = FM.x_perturb.detach()
    h_para = FM.h_perturb.detach()



    logger.info(f'Successfully load the model! {config.model.checkpoint}')


    for data_id in tqdm(range(len(test_set))):
        data = test_set[data_id]
        with torch.no_grad():
            lp, lv = sample_ode(args.device, data, x_bar_predictor,h_bar_predictor,x_para,h_para, prior, net_dynamics, sample_num_atoms='prior', num_mole=10)

        result = {
            'data': data,
            'pred_ligand_pos': lp,
            'pred_ligand_v': lv,
        }

        logger.info('Sample done!')
        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))

