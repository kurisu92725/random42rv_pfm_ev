import os
import shutil
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
import argparse
import wandb
import losses
from prior_distributions import PositionFeaturePrior_PFM
import torch
import time
import numpy as np
from flow_matching_utils import POS_HA,CFM_CD_PERTURB
from egnn.DotGat import DotPre_FM


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c+1]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)




def get_auroc_bond(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)

    return avg_auroc / len(y_true)


parser = argparse.ArgumentParser(description='EFG_FM_TTRJ_debug')
parser.add_argument('config', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--logdir', type=str, default='./logs_ef')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--n_iter_test', type=int, default=5000)##5000
parser.add_argument('--interrupt_protection', type=int, default=2000)
parser.add_argument('--resume', type=bool, default=False,help='')
parser.add_argument('--dp', type=eval, default=True,help='True | False')
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--clip_grad', type=eval, default=True,help='True | False')
parser.add_argument('--trace', type=str, default='hutch',help='hutch | exact')
args = parser.parse_args()

# Load configs
config = misc.load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
misc.seed_all(config.train.seed)

# Logging
log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
vis_dir = os.path.join(log_dir, 'vis')
os.makedirs(vis_dir, exist_ok=True)
logger = misc.get_logger('train', log_dir)
writer = torch.utils.tensorboard.SummaryWriter(log_dir)
logger.info(args)
logger.info(config)
shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))



# Transforms
protein_featurizer = trans.FeaturizeProteinAtom()
ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
transform_list = [
    protein_featurizer,
    ligand_featurizer,
    trans.FeaturizeLigandBond(),
]
if config.data.transform.random_rot:
    transform_list.append(trans.RandomRotation())
transform = Compose(transform_list)


# Datasets and loaders
logger.info('Loading dataset...')
dataset, subsets = get_dataset(
    config=config.data,
    transform=transform
)
train_set, val_set= subsets['train'], subsets['test']
logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
collate_exclude_keys = ['ligand_nbh_list']
train_loader = DataLoader(
    train_set,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.num_workers,
    follow_batch=FOLLOW_BATCH,
    exclude_keys=collate_exclude_keys
)
val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                        follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

# Model
logger.info('Building model...')
wandb.init(project='EFG_FM_TTRJ_debug', name=config.train.exp_name, config=args)
wandb.save('*.txt')

#get predictor
x_bar_predictor = DotPre_FM(config.modelp, 27, 12)
h_bar_predictor = DotPre_FM(config.modelp, 27, 12)
net_dynamics=DotPre_FM(config.model,27,12)

if config.train.training_mode == 'pfm':
    x_pre_dict = torch.load(config.data.x_predictor)
    h_pre_dict = torch.load(config.data.h_predictor)
    x_bar_predictor.load_state_dict(x_pre_dict['net_dynamics_state_dict'])
    h_bar_predictor.load_state_dict(h_pre_dict['net_dynamics_state_dict'])
    x_bar_predictor = x_bar_predictor.to(args.device)
    h_bar_predictor = h_bar_predictor.to(args.device)
    for param in x_bar_predictor.parameters():
        param.requires_grad = False
    for param in h_bar_predictor.parameters():
        param.requires_grad = False


if config.train.training_mode == 'pfm':
    get_losses= losses.compute_loss_pfm
elif config.train.training_mode == 'xp':
    get_losses= losses.compute_loss_x_predictor
else:
    get_losses= losses.compute_loss_h_predictor

FM = CFM_CD_PERTURB(config.train.sigma_pos).to(args.device)
FM_align=POS_HA(config.train.sigma_pos)
prior = PositionFeaturePrior_PFM(n_dim=3, in_node_nf=12)
net_dynamics=net_dynamics.to(args.device)
optimizer = utils_train.get_optimizer(config.train.optimizer, net_dynamics)
scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


def train(args,loader,test_loader,epoch,best_total_loss):

    loss_epoch = []
    optimizer.zero_grad()
    tmp_best_total_loss=best_total_loss

    for i, batch in tqdm(enumerate(loader)):
        net_dynamics.train()
        optimizer.zero_grad()
        batch=batch.to(args.device)
        total_loss, field_loss, pos_plot, h_plot_hyper = get_losses(config,x_bar_predictor,h_bar_predictor,FM,FM_align, prior, net_dynamics,batch,test=False)

        total_loss.backward()
        orig_grad_norm = clip_grad_norm_(list(net_dynamics.parameters()), config.train.max_grad_norm)
        optimizer.step()

        if i % args.interrupt_protection == 0:
            torch.save({
                'epoch': epoch,
                'iter_num': i,
                'net_dynamics_state_dict': net_dynamics.state_dict(),
                'FM_state_dict': FM.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'{log_dir}/checkpoints/protection.pth')

        if i % args.n_iter_test == 0:
            total_val,field_val,pos_plot_val,h_plot_hyper_val,atom_auroc = test(test_loader) #
            scheduler.step(total_val)

            if total_val < tmp_best_total_loss:
                tmp_best_total_loss = total_val
                torch.save({
                    'epoch': epoch,
                    'iter_num': i,
                    'net_dynamics_state_dict': net_dynamics.state_dict(),
                    'FM_state_dict': FM.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f'{log_dir}/checkpoints/{epoch}_{i}_best.pth')
                logger.info(f'Val loss: {total_val:.3f},'
                            f"Val Field Loss {field_val:.3f},"
                            f"Val pos_plot: {pos_plot_val:.3f},"
                            # f"Val bond_plot: {bond_plot_hyper_val:.3f},"
                            f"Val h_plot: {h_plot_hyper_val:.3f},"
                            f'Val AUROC: {atom_auroc:.3f}'
                            # f'Val BOND AUROC: {bond_auroc:.3f}'

                      )
            wandb.log({"Val AUROC": atom_auroc}, commit=False)
            # wandb.log({"Val BOND AUROC": bond_auroc}, commit=False)
            wandb.log({"Val loss": total_val}, commit=False)
            wandb.log({"Val Field Loss": field_val}, commit=False)##
            wandb.log({"Val Pos Loss": pos_plot_val}, commit=False)
            # wandb.log({"Val Bond_hyper Loss": bond_plot_hyper_val}, commit=False)
            wandb.log({"Val H_hyper Loss": h_plot_hyper_val}, commit=False)

        if i % 100 == 0:
            logger.info(f"\repoch: {epoch}, iter: {i}/{len(loader)}, "
                        f"Loss {total_loss.item():.3f},"
                        f"Field Loss {field_loss.item():.3f},"
                        f"pos_plot: {pos_plot.item():.3f},"
                        # f"bond_plot: {bond_plot_hyper.item():.3f},"
                        f"h_plot: {h_plot_hyper.item():.3f},"
                        f"GradNorm: {orig_grad_norm:.1f},"
                        f"Lr: {(optimizer.param_groups[0]['lr']):.6f}"
                        )



        loss_epoch.append(total_loss.item())

        # wandb.log({"mean(abs(z))": mean_abs_z}, commit=False)
        wandb.log({"Batch Total Loss": total_loss.item()}, commit=False)
        wandb.log({"Batch Field Loss": field_loss.item()}, commit=False)
        wandb.log({"Batch Pos Loss": pos_plot.item()}, commit=False)
        # wandb.log({"Batch Bond_hyper Loss": bond_plot_hyper.item()}, commit=False)
        wandb.log({"Batch H_hyper Loss": h_plot_hyper.item()}, commit=True)
        # wandb.log({"Field Loss": field_loss.item()}, commit=True)

    wandb.log({"Train Epoch": np.mean(loss_epoch)}, commit=False)
    return tmp_best_total_loss,np.mean(loss_epoch)

def test(loader):
    with torch.no_grad():
        net_dynamics.eval()
        n_samples = 0
        total_loss_epoch = 0
        confield_epoch=0
        pos_plot_epoch=0
        h_plot_hyper_epoch=0
        bond_plot_hyper_epoch=0
        all_pred_v, all_true_v = [], []
        all_pre_bond, all_true_bond = [], []

        for i, batch in (enumerate(loader)):
            # Get data
            batch = batch.to(args.device)
            for ii in range(10):
                total_loss, field_loss, pos_plot, h_plot_hyper, bond_plot_hyper, pre_v, real_v =get_losses(config,x_bar_predictor,h_bar_predictor,FM,FM_align, prior, net_dynamics,batch,test=True)

                batch_size = batch.protein_element_batch.max().item()+1
                total_loss_epoch += total_loss.item() * batch_size

                confield_epoch+=field_loss.item()*batch_size
                pos_plot_epoch+=pos_plot.item()*batch_size
                h_plot_hyper_epoch+=h_plot_hyper.item()*batch_size
                # bond_plot_hyper_epoch += bond_plot_hyper.item() * batch_size   ,pre_bond,real_bond

                n_samples += batch_size
                all_pred_v.append(pre_v.detach().cpu().numpy())
                all_true_v.append(real_v.detach().cpu().numpy())
                # all_pre_bond.append(pre_bond.detach().cpu().numpy())
                # all_true_bond.append(real_bond.detach().cpu().numpy())

        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),feat_mode=config.data.transform.ligand_atom_mode)

        # bond_auroc=get_auroc_bond(np.concatenate(all_true_bond), np.concatenate(all_pre_bond, axis=0))


    return total_loss_epoch/n_samples,confield_epoch/n_samples,pos_plot_epoch/n_samples,h_plot_hyper_epoch/n_samples,atom_auroc #,bond_auroc  ,bond_plot_hyper_epoch/n_samples


def main():
    if args.resume:
        pass

    if args.dp and torch.cuda.device_count() > 1:
        pass


    best_total_train = 1e8
    best_total_val = 1e8
    best_nll_test = 1e8
    for epoch in range(0, args.n_epochs):
        start_epoch = time.time()
        best_total_val,tmp_train_epoch=train(args,train_loader,val_loader,epoch,best_total_val)
        if tmp_train_epoch<best_total_train:
            best_total_train=tmp_train_epoch
        logger.info(f"Epoch took {time.time() - start_epoch:.1f} seconds.")


if __name__=="__main__":
    main()
