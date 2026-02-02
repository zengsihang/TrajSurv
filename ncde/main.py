from dataset import SurvivalDataset, my_collate_fn
from model import NCDESurv, DSMHead
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import argparse
import json
from tqdm import tqdm
import wandb
import torchcde
from evaluate import get_times, predict_surv, evaluate


def pretrain(
    dsm_head,
    train_dataloader,
    valid_dataloader,
    args
):
    optimizer = optim.AdamW(dsm_head.parameters(), lr=args.lr)
    best_valid_loss = 1e10
    pat = 0
    dsm_head.to(args.device)
    for epoch in range(100):
        dsm_head.train()
        pbar = tqdm(train_dataloader, desc=f'Pretrain Epoch {epoch}', postfix={'loss': 0.0})
        for x, t, e in pbar:
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            optimizer.zero_grad()
            loss = dsm_head.compute_unconditional_loss(t, e)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            wandb.log({'pretrain_loss': loss.item()})
        # validation
        dsm_head.eval()
        val_loss = 0
        for x, t, e in tqdm(valid_dataloader, desc='Pretrain Validation'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            loss = dsm_head.compute_unconditional_loss(t, e)
            val_loss += loss.item()
        val_loss /= len(valid_dataloader)
        wandb.log({'pretrain_valid_loss': val_loss})
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_shape = dsm_head.shape.detach().cpu().numpy()
            best_scale = dsm_head.scale.detach().cpu().numpy()
            pat = 0
        else:
            # patience and early stopping
            pat += 1
            if pat == args.patience * 2:
                break
    return best_shape, best_scale
    



def train(
    model, 
    optimizer, 
    train_dataloader, 
    valid_dataloader,
    args
):
    # train
    # use tqdm to show progress, description is the epoch, show loss in the progress bar postfix
    # use wandb to log
    
    # save args
    with open(os.path.join(wandb.run.dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    best_valid_loss = 1e10
    pat = 0
    model.to(args.device)
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}', postfix={'loss': 0.0})
        for x, t, e, sofa_label, x_length in pbar:
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            sofa_label = sofa_label.to(args.device)
            x_length = x_length.to(args.device)
            if args.time_decay:
                times = x[:, :, 0].squeeze()
                # use x_length to make times a vector
                times = [times[i, :x_length[i]] for i in range(len(times))]
                times = torch.cat(times).reshape(-1, 1)
            else:
                times = None
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            optimizer.zero_grad()
            loss_dict = model.compute_loss(coeffs, t, e, sofa_label, x_length, times)
            loss = loss_dict['total_loss']
            try:
                loss.backward()
            except:
                # catch the error
                print('Error in backward')
                continue
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            for k, v in loss_dict.items():
                wandb.log({'train'+k: v.item()})
        # validation
        model.eval()
        val_loss = 0
        for x, t, e, sofa_label, x_length in tqdm(valid_dataloader, desc='Validation'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            sofa_label = sofa_label.to(args.device)
            x_length = x_length.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            loss_dict = model.compute_loss(coeffs, t, e, sofa_label, x_length)
            loss = loss_dict['total_loss']
            val_loss += loss.item()
        val_loss /= len(valid_dataloader)
        wandb.log({'valid_loss': val_loss})
        pred_surv = predict_surv(model, valid_dataloader, args.times, args)
        
        cis, brs, roc_auc = evaluate(train_dataloader, valid_dataloader, pred_surv, args.times)
        wandb.log({'val_cis': np.mean(cis)})
        wandb.log({'val_brs': np.mean(brs)})
        wandb.log({'val_roc_auc': np.mean([roc_auc[i][0] for i in range(len(args.times))])})
        
        if -np.mean(cis) < best_valid_loss:
            best_valid_loss = -np.mean(cis)
            best_model_state = model.state_dict()   # this is a shallow copy, so it's actually not the best model state, but the last model state
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
            pat = 0
        else:
            # patience and early stopping
            pat += 1
            if pat == args.patience:
                break
        # save intermediate model
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'epoch_{epoch}.pth'))
    return best_model_state



def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    dataset = SurvivalDataset(args.data_path, args.separate, args.sofa_derivative_coeff)
    train_size = int(len(dataset) * args.train_ratio)
    valid_size = int(len(dataset) * args.valid_ratio)
    if args.debug:
        train_size = valid_size = 100
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    
    args.times = get_times(dataset)
    
    # load model
    model = NCDESurv(
        input_channels=dataset[0][0].shape[1],
        hidden_channels=args.hidden_channels,
        k=args.k,
        dist=args.dist,
        temp=args.temp,
        layer_num=args.layer_num,
        nonlinear=args.nonlinear,
        pairwise=args.pairwise,
        silhouette=args.silhouette,
        num_clusters=args.num_clusters,
        silhouette_weight=args.silhouette_weight,
        contrastive=args.contrastive,
        contrastive_weight=args.contrastive_weight,
        ae=args.ae,
        weight_decay_temp=args.weight_decay_temp,
        no_ffn=args.no_ffn
    )
    if args.existed_ckpt is not None:
        model.load_state_dict(torch.load(args.existed_ckpt))
        print(f'Loaded model from {args.existed_ckpt}')
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # wandb init
    wandb.init(project='ncde_survival', config=args)
    wandb.watch(model)
    if args.head == 'dsm':
        single_dsm_head = DSMHead(
            input_dim=1,
            k=1,
            dist=args.dist,
            temp=args.temp
        )
        
        # pretrain
        shape, scale = pretrain(single_dsm_head, train_dataloader, valid_dataloader, args)
        model.head.shape.data.fill_(float(shape))
        model.head.scale.data.fill_(float(scale))
    
    # train
    best_model_state = train(model, optimizer, train_dataloader, valid_dataloader, args)
    # save best model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'final_model.pth'))
    # use the best model to evaluate on test set
    model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pth')))
    model.eval()
    pred_surv = predict_surv(model, test_dataloader, args.times, args)
    cis, brs, roc_auc = evaluate(train_dataloader, test_dataloader, pred_surv, args.times)
    wandb.log({'test_cis': np.mean(cis)})
    wandb.log({'test_brs': np.mean(brs)})
    wandb.log({'test_roc_auc': np.mean([roc_auc[i][0] for i in range(len(args.times))])})
    print('Concordance Index:', np.mean(cis))
    print('Brier Score:', np.mean(brs))
    print('ROC AUC:', np.mean([roc_auc[i][0] for i in range(len(args.times))]))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data CSV')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--dist', type=str, default='Weibull')
    parser.add_argument('--temp', type=float, default=1000.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--head', type=str, default='dsm')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--nonlinear', action='store_true', default=False)
    parser.add_argument('--pairwise', action='store_true', default=False)
    parser.add_argument('--silhouette', action='store_true', default=False)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--silhouette_weight', type=float, default=0.01)
    parser.add_argument('--contrastive', action='store_true', default=False)
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    parser.add_argument('--existed_ckpt', type=str, default=None)
    parser.add_argument('--ae', action='store_true', default=False)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--time_decay', action='store_true', default=False)
    parser.add_argument('--weight_decay_temp', type=float, default=10.0)
    parser.add_argument('--no_ffn', action='store_true', default=False, help='no feedforward network for contrastive learning')
    parser.add_argument('--sofa_derivative_coeff', type=float, default=10.0)
    
    return parser.parse_args()
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
