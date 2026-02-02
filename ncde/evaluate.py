from dataset import SurvivalDataset, my_collate_fn
from model import NCDESurv
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import pickle
import torchcde
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from sksurv.linear_model import BreslowEstimator


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='wandb/output')
    parser.add_argument('--state_dict', type=str, default='best_model')
    parser.add_argument('--use_abs_surv', action='store_true', default=False)
    args = parser.parse_args()
    # use json to load args.output_dir/args.json
    with open(f'{args.output_dir}/args.json', 'r') as f:
        args_dict = json.load(f)
    args.__dict__.update(args_dict)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if nonlinear not in args_dict, set it to False
    if 'nonlinear' not in args_dict:
        args.nonlinear = False
    if 'contrastive' not in args_dict:
        args.contrastive = False
    if 'ae' not in args_dict:
        args.ae = False
    if 'no_ffn' not in args_dict:
        args.no_ffn = False
    return args

def load_model(dataset, args):
    model = NCDESurv(
        input_channels=dataset[0][0].shape[1],
        hidden_channels=args.hidden_channels,
        k=args.k,
        dist=args.dist,
        temp=args.temp,
        layer_num=args.layer_num,
        nonlinear=args.nonlinear,
        contrastive=args.contrastive,
        ae=args.ae,
        no_ffn=args.no_ffn
    )
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.state_dict}.pth'))
    model.eval()
    print('Model loaded from best_model.pth')
    return model

def predict_surv(model, test_dataloader, times, args, relative=True):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        times = [torch.tensor([t]).to(args.device) for t in times]
        all_surv = []
        for x, _, e, _, _ in tqdm(test_dataloader, desc='Testing'):
            x, e = x.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            batch_surv = []
            for t in times:
                # broadcast t to match the batch size
                t = t.repeat(x.shape[0], 1).reshape(-1)
                pred_surv = model.predict_surv(coeffs, t, relative)
                batch_surv.append(pred_surv)    # (batch, )
            # (batch, times)
            batch_surv = torch.stack(batch_surv, dim=1)
            all_surv.append(batch_surv)
        all_surv = torch.cat(all_surv, dim=0).detach().cpu().numpy()
    return all_surv

def prepare_baseline_hazard(model, train_dataloader, args):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_risk = []
        all_t = []
        all_e = []
        for x, t, e, _, _ in tqdm(train_dataloader, desc='Preparing baseline hazard'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            features = model(coeffs)
            risk = model.head(features)
            all_risk.append(risk)
            all_t.append(t)
            all_e.append(e)
        all_risk = torch.cat(all_risk, dim=0)
        all_t = torch.cat(all_t, dim=0)
        all_e = torch.cat(all_e, dim=0)
    # save risk, t, e to npy
    np.save(f'{args.output_dir}/risk.npy', all_risk.detach().cpu().numpy())
    np.save(f'{args.output_dir}/t.npy', all_t.detach().cpu().numpy())
    np.save(f'{args.output_dir}/e.npy', all_e.detach().cpu().numpy())
    
    return all_risk.detach().cpu().numpy(), all_t.detach().cpu().numpy(), all_e.detach().cpu().numpy()

def fit_baseline_hazard(all_risk, all_t, all_e, args):
    baseline_model = BreslowEstimator()
    baseline_model.fit(all_risk, all_e, all_t)
    # save with pickle
    with open(f'{args.output_dir}/baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline_model, f)
    return baseline_model

def predict_abs_surv(test_dataloader, model, baseline_model, times, args):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_risk = []
        all_t = []
        all_e = []
        for x, t, e, _, _ in tqdm(test_dataloader, desc='Predict survival function'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            features = model(coeffs)
            risk = model.head(features)
            all_risk.append(risk)
            all_t.append(t)
            all_e.append(e)
        all_risk = torch.cat(all_risk, dim=0)
        all_t = torch.cat(all_t, dim=0)
        all_e = torch.cat(all_e, dim=0)
    all_risk = all_risk.detach().cpu().numpy()
    all_t = all_t.detach().cpu().numpy()
    all_e = all_e.detach().cpu().numpy()
    # predict survival function
    pred_surv_func_list = baseline_model.get_survival_function(all_risk)
    pred_surv = np.zeros((len(all_t), len(times)))
    for i, fn in enumerate(pred_surv_func_list):
        for j, t in enumerate(times):
            pred_surv[i, j] = fn(t)
    return pred_surv
    
        

def evaluate(train_dataloader, test_dataloader, pred_surv, times):
    # get all et for train and test data
    et_train = []
    for x, t, e, _, _ in train_dataloader:
        # concat e and t on dim1
        et_train.append(torch.stack([e, t], dim=1))
    et_train = torch.cat(et_train, dim=0).numpy()
    et_train = np.array([(e, t) for e, t in et_train], dtype=[('e', bool), ('t', float)])
    et_test = []
    for x, t, e, _, _ in test_dataloader:
        et_test.append(torch.stack([e, t], dim=1))
    et_test = torch.cat(et_test, dim=0).numpy()
    et_test = np.array([(e, t) for e, t in et_test], dtype=[('e', bool), ('t', float)])
    
    
    cis = []
    for i, _ in enumerate(times):
        cis.append(concordance_index_ipcw(et_train, et_test, 1-pred_surv[:, i], times[i])[0])
    try:
        brs = brier_score(et_train, et_test, pred_surv, times)[1]
    except:
        brs = [0] * len(times)
    roc_auc = []
    for i, _ in enumerate(times):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test, 1-pred_surv[:, i], times[i])[0])
    return cis, brs, roc_auc    


def get_times(dataset):
    quantile = [.25, .5, .75]
    # get all time
    all_time = []
    for _, t, _, _, _ in dataset:
        all_time.append(t)
    all_time = np.array(all_time)
    # get time quantile
    times = np.quantile(all_time, quantile).tolist()
    return times


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    print('Loading data...')
    dataset = SurvivalDataset(args.data_path, args.separate, args.sofa_derivative_coeff)
    train_size = int(len(dataset) * args.train_ratio)
    valid_size = int(len(dataset) * args.valid_ratio)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    # get times
    print('Getting times...')
    times = get_times(dataset)
    
    # load model
    print('Loading model...')
    model = load_model(dataset, args)
    
    # predict survival
    print('Predicting survival...')
    
    if args.use_abs_surv:
        if os.path.exists(f'{args.output_dir}/risk.npy'):
            all_risk = np.load(f'{args.output_dir}/risk.npy')
            all_t = np.load(f'{args.output_dir}/t.npy')
            all_e = np.load(f'{args.output_dir}/e.npy')
            print('Baseline hazard loaded from risk.npy, t.npy, e.npy')
        else:
            print('Preparing baseline hazard...')
            all_risk, all_t, all_e = prepare_baseline_hazard(model, train_dataloader, args)
        if os.path.exists(f'{args.output_dir}/baseline_model.pkl'):
            with open(f'{args.output_dir}/baseline_model.pkl', 'rb') as f:
                baseline_model = pickle.load(f)
            print('Baseline model loaded from baseline_model.pkl')
        else:
            # fit baseline hazard
            print('Fitting baseline hazard...')
            baseline_model = fit_baseline_hazard(all_risk, all_t, all_e, args)
        # predict survival
        print('Predicting survival...')
        pred_surv = predict_abs_surv(test_dataloader, model, baseline_model, times, args)
    else:
        pred_surv = predict_surv(model, test_dataloader, times, args)
    # evaluate
    print('Evaluating...')
    cis, brs, roc_auc = evaluate(train_dataloader, test_dataloader, pred_surv, times)
    # print results
    print('Concordance Index:', np.mean(cis))
    print('Brier Score:', np.mean(brs))
    print('ROC AUC:', np.mean([roc_auc[i][0] for i in range(len(times))]))
    
    # write the results to a csv file
    c_index = np.mean(cis)
    brier_score_val = np.mean(brs)
    roc_auc_val = np.mean([roc_auc[i][0] for i in range(len(times))])
    result = pd.DataFrame({'Concordance Index': [c_index], 'Brier Score': [brier_score_val], 'ROC AUC': [roc_auc_val]})
    result.to_csv(f'{args.output_dir}/result.csv', index=False)
    
    
    # prepare data for calibration plot
    # save pred_surv and corresponding quantile and corresponding e, t
    pred_surv_df = pd.DataFrame(pred_surv, columns=[f'quantile_{i}' for i in range(len(times))])
    # get all t, e from test_dataloader
    test_t = []
    test_e = []
    for _, t, e, _, _ in test_dataloader:
        test_t.append(t)
        test_e.append(e)
    all_t = torch.cat(test_t, dim=0).reshape(-1).numpy().tolist()
    all_e = torch.cat(test_e, dim=0).reshape(-1).numpy().tolist()
    
    pred_surv_df['e'] = all_e
    pred_surv_df['t'] = all_t
    pred_surv_df.to_csv(f'{args.output_dir}/pred_surv_for_calibration.csv', index=False)

if __name__ == '__main__':
    args = load_arguments()
    main(args)
