from model import NCDESurv
import torch
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, trange
import torchcde
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.nonparametric import kaplan_meier_estimator
import umap
import pickle
import srsly
from scipy.stats import f_oneway, kruskal, chi2_contingency
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline, interp1d
from sklearn.decomposition import PCA
import os
from matplotlib.collections import LineCollection
import faiss
import dcor

from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import dtw_barycenter_averaging
from statsmodels.graphics.mosaicplot import mosaic
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

class AnalysisSurvivalDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.data_standardization()
        self.split_by_id()
    
    def data_standardization(self):
        scaler = StandardScaler()
        feature = self.data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        self.data[feature.columns] = scaler.fit_transform(feature)

        
    
    def split_by_id(self):
        print("Processing by id...")
        self.id_list = self.data['HADM_ID'].unique()
        self.id2data = {}
        for id in tqdm(self.id_list):
            self.id2data[id] = self.data[self.data['HADM_ID'] == id].sort_values(by='interval_id')
    
    def __getitem__(self, index):
        id = self.id_list[index]
        data = self.id2data[id]
        t = data['event_in_hours'].values[0] - data['interval_id'].values[-1]
        time_steps = data['interval_id'].values
        e = data['event_indicator'].values[-1]
        data = data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        data_matrix = data.values
        mask = (~np.isnan(data_matrix)).cumsum(axis=0)
        data_matrix = np.concatenate([data_matrix, mask], axis=1)
        data_matrix = np.concatenate([time_steps.reshape(-1, 1), data_matrix], axis=1)
        return data_matrix, t, e, id
    
    def __len__(self):
        return len(self.id_list)




class SingleSurvivalDataset(Dataset):
    def __init__(self, data_path, hadm_id):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.data_standardization()
        self.data = self.data[self.data['HADM_ID'] == hadm_id].sort_values(by='interval_id')
        self.hadm_id = hadm_id
    
    def data_standardization(self):
        scaler = StandardScaler()
        feature = self.data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        self.data[feature.columns] = scaler.fit_transform(feature)

    
    def __getitem__(self, index):
        data = self.data
        t = data['event_in_hours'].values[0] - data['interval_id'].values[-1]
        time_steps = data['interval_id'].values
        e = data['event_indicator'].values[-1]
        data = data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        data_matrix = data.values
        mask = (~np.isnan(data_matrix)).cumsum(axis=0)
        data_matrix = np.concatenate([data_matrix, mask], axis=1)
        data_matrix = np.concatenate([time_steps.reshape(-1, 1), data_matrix], axis=1)
        return data_matrix, t, e, self.hadm_id
    
    def __len__(self):
        return 1


def fill_forward(x_list):
    max_length = max([x.size(0) for x in x_list])
    x_list = [torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))]) for x in x_list]
    return torch.stack(x_list)

def my_collate_fn(batch):
    x_list, t_list, e_list, id_list = zip(*batch)
    x_list = [torch.FloatTensor(x) for x in x_list]
    x = fill_forward(x_list)
    t = torch.FloatTensor(t_list)
    e = torch.FloatTensor(e_list)
    return x, t, e, id_list



def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='wandb/output')
    parser.add_argument('--state_dict', type=str, default='best_model')
    parser.add_argument('--existed_data', action='store_true', default=False)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--which_time', type=str, default='last', choices=['last', 'first', 'all', 'trajectory', 'alltrajectory', 'half', 'allcorr'])
    parser.add_argument('--hadm_id', type=int, default=109255)
    parser.add_argument('--sofa_num', type=int, default=-1)
    parser.add_argument('--calculate_corr', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data CSV')
    parser.add_argument('--sofa_path', type=str, required=True, help='Path to SOFA scores CSV')
    args = parser.parse_args()
    # use json to load args.output_dir/args.json
    with open(f'{args.output_dir}/args.json', 'r') as f:
        args_dict = json.load(f)
    # Keep user-provided paths
    user_data_path = args.data_path
    user_sofa_path = args.sofa_path
    args.__dict__.update(args_dict)
    args.data_path = user_data_path
    args.sofa_path = user_sofa_path
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if 'nonlinear' not in args_dict:
        args.nonlinear = False
    if 'contrastive' not in args_dict:
        args.contrastive = False
    if 'no_ffn' not in args_dict:
        args.no_ffn = False
    return args

def load_sofa(sofa_path, num=-1):
    sofa_dat = pd.read_csv(sofa_path)
    id2sofa = defaultdict(list)
    hadm_id_list = sofa_dat['HADM_ID'].unique()
    for hadm_id in tqdm(hadm_id_list):
        sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
        sofa = sofa.sort_values(by='interval_id')
        if num == -1:
            id2sofa[hadm_id].append(sofa['sofa'].values[:36])
        elif num in [1, 2, 3, 4, 5, 6]:
            id2sofa[hadm_id].append(sofa[f'sofa{num}'].values[:36])
        else:
            print("invalid sofa number, use overall sofa instead")
            id2sofa[hadm_id].append(sofa['sofa'].values[:36])
        
        time_points = sofa['interval_id'].values
        if num == -1:
            sofa_values = sofa['sofa'].values
        elif num in [1, 2, 3, 4, 5, 6]:
            sofa_values = sofa[f'sofa{num}'].values
        else:
            print("invalid sofa number, use overall sofa instead")
            sofa_values = sofa['sofa'].values
        
        k = 4
        time_points1 = time_points[::k]
        sofa_values1 = [max(sofa_values[i:i+k]) for i in range(0, len(sofa_values), k)]
        sofa_derivative = np.diff(sofa_values1) / k
        sofa_derivative = np.repeat(sofa_derivative, k)
        sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], len(time_points) - len(sofa_derivative)))
        id2sofa[hadm_id].append(sofa_derivative[:36])
        
    return id2sofa
    
    

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
        no_ffn=args.no_ffn
    )
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.state_dict}.pth'))
    model.eval()
    print('Model loaded from best_model.pth')
    return model

def get_single_latent_state(model, test_dataloader, args):
    assert args.which_time in ['last', 'first'] 
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_latent = []
        all_t = []
        all_e = []
        all_id = []
        for x, t, e, id in tqdm(test_dataloader, desc='Testing'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            latent = model.get_latent_state(coeffs, args.which_time)
            if model.contrastive:
                latent = model.contrastive_head(latent)
            all_latent.append(latent)
            all_t.append(t)
            all_e.append(e)
            all_id.extend(id)
        all_latent = torch.cat(all_latent, dim=0).detach().cpu().numpy()
        all_t = torch.cat(all_t, dim=0).detach().cpu().numpy()
        all_e = torch.cat(all_e, dim=0).detach().cpu().numpy()
    return all_latent, all_t, all_e, all_id

def get_all_latent_state(model, test_dataloader, args):
    assert args.which_time == 'all'
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_latent = []
        all_t = []
        all_e = []
        all_id = []
        all_interval_id = []
        for x, t, e, id in tqdm(test_dataloader, desc='Testing'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            latent = model.get_latent_state(coeffs, 'all')
            if model.contrastive:
                latent = model.contrastive_head(latent)
            intervals = x[:, :, 0]
            processed_latent = []
            processed_t = []
            processed_e = []
            processed_id = []
            processed_interval_id = []
            for i in range(latent.size(0)):
                temp_time = -1
                for j in range(latent.size(1)):
                    if intervals[i, j] != temp_time:
                        processed_latent.append(latent[i, j])
                        processed_t.append(t[i])
                        processed_e.append(e[i])
                        processed_id.append(id[i])
                        processed_interval_id.append(intervals[i, j].item())
                        temp_time = intervals[i, j]
            latent = torch.stack(processed_latent)
            t = torch.stack(processed_t)
            e = torch.stack(processed_e)
            id = processed_id
            intervals = processed_interval_id
            all_latent.append(latent)
            all_t.append(t)
            all_e.append(e)
            all_id.extend(id)
            all_interval_id.extend(intervals)
        all_latent = torch.cat(all_latent, dim=0).detach().cpu().numpy()
        all_t = torch.cat(all_t, dim=0).detach().cpu().numpy()
        all_e = torch.cat(all_e, dim=0).detach().cpu().numpy()
    return all_latent, all_t, all_e, all_id, all_interval_id


def get_single_survival_distribution(model, test_dataloader, baseline_model, args):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_risk = []
        all_t = []
        all_e = []
        for x, t, e, id in test_dataloader:
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            features = model(coeffs)
            risk = model.head(features)
            all_risk.append(risk)
            all_t.append(t)
            all_e.append(e)
        all_risk = torch.tensor(all_risk).reshape(-1, 1)
        all_t = torch.tensor(all_t).squeeze().reshape(-1, 1)
        all_e = torch.tensor(all_e).squeeze().reshape(-1, 1)
    all_risk = all_risk.detach().cpu().numpy()
    all_t = all_t.detach().cpu().numpy()
    all_e = all_e.detach().cpu().numpy()
    pred_surv_func_list = baseline_model.get_survival_function(all_risk)
    fn = pred_surv_func_list[0]
    times = np.arange(0, 500, 1)
    pred_surv = [fn(t) for t in times]
    df = pd.DataFrame({'time': times, 'prob': pred_surv})
    return df

def get_single_trajectory(model, test_dataloader, args):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_latent = []
        all_t = []
        all_e = []
        all_id = []
        for x, t, e, id in test_dataloader:
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            time_points = x[0, :, 0].squeeze().detach().cpu().numpy().tolist()
            output_indices = []
            for i in range(int(time_points[-1])+1):
                if i in time_points:
                    output_indices.append(time_points.index(i))
                else:
                    for j in range(len(time_points)-1):
                        if time_points[j] < i < time_points[j+1]:
                            output_indices.append(j + (i - time_points[j]) / (time_points[j+1] - time_points[j]))
                            break
            output_indices = torch.FloatTensor(output_indices).to(x.device)
            latent = model.get_one_latent_trajectory(coeffs, output_indices)
            if model.contrastive:
                latent = model.contrastive_head(latent)
            all_latent.append(latent)
            all_t.append(t)
            all_e.append(e)
            all_id.extend(id)
        all_latent = torch.cat(all_latent, dim=0).detach().cpu().numpy()
        all_t = torch.cat(all_t, dim=0).detach().cpu().numpy()
        all_e = torch.cat(all_e, dim=0).detach().cpu().numpy()
    return all_latent, all_t, all_e, all_id


def get_all_trajectory(model, test_dataloader, args):
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_latent = []
        all_t = []
        all_e = []
        all_id = []
        all_interval_id = []
        for x, t, e, id in tqdm(test_dataloader, desc='Testing'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            time_points = x[:, :, 0].squeeze().detach().cpu().numpy()
            max_time = int(time_points.max())
            output_indices = []
            for b in range(x.size(0)):
                output_indices_b = []
                for i in range(max_time+1):
                    if i in time_points:
                        output_indices.append(time_points.index(i))
                    else:
                        for j in range(len(time_points)-1):
                            if time_points[j] < i < time_points[j+1]:
                                output_indices.append(j + (i - time_points[j]) / (time_points[j+1] - time_points[j]))
                                break
            output_indices = torch.FloatTensor(output_indices).to(x.device)
            latent = model.get_all_latent_trajectory(coeffs, output_indices)
            if model.contrastive:
                latent = model.contrastive_head(latent)
            intervals = x[:, :, 0]
            processed_latent = []
            processed_t = []
            processed_e = []
            processed_id = []
            processed_interval_id = []
            for i in range(latent.size(0)):
                temp_time = -1
                for j in range(latent.size(1)):
                    if intervals[i, j] != temp_time:
                        processed_latent.append(latent[i, j])
                        processed_t.append(t[i])
                        processed_e.append(e[i])
                        processed_id.append(id[i])
                        processed_interval_id.append(intervals[i, j].item())
                        temp_time = intervals[i, j]
            latent = torch.stack(processed_latent)

def cluster_single_latent_state(all_latent, all_t, all_e, all_id, id2sofa, args):
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(all_latent)
    all_cluster = kmeans.labels_
    sil_score = silhouette_score(all_latent, all_cluster)
    print(f'Silhouette score: {sil_score}')
    res = []
    for i in trange(len(all_id)):
        if all_id[i] in id2sofa:
            if args.which_time == 'last':
                sofa = id2sofa[all_id[i]][0][-1]
                sofa_trend = id2sofa[all_id[i]][1][-1]
            elif args.which_time == 'first':
                sofa = id2sofa[all_id[i]][0][0]
                sofa_trend = id2sofa[all_id[i]][1][0]
            else:
                raise ValueError('which_time should be last or first, other time is not supported for now')
            res.append([all_id[i], all_cluster[i], all_t[i], all_e[i], sofa, sofa_trend])
    res = pd.DataFrame(res, columns=['id', 'cluster', 't', 'e', 'sofa', 'sofa_trend'])
    if args.calculate_corr:
        all_sofa = res['sofa'].values
        all_sofa_trend = res['sofa_trend'].values
        all_latent_distance = pdist(all_latent, metric='euclidean')
        all_sofa_distance = pdist(all_sofa.reshape(-1, 1), metric='cityblock')
        all_sofa_trend_distance = pdist(all_sofa_trend.reshape(-1, 1), metric='cityblock')
        corr_latent_sofa = np.corrcoef(all_latent_distance, all_sofa_distance)[0, 1]
        corr_latent_sofa_trend = np.corrcoef(all_latent_distance, all_sofa_trend_distance)[0, 1]
        print(f'Correlation between latent distance and sofa distance: {corr_latent_sofa}')
        print(f'Correlation between latent distance and sofa_trend distance: {corr_latent_sofa_trend}')
    return res

def cluster_all_latent_state(all_latent, all_t, all_e, all_id, all_interval_id, id2sofa, args):
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(all_latent)
    all_cluster = kmeans.labels_
    sil_score = silhouette_score(all_latent, all_cluster)
    print(f'Silhouette score: {sil_score}')
    res = []
    for i in trange(len(all_id)):
        if all_id[i] in id2sofa:
            interval_id = int(all_interval_id[i])
            sofa = id2sofa[all_id[i]][0][interval_id]
            sofa_trend = id2sofa[all_id[i]][1][interval_id]
            res.append([all_id[i], all_cluster[i], all_t[i], all_e[i], sofa, sofa_trend])
    res = pd.DataFrame(res, columns=['id', 'cluster', 't', 'e', 'sofa', 'sofa_trend'])
    if args.calculate_corr:
        all_sofa = res['sofa'].values
        all_sofa_trend = res['sofa_trend'].values
        all_latent_distance = pdist(all_latent, metric='euclidean')
        all_sofa_distance = pdist(all_sofa.reshape(-1, 1), metric='cityblock')
        all_sofa_trend_distance = pdist(all_sofa_trend.reshape(-1, 1), metric='cityblock')
        corr_latent_sofa = np.corrcoef(all_latent_distance, all_sofa_distance)[0, 1]
        corr_latent_sofa_trend = np.corrcoef(all_latent_distance, all_sofa_trend_distance)[0, 1]
        print(f'Correlation between latent distance and sofa distance: {corr_latent_sofa}')
        print(f'Correlation between latent distance and sofa_trend distance: {corr_latent_sofa_trend}')
    return res

def save_all_latent_sofa_info(all_latent, all_id, all_interval_id, args):
    def load_separate_sofa():
        sofa_dat = pd.read_csv(args.sofa_path)
        
        id2sofa = defaultdict(list)
        all_unique_id = list(set(all_id))
        for hadm_id in tqdm(all_unique_id):
            sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
            sofa = sofa.sort_values(by='interval_id')
            if sofa.shape[0] < 40:
                for i in range(40):
                    if i not in sofa['interval_id'].values:
                        sofa = pd.concat([sofa, pd.DataFrame({'interval_id': [i], 'HADM_ID': [hadm_id]})], ignore_index=True)
                sofa = sofa.sort_values(by='interval_id')
                for i in range(1, 7):
                    sofa['sofa' + str(i)] = sofa['sofa' + str(i)].ffill()
            all_sofa_score = []
            all_sofa_trend = []
            for i in range(1, 7):
                sofa_values = sofa['sofa' + str(i)].values
                time_points = sofa['interval_id'].values
                time_points1 = time_points[::4]
                sofa_values1 = [max(sofa_values[i:i+4]) for i in range(0, len(sofa_values), 4)]
                sofa_derivative = np.diff(sofa_values1) / np.diff(time_points1)
                sofa_derivative = np.repeat(sofa_derivative, 4)
                sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], len(time_points) - len(sofa_derivative)))
                all_sofa_score.append(sofa_values[:36])
                all_sofa_trend.append(sofa_derivative[:36])
                assert len(sofa_values[:36]) == 36 and len(sofa_derivative[:36]) == 36
            all_sofa_score = np.array(all_sofa_score).T
            all_sofa_trend = np.array(all_sofa_trend).T
            id2sofa[hadm_id].append(all_sofa_score)
            id2sofa[hadm_id].append(all_sofa_trend)
        return id2sofa
    if os.path.exists(f'{args.output_dir}/umap_fit.pkl'):
        umap_fit = pickle.load(open(f'{args.output_dir}/umap_fit.pkl', 'rb'))
    else:
        umap_fit = umap.UMAP(n_components=2, n_neighbors=60, metric="euclidean", min_dist=.85).fit(all_latent)
        with open(f'{args.output_dir}/umap_fit.pkl', 'wb') as f:
            pickle.dump(umap_fit, f)
    umap_emb = umap_fit.transform(all_latent)
    res_faiss = faiss.StandardGpuResources()

    index = faiss.IndexFlatL2(2)
    gpu_index = faiss.index_cpu_to_gpu(res_faiss, 0, index)

    gpu_index.add(umap_emb)

    k = 50
    D, I = gpu_index.search(umap_emb, k + 1)
    
    res = []
    id2sofa = load_separate_sofa()
    for i in trange(len(all_id)):
        interval_id = int(all_interval_id[i])
        sofa = id2sofa[all_id[i]][0][interval_id]
        sofa_trend = id2sofa[all_id[i]][1][interval_id]
        temp_res = [all_id[i], umap_emb[i, 0], umap_emb[i, 1], interval_id]
        temp_res.extend(sofa)
        temp_res.extend(sofa_trend)
        for j in range(6):
            temp_sofa = np.mean([id2sofa[all_id[idx]][0][int(all_interval_id[idx])][j] for idx in I[i][1:]])
            temp_res.append(temp_sofa)
        for j in range(6):
            temp_sofa_trend = np.mean([id2sofa[all_id[idx]][1][int(all_interval_id[idx])][j] for idx in I[i][1:]])
            temp_res.append(temp_sofa_trend)
        res.append(temp_res)
    res = pd.DataFrame(res, columns=['id', 'umap_x', 'umap_y', 'interval_id', 'sofa1', 'sofa2', 'sofa3', 'sofa4', 'sofa5', 'sofa6', 'sofa_trend1', 'sofa_trend2', 'sofa_trend3', 'sofa_trend4', 'sofa_trend5', 'sofa_trend6', 'avg_sofa1', 'avg_sofa2', 'avg_sofa3', 'avg_sofa4', 'avg_sofa5', 'avg_sofa6', 'avg_sofa_trend1', 'avg_sofa_trend2', 'avg_sofa_trend3', 'avg_sofa_trend4', 'avg_sofa_trend5', 'avg_sofa_trend6'])
    
    res.to_csv(f'{args.output_dir}/umap_sofa_info.csv', index=False)
    return None
    
    

def plot_cluster(res, all_latent, args):
    if os.path.exists(f'{args.output_dir}/umap_fit.pkl'):
        umap_fit = pickle.load(open(f'{args.output_dir}/umap_fit.pkl', 'rb'))
    else:
        umap_fit = umap.UMAP(n_components=2, n_neighbors=40, metric="euclidean", min_dist=.8).fit(all_latent)
        with open(f'{args.output_dir}/umap_fit.pkl', 'wb') as f:
            pickle.dump(umap_fit, f)
    umap_emb = umap_fit.transform(all_latent)

    res['umap_x'] = umap_emb[:, 0]
    res['umap_y'] = umap_emb[:, 1]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=res, x='umap_x', y='umap_y', hue='sofa', palette='viridis', alpha=0.5)
    plt.savefig(f'{args.output_dir}/umap_sofa_{args.which_time}_{args.sofa_num}.png')
    plt.close()
    res1 = res[(res['sofa_trend'] != 0)]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=res1, x='umap_x', y='umap_y', hue='sofa_trend', palette='viridis', alpha=0.5)
    plt.savefig(f'{args.output_dir}/umap_sofa_trend_{args.which_time}_{args.sofa_num}.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=res, x='umap_x', y='umap_y', hue='cluster', palette=sns.color_palette("tab10"), alpha=0.5)
    plt.savefig(f'{args.output_dir}/umap_cluster_{args.which_time}_{args.sofa_num}.png')
    plt.close()
    
    res.to_csv(f'{args.output_dir}/umap_sofa_cluster_{args.which_time}_{args.sofa_num}.csv', index=False)



def plot_centroids(res, all_latent, centroids, args):
    if os.path.exists(f'{args.output_dir}/umap_fit.pkl'):
        umap_fit = pickle.load(open(f'{args.output_dir}/umap_fit.pkl', 'rb'))
    else:
        umap_fit = umap.UMAP(n_components=2, n_neighbors=40, metric="euclidean", min_dist=.8).fit(all_latent)
        with open(f'{args.output_dir}/umap_fit.pkl', 'wb') as f:
            pickle.dump(umap_fit, f)
    umap_emb = umap_fit.transform(all_latent)
    res['umap_x'] = umap_emb[:, 0]
    res['umap_y'] = umap_emb[:, 1]
    centroids = centroids.reshape(-1, centroids.shape[2])
    umap_emb_centroids = umap_fit.transform(centroids)
    n = umap_emb_centroids.shape[0] / args.n_clusters
    res_centroids = pd.DataFrame(umap_emb_centroids, columns=['umap_x', 'umap_y'])
    res_centroids['cluster'] = [int(i // n) for i in range(umap_emb_centroids.shape[0])]

    res_centroids.to_csv(f'{args.output_dir}/umap_trajectory_centroids_{args.n_clusters}.csv', index=False)
    
    norm = plt.Normalize(res['sofa'].min(), res['sofa'].max())
    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.figure(figsize=(10, 8))

    sns.scatterplot(data=res, x='umap_x', y='umap_y', hue='sofa', palette=cmap, alpha=0.1, s=20, legend=True, hue_norm=norm)
    
    sns.scatterplot(data=res_centroids, x='umap_x', y='umap_y', hue='cluster', palette='tab10', alpha=1.0, s=40, legend=False)
    
    for cluster in range(args.n_clusters):
        centroid = res_centroids[res_centroids['cluster'] == cluster]
        points = np.array([centroid['umap_x'], centroid['umap_y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='tab10', alpha=0.5)
        lc.set_linewidth(2)
        plt.gca().add_collection(lc)
        
    
    plt.savefig(f'{args.output_dir}/umap_sofa_centroids_{args.which_time}_{args.sofa_num}_{args.n_clusters}.png')
    plt.close()
    return


def plot_cluster_with_single_trajectory(res, all_latent, single_trajectory, res_single, args):
    if os.path.exists(f'{args.output_dir}/umap_fit.pkl'):
        umap_fit = pickle.load(open(f'{args.output_dir}/umap_fit.pkl', 'rb'))
    else:
        umap_fit = umap.UMAP(n_components=2, n_neighbors=40, metric="euclidean", min_dist=.8).fit(all_latent)
        with open(f'{args.output_dir}/umap_fit.pkl', 'wb') as f:
            pickle.dump(umap_fit, f)
    umap_emb = umap_fit.transform(all_latent)
    res['umap_x'] = umap_emb[:, 0]
    res['umap_y'] = umap_emb[:, 1]
    umap_emb_single = umap_fit.transform(single_trajectory)
    res_single['umap_x'] = umap_emb_single[:, 0]
    res_single['umap_y'] = umap_emb_single[:, 1]
    
    res_single.to_csv(f'{args.output_dir}/umap_trajectory_single_{args.hadm_id}.csv', index=False)

    norm = plt.Normalize(res['sofa'].min(), res['sofa'].max())
    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.figure(figsize=(10, 8))

    sns.scatterplot(data=res, x='umap_x', y='umap_y', hue='sofa', palette=cmap, alpha=0.1, s=20, legend=True, hue_norm=norm)

    sns.scatterplot(data=res_single, x='umap_x', y='umap_y', hue='sofa', palette=cmap, alpha=1.0, s=40, hue_norm=norm, legend=False)

    colors = res_single['sofa'].values
    points = np.array([res_single['umap_x'], res_single['umap_y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.5)
    lc.set_array(colors)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.savefig(f'{args.output_dir}/umap_sofa_{args.which_time}_{args.hadm_id}.png')
    plt.close()
    norm = plt.Normalize(res['sofa_trend'].min(), res['sofa_trend'].max())
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(data=res, x='umap_x', y='umap_y', hue='sofa_trend', palette=cmap, alpha=0.1, s=20, legend=True, hue_norm=norm)
    
    sns.scatterplot(data=res_single, x='umap_x', y='umap_y', hue='sofa_trend', palette=cmap, alpha=1.0, s=40, hue_norm=norm, legend=False)
    
    colors = res_single['sofa_trend'].values
    points = np.array([res_single['umap_x'], res_single['umap_y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.5)
    lc.set_array(colors)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    plt.savefig(f'{args.output_dir}/umap_sofa_trend_{args.which_time}_{args.hadm_id}.png')
    plt.close()
    return

def statistical_analysis(res, args):
    cluster2sofa = []
    for cluster in range(args.n_clusters):
        cluster2sofa.append(res[res['cluster'] == cluster]['sofa'].values)
    f, p = kruskal(*cluster2sofa)
    print("Kruskal-Wallis H-test")
    print(f, p)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=res, x='sofa', hue='cluster', fill=True)
    plt.savefig(f'{args.output_dir}/sofa_density_{args.which_time}.png')
    plt.close()
    
    res1 = res
    cluster2sofa_trend = []
    for cluster in range(args.n_clusters):
        cluster2sofa_trend.append(res1[res1['cluster'] == cluster]['sofa_trend'].values)
    f, p = kruskal(*cluster2sofa_trend)
    print(f, p)
    f, p = f_oneway(*cluster2sofa_trend)
    print("One-way ANOVA test for sofa_trend")
    print(f, p)
    cluster2sofa_trend_freq = []
    for cluster in range(args.n_clusters):
        cluster2sofa_trend_freq.append([np.sum(res1[res1['cluster'] == cluster]['sofa_trend'] == i) for i in np.arange(-1.5, 1.5, 0.25)])
    cluster2sofa_trend_freq = np.array(cluster2sofa_trend_freq)
    print(cluster2sofa_trend_freq)
    cluster2sofa_trend_freq = cluster2sofa_trend_freq[:, np.sum(cluster2sofa_trend_freq, axis=0) != 0]
    result = chi2_contingency(cluster2sofa_trend_freq)
    print(result.pvalue)

    crosstable = pd.crosstab(res1['cluster'], res1['sofa_trend'])
    mosaic(crosstable.stack())
    plt.savefig(f'{args.output_dir}/sofa_trend_mosaic_{args.which_time}.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    for cluster in range(args.n_clusters):
        cluster_data = res1[res1['cluster'] == cluster]
        sofa_mean = cluster_data['sofa'].mean()
        sofa_std = cluster_data['sofa'].std()
        sofa_trend_mean = cluster_data['sofa_trend'].mean()
        sofa_trend_std = cluster_data['sofa_trend'].std()
        plt.errorbar(sofa_mean, sofa_trend_mean, xerr=sofa_std, yerr=sofa_trend_std, fmt='o', label=f'cluster {cluster}')
    plt.legend()
    plt.savefig(f'{args.output_dir}/sofa_sofa_trend_{args.which_time}.png')    
    plt.close()

def survival_analysis(res, args):
    if args.which_time != 'last' and args.which_time != 'alltrajectory':
        return
    for cluster in range(args.n_clusters):
        cluster_data = res[res['cluster'] == cluster]
        time, survival_prob = kaplan_meier_estimator(cluster_data['e'].astype(bool), cluster_data['t'])
        plt.step(time, survival_prob, where="post", label=f'cluster {cluster}')
    plt.legend()
    plt.savefig(f'{args.output_dir}/survival_curve_{args.which_time}_{args.n_clusters}.png')
    plt.close()


def eval_cluster(res, args):
    if args.n_clusters <= 6:
        survival_rate_diff = []
        for time_point in [200, 400, 600, 800]:
            survival_rate_diff.append([])
            for cluster1 in range(args.n_clusters):
                for cluster2 in range(cluster1+1, args.n_clusters):
                    cluster1_data = res[res['cluster'] == cluster1]
                    cluster2_data = res[res['cluster'] == cluster2]
                    time1, survival_prob1 = kaplan_meier_estimator(cluster1_data['e'].astype(bool), cluster1_data['t'])
                    time2, survival_prob2 = kaplan_meier_estimator(cluster2_data['e'].astype(bool), cluster2_data['t'])
                    f1 = interp1d(time1, survival_prob1, kind='linear')
                    f2 = interp1d(time2, survival_prob2, kind='linear')
                    survival_rate_diff[-1].append(np.abs(f2(time_point) - f1(time_point)))

            survival_rate_diff[-1] = np.median(survival_rate_diff[-1])
        print(survival_rate_diff)
    
    if 'sofa' not in res.columns or 'sofa_trend' not in res.columns:
        return
    cluster2sofa = []
    for cluster in range(args.n_clusters):
        cluster2sofa.append(res[res['cluster'] == cluster]['sofa'].values)
    f, p = kruskal(*cluster2sofa)
    print("Kruskal-Wallis H-test for sofa")
    print(f, p)
    
    cluster2sofa_trend = []
    for cluster in range(args.n_clusters):
        cluster2sofa_trend.append(res[res['cluster'] == cluster]['sofa_trend'].values)
    f, p = kruskal(*cluster2sofa_trend)
    print("Kruskal-Wallis H-test for sofa_trend")
    print(f, p)
    
    

def cluster_all_trajectory(all_latent, all_t, all_e, all_id, args):
    max_time_steps = max([x.shape[0] for x in all_latent])
    all_latent = [np.concatenate([x, np.full((max_time_steps - x.shape[0], x.shape[1]), np.nan)]) for x in all_latent]
    all_latent = np.stack(all_latent)
    print(all_latent.shape)
    model = TimeSeriesKMeans(n_clusters=args.n_clusters, metric="dtw", max_iter=100, random_state=0).fit(all_latent)
    centroids = model.cluster_centers_
    label = model.labels_
    print(centroids.shape)
    print(label.shape)
    
    np.save(f'{args.output_dir}/all_trajectory_centroids_{args.n_clusters}.npy', centroids)
    
    res = []
    for i in trange(len(all_id)):
        res.append([all_id[i], label[i], all_t[i][0], all_e[i][0]])
    
    res = pd.DataFrame(res, columns=['id', 'cluster', 't', 'e'])
    res.to_csv(f'{args.output_dir}/cluster_result_alltrajectory_{args.n_clusters}.csv', index=False)
    return res, centroids
    
def analyze_all_trajectory(res, id2sofa, args):
    sofa_centroid_list = []
    sofa_trend_centroid_list = []
    for cluster in range(args.n_clusters):
        cluster_data = res[res['cluster'] == cluster]
        cluster_data = res[res['cluster'] == cluster]
        cluster_id = cluster_data['id'].values
        sofa = []
        sofa_trend = []
        for id in cluster_id:
            sofa.append(id2sofa[id][0])
            sofa_trend.append(id2sofa[id][1])
        sofa = np.stack(sofa)
        sofa_trend = np.stack(sofa_trend)
        sofa_centroid = np.nanmean(sofa, axis=0)
        sofa_trend_centroid = np.nanmean(sofa_trend, axis=0)
        sofa_centroid_list.append(sofa_centroid)
        sofa_trend_centroid_list.append(sofa_trend_centroid)
    sofa_centroid_list = np.stack(sofa_centroid_list)
    sofa_trend_centroid_list = np.stack(sofa_trend_centroid_list)
    plt.figure(figsize=(8, 6))
    for i in range(sofa_centroid_list.shape[0]):
        plt.plot(sofa_centroid_list[i], label=f'cluster {i}')
    plt.legend()
    plt.savefig(f'{args.output_dir}/sofa_centroids_alltrajectory_sofa{args.sofa_num}_{args.n_clusters}.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    for i in range(sofa_trend_centroid_list.shape[0]):
        plt.plot(sofa_trend_centroid_list[i], label=f'cluster {i}')
    plt.legend()
    plt.savefig(f'{args.output_dir}/sofa_trend_centroids_alltrajectory{args.sofa_num}_{args.n_clusters}.png')
    plt.close()

def save_sofa_trajectory(res, args):
    all_id = res['id'].values
    def load_separate_sofa():
        sofa_dat = pd.read_csv(args.sofa_path)
        
        id2sofa = defaultdict(list)
        for hadm_id in tqdm(all_id):
            sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
            sofa = sofa.sort_values(by='interval_id')
            if sofa.shape[0] < 40:
                for i in range(40):
                    if i not in sofa['interval_id'].values:
                        sofa = pd.concat([sofa, pd.DataFrame({'interval_id': [i], 'HADM_ID': [hadm_id]})], ignore_index=True)
                sofa = sofa.sort_values(by='interval_id')
                for i in range(1, 7):
                    sofa['sofa' + str(i)] = sofa['sofa' + str(i)].ffill()
                sofa['sofa'] = sofa['sofa'].ffill()
            all_sofa_score = []
            all_sofa_trend = []
            for i in range(7):
                if i == 0:
                    sofa_values = sofa['sofa'].values
                else:
                    sofa_values = sofa['sofa' + str(i)].values
                time_points = sofa['interval_id'].values
                time_points1 = time_points[::4]
                sofa_values1 = [max(sofa_values[i:i+4]) for i in range(0, len(sofa_values), 4)]
                sofa_derivative = np.diff(sofa_values1) / np.diff(time_points1)
                sofa_derivative = np.repeat(sofa_derivative, 4)
                sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], len(time_points) - len(sofa_derivative)))
                all_sofa_score.append(sofa_values[:36])
                all_sofa_trend.append(sofa_derivative[:36])
                assert len(sofa_values[:36]) == 36 and len(sofa_derivative[:36]) == 36
            all_sofa_score = np.array(all_sofa_score).T
            all_sofa_trend = np.array(all_sofa_trend).T
            id2sofa[hadm_id].append(all_sofa_score)
            id2sofa[hadm_id].append(all_sofa_trend)
        return id2sofa
    id2sofa = load_separate_sofa()
    sofa_centroid_list = []
    sofa_trend_centroid_list = []
    sofa_sd_list = []
    sofa_trend_sd_list = []
    for cluster in range(args.n_clusters):
        cluster_data = res[res['cluster'] == cluster]
        cluster_data = res[res['cluster'] == cluster]
        cluster_id = cluster_data['id'].values
        sofa = []
        sofa_trend = []
        for id in cluster_id:
            sofa.append(id2sofa[id][0])
            sofa_trend.append(id2sofa[id][1])
        sofa = np.stack(sofa)
        sofa_trend = np.stack(sofa_trend)
        sofa_centroid = np.nanmean(sofa, axis=0)
        sofa_trend_centroid = np.nanmean(sofa_trend, axis=0)
        sofa_sd = np.nanstd(sofa, axis=0)
        sofa_trend_sd = np.nanstd(sofa_trend, axis=0)
        sofa_centroid_list.append(sofa_centroid)
        sofa_trend_centroid_list.append(sofa_trend_centroid)
        sofa_sd_list.append(sofa_sd)
        sofa_trend_sd_list.append(sofa_trend_sd)
    sofa_centroid_list = np.stack(sofa_centroid_list)
    sofa_trend_centroid_list = np.stack(sofa_trend_centroid_list)
    sofa_sd_list = np.stack(sofa_sd_list)
    sofa_trend_sd_list = np.stack(sofa_trend_sd_list)
    all_sofa = sofa_centroid_list.reshape(-1, 7)
    all_sofa_trend = sofa_trend_centroid_list.reshape(-1, 7)
    all_sofa_sd = sofa_sd_list.reshape(-1, 7)
    all_sofa_trend_sd = sofa_trend_sd_list.reshape(-1, 7)
    all_cluster = np.repeat(np.arange(args.n_clusters), sofa_centroid_list.shape[1])
    all_time = np.tile(np.arange(sofa_centroid_list.shape[1]), args.n_clusters)
    res = pd.DataFrame({'cluster': all_cluster, 'time': all_time})
    for i in range(7):
        if i == 0:
            res['sofa'] = all_sofa[:, i]
            res['sofa_trend'] = all_sofa_trend[:, i]
            res['sofa_sd'] = all_sofa_sd[:, i]
            res['sofa_trend_sd'] = all_sofa_trend_sd[:, i]
        else:
            res['sofa' + str(i)] = all_sofa[:, i]
            res['sofa_trend' + str(i)] = all_sofa_trend[:, i]
            res['sofa_sd' + str(i)] = all_sofa_sd[:, i]
            res['sofa_trend_sd' + str(i)] = all_sofa_trend_sd[:, i]
    res.to_csv(f'{args.output_dir}/cluster_sofa_trajectory_{args.n_clusters}.csv', index=False)
    return
    

def get_corr_lineplot(all_latent, all_id, args):
    def load_separate_sofa():
        sofa_dat = pd.read_csv(args.sofa_path)
        
        id2sofa = defaultdict(list)
        for hadm_id in tqdm(all_id):
            sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
            sofa = sofa.sort_values(by='interval_id')
            if sofa.shape[0] < 40:
                for i in range(40):
                    if i not in sofa['interval_id'].values:
                        sofa = pd.concat([sofa, pd.DataFrame({'interval_id': [i], 'HADM_ID': [hadm_id]})], ignore_index=True)
                sofa = sofa.sort_values(by='interval_id')
                for i in range(1, 7):
                    sofa['sofa' + str(i)] = sofa['sofa' + str(i)].ffill()
            all_sofa_score = []
            all_sofa_trend = []
            for i in range(1, 7):
                sofa_values = sofa['sofa' + str(i)].values
                time_points = sofa['interval_id'].values
                
                sofa_derivative = sofa_values[4:] - sofa_values[:-4]
                sofa_derivative = np.append(np.repeat(sofa_derivative[0], 2), sofa_derivative)
                sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], 2))
                
                
                all_sofa_score.append(sofa_values[:36])
                all_sofa_trend.append(sofa_derivative[:36])
                assert len(sofa_values[:36]) == 36 and len(sofa_derivative[:36]) == 36
            all_sofa_score = np.array(all_sofa_score).T
            all_sofa_trend = np.array(all_sofa_trend).T
            id2sofa[hadm_id].append(all_sofa_score)
            id2sofa[hadm_id].append(all_sofa_trend)
            all_sofa_comp = np.concatenate([all_sofa_score, 20*all_sofa_trend], axis=1)
            id2sofa[hadm_id].append(all_sofa_comp)
        return id2sofa
    id2sofa = load_separate_sofa()
    sofa_corr = []
    sofa_trend_corr = []
    sofa_comp_corr = []
    for i in trange(36):
        curr_latent, curr_id = zip(*[(latent[i, :], id) for latent, id in zip(all_latent, all_id) if latent.shape[0] > i and id2sofa[id][0].shape[0] > i])
        curr_sofa = [id2sofa[id][0][i] for id in curr_id]
        curr_sofa_trend = [id2sofa[id][1][i] for id in curr_id]
        curr_sofa_comp = [id2sofa[id][2][i] for id in curr_id]
        curr_sofa = np.stack(curr_sofa)
        curr_sofa_trend = np.stack(curr_sofa_trend)
        curr_sofa_comp = np.stack(curr_sofa_comp)
        curr_latent = np.stack(curr_latent)
        all_latent_distance = pdist(curr_latent, metric='euclidean')
        all_sofa_distance = pdist(curr_sofa, metric='cityblock')
        all_sofa_trend_distance = pdist(curr_sofa_trend, metric='cityblock')
        all_sofa_comp_distance = pdist(curr_sofa_comp, metric='cityblock')
        corr_latent_sofa = dcor.distance_correlation(curr_latent, curr_sofa)
        corr_latent_sofa_trend = dcor.distance_correlation(curr_latent, curr_sofa_trend)
        corr_latent_sofa_comp = dcor.distance_correlation(curr_latent, curr_sofa_comp)
        
        sofa_corr.append(corr_latent_sofa)
        sofa_trend_corr.append(corr_latent_sofa_trend)
        sofa_comp_corr.append(corr_latent_sofa_comp)
        print(f'Correlation between latent distance and sofa distance at time {i}: {corr_latent_sofa}')
        print(f'Correlation between latent distance and sofa_trend distance at time {i}: {corr_latent_sofa_trend}')
        print(f'Correlation between latent distance and sofa_comp distance at time {i}: {corr_latent_sofa_comp}')
    plt.figure(figsize=(8, 6))
    plt.plot(sofa_corr, label='sofa')
    plt.plot(sofa_trend_corr, label='sofa_trend')
    plt.legend()
    plt.savefig(f'{args.output_dir}/corr_lineplot_dcor.png')
    
    interval_id = np.arange(36)
    corr_df = pd.DataFrame({'interval_id': interval_id, 'sofa_corr': sofa_corr, 'sofa_trend_corr': sofa_trend_corr, 'sofa_comp_corr': sofa_comp_corr})
    corr_df.to_csv(f'{args.output_dir}/corr_lineplot_dcor.csv', index=False)
    return
        


def get_corr(latent, all_id, args, id2half_index=None):
    def load_last_separate_sofa():
        sofa_dat = pd.read_csv(args.sofa_path)
        ori_dat = pd.read_csv(args.data_path)
        id2sofa = defaultdict(list)
        hadm_id_list = all_id
        for hadm_id in tqdm(hadm_id_list):
            sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
            sofa = sofa.sort_values(by='interval_id')
            ori = ori_dat[ori_dat['HADM_ID'] == hadm_id]
            max_interval_id = int(ori['interval_id'].max())
            if id2half_index is not None:
                half_index = id2half_index[hadm_id]
            all_sofa_score = []
            all_sofa_derivative = []
            for i in range(1, 7):
                sofa_values = sofa['sofa' + str(i)].values
                time_points = sofa['interval_id'].values
                time_points1 = time_points[::4]
                sofa_values1 = [max(sofa_values[i:i+4]) for i in range(0, len(sofa_values), 4)]
                sofa_derivative = np.diff(sofa_values1) / np.diff(time_points1)
                sofa_derivative = np.repeat(sofa_derivative, 4)
                sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], len(time_points) - len(sofa_derivative)))
                
                if args.which_time == 'last':
                    all_sofa_score.append(sofa_values[max_interval_id])
                    all_sofa_derivative.append(sofa_derivative[max_interval_id])
                elif args.which_time == 'first':
                    all_sofa_score.append(sofa_values[0])
                    all_sofa_derivative.append(sofa_derivative[0])
                elif args.which_time == 'half':
                    all_sofa_score.append(sofa_values[half_index])
                    all_sofa_derivative.append(sofa_derivative[half_index])
                else:
                    raise ValueError('which_time should be last, first, or half, other time is not supported for now')
            all_sofa_score = np.array(all_sofa_score)
            all_sofa_derivative = np.array(all_sofa_derivative)
            id2sofa[hadm_id].append(all_sofa_score)
            id2sofa[hadm_id].append(all_sofa_derivative)
        return id2sofa
    id2sofa = load_last_separate_sofa()
    all_sofa = []
    all_sofa_trend = []
    for i in trange(len(all_id)):
        if all_id[i] in id2sofa:
            all_sofa.append(id2sofa[all_id[i]][0])
            all_sofa_trend.append(id2sofa[all_id[i]][1])
    all_sofa = np.stack(all_sofa)
    all_sofa_trend = np.stack(all_sofa_trend)
    all_latent_distance = pdist(latent, metric='euclidean')
    all_sofa_distance = pdist(all_sofa, metric='cityblock')
    all_sofa_trend_distance = pdist(all_sofa_trend, metric='cityblock')
    corr_latent_sofa = np.corrcoef(all_latent_distance, all_sofa_distance)[0, 1]
    corr_latent_sofa_trend = np.corrcoef(all_latent_distance, all_sofa_trend_distance)[0, 1]
    print(f'Correlation between latent distance and sofa distance: {corr_latent_sofa}')
    print(f'Correlation between latent distance and sofa_trend distance: {corr_latent_sofa_trend}')
    
    corr_latent_sofa = spearmanr(all_latent_distance, all_sofa_distance)
    corr_latent_sofa_trend = spearmanr(all_latent_distance, all_sofa_trend_distance)
    print(f'Spearman correlation between latent distance and sofa distance: {corr_latent_sofa}')
    print(f'Spearman correlation between latent distance and sofa_trend distance: {corr_latent_sofa_trend}')
    
    corr = pd.DataFrame({'corr_latent_sofa': [corr_latent_sofa[0]], 'pvalue_latent_sofa': [corr_latent_sofa[1]], 'corr_latent_sofa_trend': [corr_latent_sofa_trend[0]], 'pvalue_latent_sofa_trend': [corr_latent_sofa_trend[1]]})
    corr.to_csv(f'{args.output_dir}/corr_latent_sofa_{args.which_time}.csv', index=False)
    
    if len(all_latent_distance) > 20000:
        idx = np.random.choice(len(all_latent_distance), 20000, replace=False)
        all_latent_distance = all_latent_distance[idx]
        all_sofa_distance = all_sofa_distance[idx]
        all_sofa_trend_distance = all_sofa_trend_distance[idx]
    res = pd.DataFrame({'latent_distance': all_latent_distance, 'sofa_distance': all_sofa_distance, 'sofa_trend_distance': all_sofa_trend_distance})
    res.to_csv(f'{args.output_dir}/latent_sofa_distance.csv', index=False)
    

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.which_time == 'allcorr':
        try:
            with open(f'{args.output_dir}/all_latent_trajectory.pkl', 'rb') as f:
                all_latent = pickle.load(f)
            with open(f'{args.output_dir}/all_t_trajectory.pkl', 'rb') as f:
                all_t = pickle.load(f)
            with open(f'{args.output_dir}/all_e_trajectory.pkl', 'rb') as f:
                all_e = pickle.load(f)
            with open(f'{args.output_dir}/all_id_trajectory.pkl', 'rb') as f:
                all_id = pickle.load(f)
            print("all_latent_trajectory.pkl, all_t_trajectory.pkl, all_e_trajectory.pkl, all_id_trajectory.pkl loaded")
        except:
            print("all_latent_trajectory.pkl, all_t_trajectory.pkl, all_e_trajectory.pkl, all_id_trajectory.pkl not found, run all_trajectory first")
            return
        get_corr_lineplot(all_latent, all_id, args)
        return
    
    if args.which_time == "half":
        try:
            with open(f'{args.output_dir}/all_latent_trajectory.pkl', 'rb') as f:
                all_latent = pickle.load(f)
            with open(f'{args.output_dir}/all_t_trajectory.pkl', 'rb') as f:
                all_t = pickle.load(f)
            with open(f'{args.output_dir}/all_e_trajectory.pkl', 'rb') as f:
                all_e = pickle.load(f)
            with open(f'{args.output_dir}/all_id_trajectory.pkl', 'rb') as f:
                all_id = pickle.load(f)
            print("all_latent_trajectory.pkl, all_t_trajectory.pkl, all_e_trajectory.pkl, all_id_trajectory.pkl loaded")
        except:
            print("all_latent_trajectory.pkl, all_t_trajectory.pkl, all_e_trajectory.pkl, all_id_trajectory.pkl not found, run all_trajectory first")
            return
        id2sofa = load_sofa(args.sofa_path, args.sofa_num)
        id2half_index = {id: int(latent.shape[0]//2) for id, latent in zip(all_id, all_latent)}
        all_half_latent = [latent[int(latent.shape[0]//2)] for latent in all_latent]
        get_corr(all_half_latent, all_id, args, id2half_index)
        return

              
    if args.which_time in ['last', 'first']:
        if not args.existed_data:
            print('Loading data...')
            dataset = AnalysisSurvivalDataset(args.data_path)
            train_size = int(len(dataset) * args.train_ratio)
            valid_size = int(len(dataset) * args.valid_ratio)
            test_size = len(dataset) - train_size - valid_size
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            print('Loading model...')
            model = load_model(dataset, args)
            print('Getting latent state...')
            latent, t, e, all_id = get_single_latent_state(model, test_dataloader, args)
            np.save(f'{args.output_dir}/latent_{args.which_time}.npy', latent)
            np.save(f'{args.output_dir}/t_{args.which_time}.npy', t)
            np.save(f'{args.output_dir}/e_{args.which_time}.npy', e)
            with open(f'{args.output_dir}/all_id_{args.which_time}.pkl', 'wb') as f:
                pickle.dump(all_id, f)
            

        else:
            try:
                latent = np.load(f'{args.output_dir}/latent_{args.which_time}.npy')
                t = np.load(f'{args.output_dir}/t_{args.which_time}.npy')
                e = np.load(f'{args.output_dir}/e_{args.which_time}.npy')
                with open(f'{args.output_dir}/all_id_{args.which_time}.pkl', 'rb') as f:
                    all_id = pickle.load(f)
                    
                print('latent.npy, t.npy, e.npy, all_id.pkl loaded')
            except:
                print('latent.npy, t.npy, e.npy, all_id.pkl not found, please set existed_data to False')
                return
        if args.which_time == 'last':
            get_corr(latent, all_id, args)
        id2sofa = load_sofa(args.sofa_path, args.sofa_num)
        res = cluster_single_latent_state(latent, t, e, all_id, id2sofa, args)
        res.to_csv(f'{args.output_dir}/cluster_result_{args.which_time}.csv', index=False)
        print('cluster_result.csv saved')
        plot_cluster(res, latent, args)
        statistical_analysis(res, args)
        survival_analysis(res, args)
        eval_cluster(res, args)

        print('Analysis finished')
    elif args.which_time == 'all':
        if not args.existed_data:
            print('Loading data...')
            dataset = AnalysisSurvivalDataset(args.data_path)
            train_size = int(len(dataset) * args.train_ratio)
            valid_size = int(len(dataset) * args.valid_ratio)
            test_size = len(dataset) - train_size - valid_size
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            print('Loading model...')
            model = load_model(dataset, args)
            print('Getting latent state...')
            latent, t, e, all_id, all_interval_id = get_all_latent_state(model, test_dataloader, args)
            np.save(f'{args.output_dir}/latent_{args.which_time}.npy', latent)
            np.save(f'{args.output_dir}/t_{args.which_time}.npy', t)
            np.save(f'{args.output_dir}/e_{args.which_time}.npy', e)
            with open(f'{args.output_dir}/all_id_{args.which_time}.pkl', 'wb') as f:
                pickle.dump(all_id, f)
            with open(f'{args.output_dir}/all_interval_id_{args.which_time}.pkl', 'wb') as f:
                pickle.dump(all_interval_id, f)
                
            

        else:
            try:
                latent = np.load(f'{args.output_dir}/latent_{args.which_time}.npy')
                t = np.load(f'{args.output_dir}/t_{args.which_time}.npy')
                e = np.load(f'{args.output_dir}/e_{args.which_time}.npy')
                with open(f'{args.output_dir}/all_id_{args.which_time}.pkl', 'rb') as f:
                    all_id = pickle.load(f)
                with open(f'{args.output_dir}/all_interval_id_{args.which_time}.pkl', 'rb') as f:
                    all_interval_id = pickle.load(f)
                    
                print('latent.npy, t.npy, e.npy, all_id.pkl, all_interval_id.pkl loaded')
            except:
                print('latent.npy, t.npy, e.npy, all_id.pkl, all_interval_id.pkl not found, please set existed_data to False')
                return
        save_all_latent_sofa_info(latent, all_id, all_interval_id, args)
        id2sofa = load_sofa(args.sofa_path, args.sofa_num)
        res = cluster_all_latent_state(latent, t, e, all_id, all_interval_id, id2sofa, args)
        res.to_csv(f'{args.output_dir}/cluster_result_{args.which_time}.csv', index=False)
        print('cluster_result.csv saved')
        plot_cluster(res, latent, args)
        statistical_analysis(res, args)
        survival_analysis(res, args)  
        eval_cluster(res, args)
        print('Analysis finished')    
    elif args.which_time == 'trajectory':
        if not args.existed_data:
            print('Loading data...')
            dataset = SingleSurvivalDataset(args.data_path, args.hadm_id)
            test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
            print('Loading model...')
            model = load_model(dataset, args)
            print('Getting latent state...')
            latent, t, e, all_id = get_single_trajectory(model, test_dataloader, args)
            np.save(f'{args.output_dir}/latent_trajectory_{args.hadm_id}.npy', latent)
            np.save(f'{args.output_dir}/t_trajectory_{args.hadm_id}.npy', t)
            np.save(f'{args.output_dir}/e_trajectory_{args.hadm_id}.npy', e)
            with open(f'{args.output_dir}/all_id_trajectory_{args.hadm_id}.pkl', 'wb') as f:
                pickle.dump(all_id, f)            
            

        else:
            try:
                latent = np.load(f'{args.output_dir}/latent_trajectory_{args.hadm_id}.npy')
                t = np.load(f'{args.output_dir}/t_trajectory_{args.hadm_id}.npy')
                e = np.load(f'{args.output_dir}/e_trajectory_{args.hadm_id}.npy')
                with open(f'{args.output_dir}/all_id_trajectory_{args.hadm_id}.pkl', 'rb') as f:
                    all_id = pickle.load(f)

                print('latent_trajectory.npy, t_trajectory.npy, e_trajectory.npy, all_id_trajectory.pkl loaded')
            except:
                print('latent_trajectory.npy, t_trajectory.npy, e_trajectory.npy, all_id_trajectory.pkl not found, please set existed_data to False')
                return
        if not os.path.exists(f'{args.output_dir}/baseline_model.pkl'):
            print('Baseline model not found, please run absolute evaluation first')
            return
        baseline_model = pickle.load(open(f'{args.output_dir}/baseline_model.pkl', 'rb'))
        pred_surv_df = get_single_survival_distribution(model, test_dataloader, baseline_model, args)
        pred_surv_df.to_csv(f'{args.output_dir}/pred_surv_{args.hadm_id}.csv', index=False)
        
        all_latent = np.load(f'{args.output_dir}/latent_all.npy')
        all_t = np.load(f'{args.output_dir}/t_all.npy')
        all_e = np.load(f'{args.output_dir}/e_all.npy')
        with open(f'{args.output_dir}/all_id_all.pkl', 'rb') as f:
            all_id = pickle.load(f)
        with open(f'{args.output_dir}/all_interval_id_all.pkl', 'rb') as f:
            all_interval_id = pickle.load(f)
        id2sofa = load_sofa(args.sofa_path, args.sofa_num)
        res_all = cluster_all_latent_state(all_latent, all_t, all_e, all_id, all_interval_id, id2sofa, args)
        sofa = id2sofa[args.hadm_id][0][:len(latent)]
        sofa_trend = id2sofa[args.hadm_id][1][:len(latent)]
        res_single = pd.DataFrame({'sofa': sofa, 'sofa_trend': sofa_trend})
        plot_cluster_with_single_trajectory(res_all, all_latent, latent, res_single, args)
        print('Analysis finished')

    elif args.which_time == 'alltrajectory':
        if not args.existed_data:
            print('Loading data...')
            dataset = AnalysisSurvivalDataset(args.data_path)
            train_size = int(len(dataset) * args.train_ratio)
            valid_size = int(len(dataset) * args.valid_ratio)
            test_size = len(dataset) - train_size - valid_size
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
            model = load_model(dataset, args)
            all_hadm_id = []
            for x, t, e, id in tqdm(test_dataloader, desc='Getting all hadm_id'):
                all_hadm_id.extend(id)
            
            all_last_hadm_id = pickle.load(open(f'{args.output_dir}/all_id_last.pkl', 'rb'))
            assert all_hadm_id == all_last_hadm_id
                
            all_latent = []
            all_e = []
            all_t = []
            all_id = []
            for hadm_id in tqdm(all_hadm_id, desc='Getting latent state'):
                dataset = SingleSurvivalDataset(args.data_path, hadm_id)
                test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
                latent, t, e, id = get_single_trajectory(model, test_dataloader, args)
                all_latent.append(latent)
                all_t.append(t)
                all_e.append(e)
                all_id.append(hadm_id)
            with open(f'{args.output_dir}/all_latent_trajectory.pkl', 'wb') as f:
                pickle.dump(all_latent, f)
            with open(f'{args.output_dir}/all_t_trajectory.pkl', 'wb') as f:
                pickle.dump(all_t, f)
            with open(f'{args.output_dir}/all_e_trajectory.pkl', 'wb') as f:
                pickle.dump(all_e, f)
            with open(f'{args.output_dir}/all_id_trajectory.pkl', 'wb') as f:
                pickle.dump(all_id, f)
            assert len(all_latent) == len(all_t) == len(all_e) == len(all_id)
            print('Clustering...')
            res, centroids = cluster_all_trajectory(all_latent, all_t, all_e, all_id, args)
        else:
            try:
                with open(f'{args.output_dir}/all_latent_trajectory.pkl', 'rb') as f:
                    all_latent = pickle.load(f)
                with open(f'{args.output_dir}/all_t_trajectory.pkl', 'rb') as f:
                    all_t = pickle.load(f)
                with open(f'{args.output_dir}/all_e_trajectory.pkl', 'rb') as f:
                    all_e = pickle.load(f)
                with open(f'{args.output_dir}/all_id_trajectory.pkl', 'rb') as f:
                    all_id = pickle.load(f)
                if os.path.exists(f'{args.output_dir}/cluster_result_alltrajectory_{args.n_clusters}.csv'):
                    res = pd.read_csv(f'{args.output_dir}/cluster_result_alltrajectory_{args.n_clusters}.csv')
                    centroids = np.load(f'{args.output_dir}/all_trajectory_centroids.npy')
                elif args.n_clusters != 4:
                    res, centroids = cluster_all_trajectory(all_latent, all_t, all_e, all_id, args)
                elif os.path.exists(f'{args.output_dir}/cluster_result_alltrajectory.csv'):
                    res = pd.read_csv(f'{args.output_dir}/cluster_result_alltrajectory.csv')
                    centroids = np.load(f'{args.output_dir}/all_trajectory_centroids.npy')
                else:
                    res, centroids = cluster_all_trajectory(all_latent, all_t, all_e, all_id, args)
                    
                if type(res['t'][0]) == str: 
                    res["t"] = res["t"].apply(lambda x: eval(x)[0])
                    res["e"] = res["e"].apply(lambda x: eval(x)[0])
                    
                    
                print('latent_trajectory.npy, t_trajectory.npy, e_trajectory.npy, all_id_trajectory.pkl loaded')
            except:
                print('latent_trajectory.npy, t_trajectory.npy, e_trajectory.npy, all_id_trajectory.pkl not found, please set existed_data to False')
                return

        eval_cluster(res, args)
        survival_analysis(res, args)
        id2sofa = load_sofa(args.sofa_path, args.sofa_num)
        analyze_all_trajectory(res, id2sofa, args)
        all_latent = np.load(f'{args.output_dir}/latent_all.npy')
        all_res = pd.read_csv(f'{args.output_dir}/cluster_result_all.csv')
        plot_centroids(all_res, all_latent, centroids, args)
        save_sofa_trajectory(res, args)
    
    
    
    
if __name__ == '__main__':
    args = load_arguments()
    main(args)
