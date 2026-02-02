from dataset import SurvivalDataset, my_collate_fn
from model import NCDESurv
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torchcde
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.nonparametric import kaplan_meier_estimator
import umap
from collections import defaultdict



def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='wandb/output')
    parser.add_argument('--state_dict', type=str, default='best_model')
    parser.add_argument('--existed_data', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data CSV')
    parser.add_argument('--sofa_path', type=str, default=None, help='Path to SOFA scores CSV')
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
    if 'no_ffn' not in args_dict:
        args.no_ffn = False
    return args

def load_sofa(sofa_path):
    """Load SOFA scores from CSV file"""
    sofa_dat = pd.read_csv(sofa_path)
    id2sofa = defaultdict(list)
    hadm_id_list = sofa_dat['HADM_ID'].unique()
    for hadm_id in tqdm(hadm_id_list, desc='Loading SOFA scores'):
        sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
        sofa = sofa.sort_values(by='interval_id')
        id2sofa[hadm_id].append(sofa['sofa'].values)
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
        no_ffn=args.no_ffn
    )
    model.load_state_dict(torch.load(f'{args.output_dir}/{args.state_dict}.pth'))
    model.eval()
    print('Model loaded from best_model.pth')
    return model

def get_latent_trajectory(model, test_dataloader, args):
    # get last hidden state
    with torch.no_grad():
        model.eval()
        model.to(args.device)
        all_last_hidden = []
        all_t = []
        all_e = []
        all_latent_trajectory = []
        for x, t, e in tqdm(test_dataloader, desc='Testing'):
            x, t, e = x.to(args.device), t.to(args.device), e.to(args.device)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            traj = model.get_latent_trajectory(coeffs)
            last_hidden = traj[..., -1, :]
            all_last_hidden.append(last_hidden)
            all_t.append(t)
            all_e.append(e)
            all_latent_trajectory.append(traj)
        all_last_hidden = torch.cat(all_last_hidden, dim=0).detach().cpu().numpy()
        all_t = torch.cat(all_t, dim=0).detach().cpu().numpy()
        all_e = torch.cat(all_e, dim=0).detach().cpu().numpy()
        all_latent_trajectory = torch.cat(all_latent_trajectory, dim=0).detach().cpu().numpy()
    return all_last_hidden, all_t, all_e, all_latent_trajectory


def visualize_trajectory(latent_trajectory, t, e, args):
    # the latent trajectory has shape (n, m, d), where n is the number of samples, m is the number of time points, d is the dimension of latent space
    # use umap to reduce the dimension of latent space to 2
    # plot the trajectory by linking the points in the latent space
    # use color to represent time and shape to represent event
    

    # exclude those that has t > 1500
    mask = t <= 1500
    latent_trajectory = latent_trajectory[mask]
    t = t[mask]
    e = e[mask]
    
    # use umap
    m = latent_trajectory.shape[1]
    last_hidden_state = latent_trajectory[:, -1, :]
    umap_model = umap.UMAP(n_components=2)
    # fit on last hidden state and transform on all latent trajectory
    umap_model = umap_model.fit(last_hidden_state)
    umap_results = umap_model.transform(latent_trajectory.reshape(-1, latent_trajectory.shape[-1]))
    umap_results = umap_results.reshape(-1, m, 2)
    # make it a list
    latent_trajectories_reduced = [umap_results[i] for i in range(umap_results.shape[0])]
    
    
    labels = []
    for time in t:
        if time <= 120:
            labels.append("0-120")
        elif time <= 240:
            labels.append("120-240")
        elif time <= 360:
            labels.append("240-360")
        elif time <= 480:
            labels.append("360-480")
        else:
            labels.append("480+")

    print('start plotting...')
    from scipy.interpolate import make_interp_spline
    from matplotlib.collections import LineCollection

    # set figure size and dpi
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Function to create smooth trajectories using B-spline interpolation
    def smooth_trajectory(points):
        x = np.linspace(0, 1, points.shape[0])
        x_new = np.linspace(0, 1, 300)  # More points for a smoother curve
        spline = make_interp_spline(x, points, k=3)
        return spline(x_new)

    for i, (trajectory, label) in tqdm(enumerate(zip(latent_trajectories_reduced, labels))):
        smooth_x = trajectory[:, 0]
        smooth_y = trajectory[:, 1]
        points = np.array([smooth_x, smooth_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a LineCollection for each trajectory
        norm = plt.Normalize(0, 1)
        if label == "0-120":
            lc = LineCollection(segments, color='green', norm=norm, linewidth=1, alpha=0.4)
        elif label == "120-240":
            lc = LineCollection(segments, color='blue', norm=norm, linewidth=1, alpha=0.4)
        elif label == "240-360":
            lc = LineCollection(segments, color='red', norm=norm, linewidth=1, alpha=0.4)
        elif label == "360-480":
            lc = LineCollection(segments, color='orange', norm=norm, linewidth=1, alpha=0.4)
        else:
            lc = LineCollection(segments, color='yellow', norm=norm, linewidth=1, alpha=0.4)
        lc.set_array(np.linspace(0, 1, segments.shape[0]))
        ax.add_collection(lc)
    # set the legend elements
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='green', lw=2, label='0-120'),
                          Line2D([0], [0], color='blue', lw=2, label='120-240'),
                          Line2D([0], [0], color='red', lw=2, label='240-360'),
                          Line2D([0], [0], color='orange', lw=2, label='360-480'),
                          Line2D([0], [0], color='yellow', lw=2, label='480+')]
    ax.legend(handles=legend_elements, title='Survival time', loc='upper right')

    trajectory = np.concatenate(latent_trajectories_reduced, axis=0)
    plt.xlim(trajectory.min() - 1, trajectory.max() + 1)
    plt.ylim(trajectory.min() - 1, trajectory.max() + 1)
    plt.tight_layout()
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('Smooth Latent Trajectories with Time Progression')
    plt.savefig(f'{args.output_dir}/trajectory.png')
    plt.close()

    
    
def visualize_last_hidden_state(last_hidden, t, e, args):
    # use t-sne to visualize last hidden state
    # use k-means to cluster last hidden state
    # use k-m survival curve to visualize the cluster

    # exclude those that has t > 1500
    mask = t <= 800
    last_hidden = last_hidden[mask]
    t = t[mask]
    e = e[mask]

    # use umap
    umap_results = umap.UMAP(n_components=2).fit_transform(last_hidden)
    tsne_df = pd.DataFrame(umap_results, columns=['umap1', 'umap2'])
    tsne_df['t'] = t
    tsne_df['e'] = e
    

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='umap1', y='umap2', hue='t', style='e', data=tsne_df)
    plt.savefig(f'{args.output_dir}/tsne.png')
    plt.close()
    
    # k-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(last_hidden)
    cluster = kmeans.predict(last_hidden)
    tsne_df['cluster'] = cluster
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='umap1', y='umap2', hue='cluster', data=tsne_df)
    plt.savefig(f'{args.output_dir}/kmeans.png')
    
    # k-m survival curve in one plot

    # make e boolean
    tsne_df['e'] = tsne_df['e'].astype(bool)
    
    plt.figure(figsize=(10, 10))
    sns.set(style='whitegrid')
    for i in range(4):
        mask_cluster = tsne_df['cluster'] == i
        time_cluster, survival_prob_cluster, conf_int = kaplan_meier_estimator(
            tsne_df['e'][mask_cluster],
            tsne_df['t'][mask_cluster],
            conf_type='log-log'
        )
        plt.step(time_cluster, survival_prob_cluster, where='post', label=f'Cluster = {i}')
        plt.fill_between(time_cluster, conf_int[0], conf_int[1], alpha=0.25, step='post')
    plt.ylim(0, 1)
    plt.ylabel(r'est. probability of survival $\hat{S}(t)$')
    plt.xlabel('time $t$')
    plt.legend(loc='best')
    plt.savefig(f'{args.output_dir}/km_survival.png')
    

    


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    print('Loading data...')
    dataset = SurvivalDataset(args.data_path)
    train_size = int(len(dataset) * args.train_ratio)
    valid_size = int(len(dataset) * args.valid_ratio)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    # load model
    print('Loading model...')
    model = load_model(dataset, args)
    
    # get last hidden state
    if not args.existed_data:
        print('Getting last hidden state...')
        last_hidden, t, e, latent_trajectory = get_latent_trajectory(model, test_dataloader, args)
        np.save(f'{args.output_dir}/last_hidden.npy', last_hidden)
        np.save(f'{args.output_dir}/t.npy', t)
        np.save(f'{args.output_dir}/e.npy', e)
        np.save(f'{args.output_dir}/latent_trajectory.npy', latent_trajectory)
        print('last_hidden.npy, t.npy, e.npy saved')
    else:
        try:
            last_hidden = np.load(f'{args.output_dir}/last_hidden.npy')
            t = np.load(f'{args.output_dir}/t.npy')
            e = np.load(f'{args.output_dir}/e.npy')
            latent_trajectory = np.load(f'{args.output_dir}/latent_trajectory.npy')
            print('last_hidden.npy, t.npy, e.npy, latent_trajectory loaded')
        except:
            print('last_hidden.npy, t.npy, e.npy, latent_trajectory not found, please set --existed_data to False')
            return
    
    # visualize last hidden state
    print('Visualizing last hidden state...')
    visualize_last_hidden_state(last_hidden, t, e, args)
    print('tsne.png, kmeans.png, km_survival.png saved')

    # visualize latent trajectory
    print('Visualizing latent trajectory...')
    visualize_trajectory(latent_trajectory, t, e, args)
    print('trajectory.png saved')
    
    
    
    
    
if __name__ == '__main__':
    args = load_arguments()
    main(args)
