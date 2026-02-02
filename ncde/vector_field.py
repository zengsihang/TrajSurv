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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

clinical_features_formal = {
    "time": "Time",
    "abp_diastolic": "Arterial Blood Pressure Diastolic",
    "abp_mean": "Arterial Blood Pressure Mean",
    "abp_systolic": "Arterial Blood Pressure Systolic",
    "alanine_aminotransferase": "Alanine Aminotransferase (ALT)",
    "alkaline_phosphatase": "Alkaline Phosphatase",
    "anion_gap": "Anion Gap",
    "asparate_aminotransferase": "Aspartate Aminotransferase (AST)",
    "base_excess": "Base Excess",
    "basophils": "Basophils",
    "bicarbonate": "Bicarbonate",
    "bilirubin_total": "Total Bilirubin",
    "calcium_total": "Total Calcium",
    "calculated_total_co2": "Calculated Total CO2",
    "chloride": "Chloride",
    "creatinine": "Creatinine",
    "cvp": "Central Venous Pressure (CVP)",
    "eosinophils": "Eosinophils",
    "free_calcium": "Free Calcium",
    "glucose": "Glucose",
    "heart_rate": "Heart Rate",
    "hematocrit": "Hematocrit",
    "hemoglobin": "Hemoglobin",
    "inr": "International Normalized Ratio (INR)",
    "lactate": "Lactate",
    "lymphocytes": "Lymphocytes",
    "magnesium": "Magnesium",
    "mch": "Mean Corpuscular Hemoglobin (MCH)",
    "mchc": "Mean Corpuscular Hemoglobin Concentration (MCHC)",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "monocytes": "Monocytes",
    "oxygen_saturation": "Oxygen Saturation",
    "pap_diastolic": "PAP (Diastolic)",
    "pap_mean": "PAP (Mean)",
    "pap_systolic": "PAP (Systolic)",
    "pco2": "pCO2",
    "ph": "pH Level",
    "phosphate": "Phosphate",
    "platelet": "Platelet Count",
    "po2": "Partial Pressure of Oxygen (pO2)",
    "potassium": "Potassium",
    "potassium_whole_blood": "Whole Blood Potassium",
    "pt": "Prothrombin Time (PT)",
    "ptt": "Partial Thromboplastin Time (PTT)",
    "rdw": "Red Cell Distribution Width (RDW)",
    "red_blood_cells": "Red Blood Cells Count",
    "respiratory_rate": "Respiratory Rate",
    "sodium": "Sodium",
    "temperature": "Body Temperature",
    "urea_nitrogen": "Urea Nitrogen",
    "wbc": "White Blood Cells Count",
    "gender": "Gender",
    "age": "Age",
    "bmi": "Body Mass Index (BMI)"
}




def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='wandb/output')
    parser.add_argument('--state_dict', type=str, default='best_model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data CSV')
    args = parser.parse_args()
    # use json to load args.output_dir/args.json
    with open(f'{args.output_dir}/args.json', 'r') as f:
        args_dict = json.load(f)
    # Keep user-provided data_path
    user_data_path = args.data_path
    args.__dict__.update(args_dict)
    args.data_path = user_data_path
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

def get_vector_field_func(model):
    return model.func


def get_vector_field(args):
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
    
    vector_field_func = get_vector_field_func(model)
    
    # read the latent states
    all_latent = np.load(f'{args.output_dir}/latent_all.npy')
    all_latent = torch.tensor(all_latent, dtype=torch.float32)
    vector_field = vector_field_func(t=None, z=all_latent)
    print(vector_field.shape)
    # save the vector field
    np.save(f'{args.output_dir}/vector_field.npy', vector_field.detach().numpy())
    return vector_field


def heatmap_mean_vector_field(vector_field, feature_name, args):
    
    mean_vector_field = vector_field.mean(axis=0)
    
    mean_vector_field = mean_vector_field.T[:54, ]
    feature_name = feature_name[:54]
    
    # use umap to reduce the dimension to 2
    if os.path.exists(f'{args.output_dir}/umap_fit.pkl'):
        umap_fit = pickle.load(open(f'{args.output_dir}/umap_fit.pkl', 'rb'))
    else:
        raise ValueError('umap_fit.pkl not found')
    # get the vector length of each feature
    vector_length = np.linalg.norm(mean_vector_field, axis=1)
    # plot the vector length after sorting
    idx = np.argsort(vector_length)
    plt.figure(figsize=(10, 10))
    plt.barh(np.arange(len(vector_length)), vector_length[idx])
    plt.yticks(np.arange(len(vector_length)), np.array(feature_name)[idx])
    plt.savefig(f'{args.output_dir}/vector_length.png')
    
    # save the vector length into a csv file with 2 columns
    vector_length_df = pd.DataFrame({'feature': np.array(feature_name)[idx], 'vector_length': vector_length[idx]})
    vector_length_df.to_csv(f'{args.output_dir}/vector_length.csv')

    # get the pairwise cosine similarity and plot the heatmap
    # make them into unit vectors
    mean_vector_field = mean_vector_field / np.linalg.norm(mean_vector_field, axis=1)[:, None]
    cosine_sim = mean_vector_field @ mean_vector_field.T
    plt.figure(figsize=(20, 20))
    feature_name = [clinical_features_formal[name] for name in feature_name]
    # font size 
    plt.rcParams.update({'font.size': 6, 'font.family': 'Arial'})
    # sort the heatmap
    sns.clustermap(cosine_sim, cmap='coolwarm', row_cluster=True, col_cluster=True, xticklabels=feature_name, yticklabels=feature_name)
    plt.savefig(f'{args.output_dir}/cosine_sim.png', dpi=300, bbox_inches='tight')

    # save the cosine similarity into a csv file
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=feature_name, index=feature_name)
    cosine_sim_df.to_csv(f'{args.output_dir}/cosine_sim.csv')
    
    
    # plot the clustermap
    plt.figure(figsize=(20, 40))
    sns.clustermap(mean_vector_field, yticklabels=feature_name, cmap='coolwarm', row_cluster=True, col_cluster=False)
    # smaller font
    plt.yticks(fontsize=8)
    plt.savefig(f'{args.output_dir}/mean_vector_field.png')
    

def plot_mean_vector_field_pca(vector_field, feature_name, args):
    mean_vector_field = vector_field.mean(axis=0)
    
    mean_vector_field = mean_vector_field.T[:54, ]
    feature_name = feature_name[:54]
    
    # use pca to reduce the dimension to 2
    pca = PCA(n_components=2)
    pca.fit(mean_vector_field)
    mean_vector_field = pca.transform(mean_vector_field)
    
    # plot the mean vector field
    plt.figure(figsize=(10, 10))
    plt.scatter(mean_vector_field[:, 0], mean_vector_field[:, 1])
    for i, txt in enumerate(feature_name):
        plt.annotate(txt, (mean_vector_field[i, 0], mean_vector_field[i, 1]))
    plt.savefig(f'{args.output_dir}/mean_vector_field.png')
    
    # save the mean_vector_field and names into a csv file
    mean_vector_field_df = pd.DataFrame(mean_vector_field, columns=['x', 'y'])
    mean_vector_field_df['feature'] = feature_name
    formal_name = [clinical_features_formal[name] for name in feature_name]
    mean_vector_field_df['formal_name'] = formal_name
    mean_vector_field_df.to_csv(f'{args.output_dir}/mean_vector_field_2d.csv')
    
    

def read_feature_name(data_path):
    data = pd.read_csv(data_path)
    feature = data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
    feature_name0 = feature.columns
    feature_name = ["time"] + list(feature_name0) + [f'{name}_mask' for name in feature_name0]
    return feature_name


if __name__ == '__main__':
    args = load_arguments()
    # if vector field not existed, calculate it. Otherwise, load it.
    if not os.path.exists(f'{args.output_dir}/vector_field.npy'):
        vector_field = get_vector_field(args).detach().numpy()
    else:
        vector_field = np.load(f'{args.output_dir}/vector_field.npy')
        print(vector_field.shape)
    # load the latent states
    all_latent = np.load(f'{args.output_dir}/latent_all.npy')
    feature_name = read_feature_name(args.data_path)
    heatmap_mean_vector_field(vector_field, feature_name, args)
    plot_mean_vector_field_pca(vector_field, feature_name, args)
