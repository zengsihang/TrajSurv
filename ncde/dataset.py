import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import torchcde
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse


class SurvivalDataset(Dataset):
    def __init__(self, data_path, separate=False, sofa_derivative_coeff=10.0, sofa_path=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.data_standardization()
        self.split_by_id()
        # Use provided sofa_path or derive from data_path
        if sofa_path is None:
            # Default: assume sofa file is in same directory structure
            import os
            data_dir = os.path.dirname(data_path)
            sofa_path = os.path.join(data_dir, 'mimic_sofa_scores_detail.csv')
        self.id2sofa = self.load_sofa(sofa_path, separate=separate, sofa_derivative_coeff=sofa_derivative_coeff)
    
    def data_standardization(self):
        scaler = StandardScaler()
        feature = self.data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        self.data[feature.columns] = scaler.fit_transform(feature)

    def load_sofa(self, sofa_path, separate=False, sofa_derivative_coeff=10.0):
        print("sofa_derivative_coeff", sofa_derivative_coeff)
        # load id2sofa
        sofa_dat = pd.read_csv(sofa_path)
        id2sofa = defaultdict(list)
        hadm_id_list = sofa_dat['HADM_ID'].unique()
        for hadm_id in tqdm(hadm_id_list):
            sofa = sofa_dat[sofa_dat['HADM_ID'] == hadm_id]
            sofa = sofa.sort_values(by='interval_id')
            if not separate:
                id2sofa[hadm_id].append(sofa['sofa'].values)
                # get the time points
                time_points = sofa['interval_id'].values
                # get the sofa values
                sofa_values = sofa['sofa'].values
                
                sofa_derivative = (sofa_values[4:] - sofa_values[:-4]) / 4.0
                sofa_derivative = np.append(np.repeat(sofa_derivative[0], 2), sofa_derivative)
                sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], 2))

                # make the sofa_derivative same scale as sofa score
                sofa_derivative *= sofa_derivative_coeff
                
                id2sofa[hadm_id].append(sofa_derivative)
            else:
                all_sofa_score = []
                all_sofa_derivative = []
                for i in range(7):
                    if i == 0:
                        sofa_values = sofa['sofa'].values
                    else:
                        sofa_values = sofa['sofa' + str(i)].values
                    sofa_derivative = (sofa_values[4:] - sofa_values[:-4]) / 4.0
                    sofa_derivative = np.append(np.repeat(sofa_derivative[0], 2), sofa_derivative)
                    sofa_derivative = np.append(sofa_derivative, np.repeat(sofa_derivative[-1], 2))
                    
                    # make the sofa_derivative same scale as sofa score
                    sofa_derivative *= sofa_derivative_coeff
                    
                    all_sofa_score.append(sofa_values)
                    all_sofa_derivative.append(sofa_derivative)
                # make all_sofa_score and all_sofa_derivative into numpy array of shape (len(time_points), 6)
                all_sofa_score = np.array(all_sofa_score).T
                all_sofa_derivative = np.array(all_sofa_derivative).T
                id2sofa[hadm_id].append(all_sofa_score)
                id2sofa[hadm_id].append(all_sofa_derivative)
        
            
        return id2sofa
    
    def split_by_id(self):
        print("Processing by id...")
        self.id_list = self.data['HADM_ID'].unique()
        # id2data
        self.id2data = {}
        for id in tqdm(self.id_list):
            # sort by interval_id
            self.id2data[id] = self.data[self.data['HADM_ID'] == id].sort_values(by='interval_id')
    
    def __getitem__(self, index):
        id = self.id_list[index]
        data = self.id2data[id]
        t = data['event_in_hours'].values[0] - data['interval_id'].values[-1]
        time_steps = data['interval_id'].values
        e = data['event_indicator'].values[-1]
        data = data.drop(columns=['HADM_ID', 'interval_id', 'event_in_hours', 'event_indicator'])
        data_matrix = data.values
        # concat cumsum mask to the right of data_matrix
        mask = (~np.isnan(data_matrix)).cumsum(axis=0)
        data_matrix = np.concatenate([data_matrix, mask], axis=1)
        # concat time_steps to data_matrix as first column
        data_matrix = np.concatenate([time_steps.reshape(-1, 1), data_matrix], axis=1)
        
        sofa = self.id2sofa[id][0][time_steps]
        sofa_trend = self.id2sofa[id][1][time_steps]
        return data_matrix, t, e, sofa, sofa_trend
    
    def __len__(self):
        return len(self.id_list)




def fill_forward(x_list):
    max_length = max([x.size(0) for x in x_list])
    x_list = [torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))]) for x in x_list]
    return torch.stack(x_list)

def my_collate_fn(batch):
    x_list, t_list, e_list, sofa_list, sofa_trend_list = zip(*batch)
    length_x = [x.shape[0] for x in x_list]
    x_list = [torch.FloatTensor(x) for x in x_list]
    x = fill_forward(x_list)
    t = torch.FloatTensor(t_list)
    e = torch.FloatTensor(e_list)
    length_x = torch.LongTensor(length_x)
    # concat all sofa values into a long tensor
    sofa = [torch.FloatTensor(s) for s in sofa_list]
    sofa = torch.cat(sofa)
    # concat all sofa trend values into a long tensor
    sofa_trend = [torch.FloatTensor(s) for s in sofa_trend_list]
    sofa_trend = torch.cat(sofa_trend)
    
    # if sofa and sofa_trend are 1-d tensor, then expand them to 2-d tensor
    if sofa.dim() == 1:
        sofa = sofa.unsqueeze(1)
    if sofa_trend.dim() == 1:
        sofa_trend = sofa_trend.unsqueeze(1)
    # concat sofa and sofa_trend into a single tensor with cat
    sofa_label = torch.cat([sofa, sofa_trend], dim=1)
    
    return x, t, e, sofa_label, length_x

# test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data CSV')
    parser.add_argument('--sofa_path', type=str, default=None, help='Path to SOFA scores CSV')
    args = parser.parse_args()
    
    dataset = SurvivalDataset(args.data_path, sofa_path=args.sofa_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=my_collate_fn)
    count = 0
    for x, t, e, sofa_label, length_x in dataloader:
        print(x.size(), t.size(), e.size())
        count += 1
        if count == 10:
            break
    print("Done.")
