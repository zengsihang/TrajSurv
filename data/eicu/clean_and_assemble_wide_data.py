import pandas as pd
import argparse

def concat_data(lab_data, vital_data, demo_data):
    # change "lab_time_by_admit_in_hour" to "interval_id"
    print('Changing column names...')
    lab_data = lab_data.rename(columns={"lab_time_by_admit_in_hour": "interval_id"})
    vital_data = vital_data.rename(columns={"vital_time_by_admit_in_hour": "interval_id"})
    demo_data = demo_data.rename(columns={"demo_time_by_admit_in_hour": "interval_id"})
    
    
    # print shapes
    print('Lab data shape:', lab_data.shape)
    print('Vital data shape:', vital_data.shape)
    print('Demo data shape:', demo_data.shape)
    
    # for demo_data, for each patienthealthsystemstayid, only retain one row by averaging the values
    # if there's NA, neglect NA in the averaging
    print('Averaging demo data...')
    demo_data = demo_data.groupby('patienthealthsystemstayid').mean(numeric_only=True).reset_index()
    # remove interval_id from demo_data
    demo_data = demo_data.drop(columns=['interval_id'])
    
    # get a template_df for all patienthealthsystemstayid and interval_id from 0 to 35
    print('Creating template data...')
    all_id = demo_data['patienthealthsystemstayid'].unique()
    all_interval_id = list(range(36))
    template_df = pd.DataFrame([(i, j) for i in all_id for j in all_interval_id], columns=['patienthealthsystemstayid', 'interval_id'])
    
    # merge lab_data and vital_data with template_df
    print('Merging lab and vital data with template data...')
    lab_data = template_df.merge(lab_data, on=['patienthealthsystemstayid', 'interval_id'], how='left')
    vital_data = template_df.merge(vital_data, on=['patienthealthsystemstayid', 'interval_id'], how='left')
    
    # merge lab_data and vital_data
    print('Merging lab and vital data...')
    data = lab_data.merge(vital_data, on=['patienthealthsystemstayid', 'interval_id'], how='inner')
    # print shape
    print('Data shape:', data.shape)

    # merge with demo_data
    print('Merging with demo data...')
    data = data.merge(demo_data, on='patienthealthsystemstayid')
    
    # remove rows with less than 20 non-NA values and interval_id != 0
    print('Removing rows with less than 20 non-NA values...')
    interval_0_data = data[data['interval_id'] == 0]
    # remove interval_id = 0
    data = data[data['interval_id'] != 0]
    data = data.dropna(thresh=20)
    data = pd.concat([data, interval_0_data])
    
    
    # remove rows with patienthealthsystemstayid that has less than 2 rows
    print('Removing rows with less than 2 rows...')
    data = data.groupby('patienthealthsystemstayid').filter(lambda group: len(group) >= 2)
    
    # sort by patienthealthsystemstayid and interval_id
    print('Sorting data...')
    data = data.sort_values(['patienthealthsystemstayid', 'interval_id'])
    
    # reset index
    data = data.reset_index(drop=True)

    
    # change patienthealthsystemstayid to HADM_ID
    data = data.rename(columns={
        "patienthealthsystemstayid": "HADM_ID",
        "dischargeoffsetbyhospitaladmitandhour": "event_in_hours"})
    
    # change all inf in the data to NA
    data = data.replace([float('inf'), float('-inf')], float('nan'))
    
    # save the data
    data.to_csv('eicu_processed_final.csv', index=False)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab_file', type=str, default='lab_wide_data.csv')
    parser.add_argument('--vital_file', type=str, default='vital_wide_data.csv')
    parser.add_argument('--demo_file', type=str, default='demo_wide_data.csv')
    args = parser.parse_args()
    
    # load the data
    lab_data = pd.read_csv(args.lab_file)
    vital_data = pd.read_csv(args.vital_file)
    demo_data = pd.read_csv(args.demo_file)
    
    # concat the data
    data = concat_data(lab_data, vital_data, demo_data)
    print(data.head())
    print(data.shape)
    print(data['HADM_ID'].nunique())
    print(data['event_indicator'].value_counts())
    print(data['event_in_hours'].describe())
