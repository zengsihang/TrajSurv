import pandas as pd
from tqdm import tqdm
import pickle
import os
from pathlib import Path
import argparse
from sofa_func import *


def get_hourly_vital_values(vital_data, unit_info):
    vital_data = vital_data.copy()
    unit_info = unit_info.copy()
    # only systemicmean (MAP) is needed
    print('Get MAP...')
    
    # join vital_data and unit_info by patientunitstayid
    print('Joining vital data and unit info...')
    vital_data = vital_data.merge(unit_info, on='patientunitstayid')
    # add a new column called vital_time_by_admit_in_hour, which is the difference between observationoffset and hospitaladmitoffset / 60
    print('Adding vital_time_by_admit_in_hour...')
    vital_data['interval_id'] = (vital_data.observationoffset - vital_data.hospitaladmitoffset) / 60
    # make them to int
    vital_data['interval_id'] = vital_data['interval_id'].astype(int)
    # filter vital_data with vital_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering vital data with vital_time_by_admit_in_hour >= 0 and <= 40...')
    vital_data = vital_data[(vital_data.interval_id >= 0) & (vital_data.interval_id <= 40)]
    # only left patienthealthsystemstayid, vital_time_by_admit_in_hour, temperature, sao2,heartrate,respiration,cvp,etco2,systemicsystolic,systemicdiastolic,systemicmean,pasystolic,padiastolic,pamean
    vital_data = vital_data[['patienthealthsystemstayid', 'interval_id', 'systemicmean']]
    # for values in a same hour, we take the average
    ## do not need to pivot
    
    vital_data = vital_data.groupby(['patienthealthsystemstayid', 'interval_id']).mean().reset_index()
    
    # remove rows with na
    vital_data = vital_data.dropna()
    
    return vital_data

def get_hourly_medication_values(infusiondrug_data, unit_info):
    infusiondrug_data = infusiondrug_data.copy()
    unit_info = unit_info.copy()
    
    # for the same patienthealthsystemstayid, fill the admissionweight if it's NA
    
    print('Filling admissionweight...')
    # only fill admissionweight but not other columns
    unit_info['admissionweight'] = unit_info.groupby('patienthealthsystemstayid')['admissionweight'].transform('mean')
    
    # get dopamine
    print('Get dopamine, dobutamine, epinephrine, norepinephrine...')
    # join infusiondrug_data and unit_info by patientunitstayid
    print('Joining infusiondrug data and unit info...')
    infusiondrug_data = infusiondrug_data.merge(unit_info, on='patientunitstayid')
    # add a new column called infusion_time_by_admit_in_hour, which is the difference between infusionoffset and hospitaladmitoffset / 60
    print('Adding infusion_time_by_admit_in_hour...')
    infusiondrug_data['interval_id'] = (infusiondrug_data.infusionoffset - infusiondrug_data.hospitaladmitoffset) / 60
    # make them to int
    infusiondrug_data['interval_id'] = infusiondrug_data['interval_id'].astype(int)
    # filter infusiondrug_data with infusion_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering infusiondrug data with infusion_time_by_admit_in_hour >= 0 and <= 40...')
    infusiondrug_data = infusiondrug_data[(infusiondrug_data.interval_id >= 0) & (infusiondrug_data.interval_id <= 40)]
    # only left patienthealthsystemstayid, infusion_time_by_admit_in_hour, drugname, drugrate
    infusiondrug_data = infusiondrug_data[['patienthealthsystemstayid', 'interval_id', 'drugname', 'drugrate', 'admissionweight']]
    processed = []
    # 4 columns, patienthealthsystemstayid, interval_id, dn, dr
    # if drugname is like %dopamine%, dn = dopamine
    ## if drugname is like %(ml/hr), then dr is the drugrate/3
    ## if drugname is like %(mcg/kg/min), then dr is the drugrate
    
    for i in tqdm(range(len(infusiondrug_data))):
        row = infusiondrug_data.iloc[i]
        dn = row['drugname'].lower()
        try:
            dr = float(row['drugrate'])
        except:
            continue
        if 'dopamine' in dn and '(mcg/kg/min)' in dn:
            dr = dr
            dn = 'dopamine'
        elif 'dopamine' in dn and '(ml/hr)' in dn:
            dr = dr * 1600. / 60. / row['admissionweight']  # we follow https://github.com/nus-mornin-lab/oxygenation_kc/blob/master/data-extraction/eICU/eicu_sofa_results.sql
            dn = 'dopamine'
        elif 'norepinephrine' in dn and '(mcg/kg/min)' in dn:
            dr = dr
            dn = 'norepinephrine'
        elif 'norepinephrine' in dn and '(mcg/min)' in dn:
            dr = dr / row['admissionweight']
            dn = 'norepinephrine'
        elif 'dobutamin' in dn:
            dr = 1.0
            dn = 'dobutamine'
        else:
            continue
        processed.append([row['patienthealthsystemstayid'], row['interval_id'], dn, dr])
    processed = pd.DataFrame(processed, columns=['patienthealthsystemstayid', 'interval_id', 'dn', 'dr'])
    processed = processed.groupby(['patienthealthsystemstayid', 'interval_id', 'dn']).mean().reset_index()
    
    # pivot table
    processed = processed.pivot_table(index=['patienthealthsystemstayid', 'interval_id'], columns='dn', values='dr').reset_index()
    
    return processed

def get_hourly_lab_values(lab_data, unit_info):
    lab_data = lab_data.copy()
    unit_info = unit_info.copy()
    
    # get lactate
    print('Get FiO2, paO2, creatinine')
    # join lab_data and unit_info by patientunitstayid
    print('Joining lab data and unit info...')
    lab_data = lab_data.merge(unit_info, on='patientunitstayid')
    # add a new column called lab_time_by_admit_in_hour, which is the difference between labresultoffset and hospitaladmitoffset / 60
    print('Adding lab_time_by_admit_in_hour...')
    lab_data['interval_id'] = (lab_data.labresultoffset - lab_data.hospitaladmitoffset) / 60
    # make them to int
    lab_data['interval_id'] = lab_data['interval_id'].astype(int)
    # filter lab_data with lab_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering lab data with lab_time_by_admit_in_hour >= 0 and <= 40...')
    lab_data = lab_data[(lab_data.interval_id >= 0) & (lab_data.interval_id <= 40)]
    # only left patienthealthsystemstayid, lab_time_by_admit_in_hour, labname, labresult
    lab_data = lab_data[['patienthealthsystemstayid', 'interval_id', 'labname', 'labresult']]
    lab_data = lab_data[lab_data.labname.isin(['FiO2', 'paO2', 'creatinine', 'total bilirubin', 'platelets x 1000'])]
    # average labresult in the same hour
    lab_data = lab_data.groupby(['patienthealthsystemstayid', 'interval_id', 'labname']).mean().reset_index()
    # pivot table
    lab_data = lab_data.pivot_table(index=['patienthealthsystemstayid', 'interval_id'], columns='labname', values='labresult').reset_index()
    
    return lab_data

def get_hourly_gcs_values(physicalexam_data, unit_info):
    physicalexam_data = physicalexam_data.copy()
    unit_info = unit_info.copy()
    
    # get GCS
    print('Get GCS...')
    # join physicalexam_data and unit_info by patientunitstayid
    print('Joining physicalexam data and unit info...')
    physicalexam_data = physicalexam_data.merge(unit_info, on='patientunitstayid')
    # add a new column called exam_time_by_admit_in_hour, which is the difference between examoffset and hospitaladmitoffset / 60
    print('Adding exam_time_by_admit_in_hour...')
    physicalexam_data['interval_id'] = (physicalexam_data.physicalexamoffset - physicalexam_data.hospitaladmitoffset) / 60
    # make them to int
    physicalexam_data['interval_id'] = physicalexam_data['interval_id'].astype(int)
    # filter physicalexam_data with exam_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering physicalexam data with exam_time_by_admit_in_hour >= 0 and <= 40...')
    physicalexam_data = physicalexam_data[(physicalexam_data.interval_id >= 0) & (physicalexam_data.interval_id <= 40)]
    # only left patienthealthsystemstayid, exam_time_by_admit_in_hour, GCS
    physicalexam_data = physicalexam_data[['patienthealthsystemstayid', 'interval_id', 'physicalexampath', 'physicalexamvalue']]
    # only left GCS
    processed = []
    for i in tqdm(range(len(physicalexam_data))):
        row = physicalexam_data.iloc[i]
        path = row['physicalexampath'].lower()
        value = row['physicalexamvalue']
        try:
            value = float(value)
        except:
            continue
        if 'gcs/eyes' in path:
            name = 'gcs/eyes'
            value = value
        elif 'gcs/motor' in path:
            name = 'gcs/motor'
            value = value
        elif 'gcs/verbal' in path:
            name = 'gcs/verbal'
            value = value
        else:
            continue
        processed.append([row['patienthealthsystemstayid'], row['interval_id'], name, value])
    processed = pd.DataFrame(processed, columns=['patienthealthsystemstayid', 'interval_id', 'name', 'value'])
    processed = processed.groupby(['patienthealthsystemstayid', 'interval_id', 'name']).mean().reset_index()
    # pivot table
    processed = processed.pivot_table(index=['patienthealthsystemstayid', 'interval_id'], columns='name', values='value').reset_index()
    
    return processed

def get_hourly_mv_values(respcare_data, unit_info):
    respcare_data = respcare_data.copy()
    unit_info = unit_info.copy()
    
    # get MV
    print('Get MV...')
    # join respcare_data and unit_info by patientunitstayid
    print('Joining respcare data and unit info...')
    respcare_data = respcare_data.merge(unit_info, on='patientunitstayid')
    
    # calculate care_time_by_admit_in_hour
    print('Adding care_time_by_admit_in_hour...')
    respcare_data['mv_start_id'] = (respcare_data.ventstartoffset - respcare_data.hospitaladmitoffset) / 60
    respcare_data['mv_end_id'] = (respcare_data.ventendoffset - respcare_data.hospitaladmitoffset) / 60
    
    # filter out invalid data
    respcare_data = respcare_data.dropna(subset=['mv_start_id', 'mv_end_id'])
    respcare_data = respcare_data[respcare_data['mv_start_id'] < respcare_data['mv_end_id']]
    
    # convert to integers
    respcare_data['mv_start_id'] = respcare_data['mv_start_id'].astype(int)
    respcare_data['mv_end_id'] = respcare_data['mv_end_id'].astype(int)
    
    # filter data within range
    print('Filtering respcare data with care_time_by_admit_in_hour >= 0 and <= 40...')
    respcare_data = respcare_data[(respcare_data.mv_end_id >= 0) & (respcare_data.mv_end_id <= 40)]
    respcare_data = respcare_data[['patienthealthsystemstayid', 'mv_start_id', 'mv_end_id']]
    
    # get unique patient IDs and create template dataframe
    all_id = unit_info['patienthealthsystemstayid'].unique()
    all_interval_id = list(range(41))
    
    # Use a different approach to create the result DataFrame
    # First create a dictionary to track which intervals have MV for each patient
    mv_intervals = {patient_id: set() for patient_id in all_id}
    
    # Populate the sets with the intervals that have MV
    for _, row in respcare_data.iterrows():
        pid = row['patienthealthsystemstayid']
        start = max(0, row['mv_start_id'])
        end = min(40, row['mv_end_id'])
        
        # Add all intervals between start and end (inclusive)
        mv_intervals[pid].update(range(start, end + 1))
    
    # Create the result DataFrame
    result_data = []
    for pid in all_id:
        for interval in all_interval_id:
            result_data.append((pid, interval, 1 if interval in mv_intervals[pid] else 0))
    
    processed = pd.DataFrame(result_data, columns=['patienthealthsystemstayid', 'interval_id', 'mv'])
    
    # get the average of mv across the df
    print('Getting average of MV...')
    print(processed['mv'].mean())
    
    return processed



def get_sofa_wide_data(eicu_dir, unit_info_file='./unit_info.csv', sofa_output_dir='./sofa'):
    # load data
    if os.path.exists(f'{sofa_output_dir}/mv_data_for_sofa.csv'):
        mv_data = pd.read_csv(f'{sofa_output_dir}/mv_data_for_sofa.csv')
    else:
        respcare_data = pd.read_csv(f'{eicu_dir}/respiratoryCare.csv')
        unit_info = pd.read_csv(unit_info_file)
        mv_data = get_hourly_mv_values(respcare_data, unit_info)
        mv_data.to_csv(f'{sofa_output_dir}/mv_data_for_sofa.csv', index=False)
    if os.path.exists(f'{sofa_output_dir}/gcs_data_for_sofa.csv'):
        gcs_data = pd.read_csv(f'{sofa_output_dir}/gcs_data_for_sofa.csv')
    else:
        physicalexam_data = pd.read_csv(f'{eicu_dir}/physicalExam.csv')
        unit_info = pd.read_csv(unit_info_file)
        gcs_data = get_hourly_gcs_values(physicalexam_data, unit_info)
        gcs_data.to_csv(f'{sofa_output_dir}/gcs_data_for_sofa.csv', index=False)
    if os.path.exists(f'{sofa_output_dir}/medication_data_for_sofa.csv'):
        medication_data = pd.read_csv(f'{sofa_output_dir}/medication_data_for_sofa.csv')
    else:
        infusiondrug_data = pd.read_csv(f'{eicu_dir}/infusionDrug.csv')
        unit_info = pd.read_csv(unit_info_file)
        medication_data = get_hourly_medication_values(infusiondrug_data, unit_info)
        medication_data.to_csv(f'{sofa_output_dir}/medication_data_for_sofa.csv', index=False)
    if os.path.exists(f'{sofa_output_dir}/lab_data_for_sofa.csv'):
        lab_data = pd.read_csv(f'{sofa_output_dir}/lab_data_for_sofa.csv')
    else:
        lab_data = pd.read_csv(f'{eicu_dir}/lab.csv')
        unit_info = pd.read_csv(unit_info_file)
        lab_data = get_hourly_lab_values(lab_data, unit_info)
        lab_data.to_csv(f'{sofa_output_dir}/lab_data_for_sofa.csv', index=False)
    if os.path.exists(f'{sofa_output_dir}/map_data_for_sofa.csv'):
        map_data = pd.read_csv(f'{sofa_output_dir}/map_data_for_sofa.csv')
    else:
        vital_data = pd.read_csv(f'{eicu_dir}/vitalPeriodic.csv')
        unit_info = pd.read_csv(unit_info_file)
        map_data = get_hourly_vital_values(vital_data, unit_info)
        map_data.to_csv(f'{sofa_output_dir}/map_data_for_sofa.csv', index=False)
    
    # merge them together
    
    data_list = [mv_data, gcs_data, medication_data, lab_data, map_data]
    # change the patienthealthsystemstayid to HADM_ID
    for i in range(len(data_list)):
        data_list[i] = data_list[i].rename(columns={'patienthealthsystemstayid': 'HADM_ID'})
    
    ## get all the HADM_ID in eicu_processed_final.csv
    dat = pd.read_csv("./eicu_processed_final.csv")
    HADM_IDs = dat['HADM_ID'].unique()
    
    ## create a template dataframe with all HADM_ID and interval_id from 0 to 40
    all_interval_id = list(range(41))
    all_data = []
    for HADM_ID in HADM_IDs:
        for interval_id in all_interval_id:
            all_data.append([HADM_ID, interval_id])
    all_data = pd.DataFrame(all_data, columns=['HADM_ID', 'interval_id'])
    
    print("Joining data...")
    # join them together
    for i in range(len(data_list)):
        all_data = all_data.merge(data_list[i], on=['HADM_ID', 'interval_id'], how='left')
    
    # fill the NA values by HADM_ID, forward fill and backward fill
    all_data = all_data.groupby('HADM_ID').apply(lambda group: group.ffill().bfill())
    
    # get the sofa score
    processed = []
    for i in tqdm(range(len(all_data))):
        row = all_data.iloc[i]
        MAP_value = row['systemicmean']
        GCS_Eyes = row['gcs/eyes']
        GCS_Motor = row['gcs/motor']
        GCS_Verbal = row['gcs/verbal']
        dopamine = row['dopamine']
        dobutamine = row['dobutamine']
        norepinephrine = row['norepinephrine']
        bilirubin = row['total bilirubin']
        platelets = row['platelets x 1000']
        creatinine = row['creatinine']
        pao2 = row['paO2']
        fio2 = row['FiO2']
        mv = row['mv']
        sofa1 = sofa1_func(pao2, fio2, mv)
        sofa2 = sofa2_func(GCS_Eyes + GCS_Motor + GCS_Verbal)
        
        # sofa3
        use_ddm = (dopamine > 0 and dopamine <= 5) or dobutamine > 0
        use_enpv = dopamine > 5 or (norepinephrine > 0 and norepinephrine <= .1)
        use_enpv_hard = dopamine > 15 or norepinephrine > .1
        sofa3 = sofa3_func(MAP_value, use_ddm, use_enpv, use_enpv_hard)
        sofa4 = sofa4_func(bilirubin)
        sofa5 = sofa5_func(platelets)
        sofa6 = sofa6_func(creatinine)

        sofa = sofa1 + sofa2 + sofa3 + sofa4 + sofa5 + sofa6
        
        processed.append([row['HADM_ID'], row['interval_id'], sofa1, sofa2, sofa3, sofa4, sofa5, sofa6, sofa])
    
    processed = pd.DataFrame(processed, columns=['HADM_ID', 'interval_id', 'sofa1', 'sofa2', 'sofa3', 'sofa4', 'sofa5', 'sofa6', 'sofa'])
    
    # save the processed data
    processed.to_csv(f'{sofa_output_dir}/eicu_sofa_detailed_data.csv', index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eicu_dir', type=str, required=True, help='Path to eICU-CRD dataset directory')
    parser.add_argument('--unit_info_file', type=str, default='./unit_info.csv')
    parser.add_argument('--sofa_output_dir', type=str, default='./sofa')
    args = parser.parse_args()
    
    # make a dir called sofa
    os.makedirs(args.sofa_output_dir, exist_ok=True)
    
    get_sofa_wide_data(args.eicu_dir, args.unit_info_file, args.sofa_output_dir)
