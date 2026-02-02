import pandas as pd
from tqdm import tqdm
import pickle
import argparse

def get_hourly_top_lab_values(lab_data, top_labs, unit_info):
    # filter lab_data with top_labs
    print('Filtering lab data with top labs...')
    lab_data = lab_data[lab_data.labname.isin(top_labs)]
    # join lab_data and unit_info by patientunitstayid
    print('Joining lab data and unit info...')
    lab_data = lab_data.merge(unit_info, on='patientunitstayid')
    # add a new column called lab_time_by_admit_in_hour, which is the difference between labresultoffset and hospitaladmitoffset / 60
    print('Adding lab_time_by_admit_in_hour...')
    lab_data['lab_time_by_admit_in_hour'] = (lab_data.labresultoffset - lab_data.hospitaladmitoffset) / 60
    # make them to int
    lab_data['lab_time_by_admit_in_hour'] = lab_data['lab_time_by_admit_in_hour'].astype(int)
    # filter lab_data with lab_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering lab data with lab_time_by_admit_in_hour >= 0 and <= 35...')
    lab_data = lab_data[(lab_data.lab_time_by_admit_in_hour >= 0) & (lab_data.lab_time_by_admit_in_hour <= 35)]
    # only left patienthealthsystemstayid, labname, labresult, lab_time_by_admit_in_hour
    lab_data = lab_data[['patienthealthsystemstayid', 'labname', 'labresult', 'lab_time_by_admit_in_hour']]
    # pivot into a wide table, with columns patienthealthsystemstayid, lab_time_by_admit_in_hour, and those labnames
    ## for value in a same hour and labname, we take the average
    print('Pivoting lab data...')
    lab_data = lab_data.pivot_table(index=['patienthealthsystemstayid', 'lab_time_by_admit_in_hour'], columns='labname', values='labresult', aggfunc='mean').reset_index()
    # save lab_wide_data
    print('Saving lab wide data...')
    lab_data.to_csv('lab_wide_data.csv', index=False)
    return lab_data

def get_hourly_vital_values(vital_data, unit_info):
    # join vital_data and unit_info by patientunitstayid
    print('Joining vital data and unit info...')
    vital_data = vital_data.merge(unit_info, on='patientunitstayid')
    # add a new column called vital_time_by_admit_in_hour, which is the difference between observationoffset and hospitaladmitoffset / 60
    print('Adding vital_time_by_admit_in_hour...')
    vital_data['vital_time_by_admit_in_hour'] = (vital_data.observationoffset - vital_data.hospitaladmitoffset) / 60
    # make them to int
    vital_data['vital_time_by_admit_in_hour'] = vital_data['vital_time_by_admit_in_hour'].astype(int)
    # filter vital_data with vital_time_by_admit_in_hour >= 0 and <= 35
    print('Filtering vital data with vital_time_by_admit_in_hour >= 0 and <= 35...')
    vital_data = vital_data[(vital_data.vital_time_by_admit_in_hour >= 0) & (vital_data.vital_time_by_admit_in_hour <= 35)]
    # only left patienthealthsystemstayid, vital_time_by_admit_in_hour, temperature, sao2,heartrate,respiration,cvp,etco2,systemicsystolic,systemicdiastolic,systemicmean,pasystolic,padiastolic,pamean
    vital_data = vital_data[['patienthealthsystemstayid', 'vital_time_by_admit_in_hour', 'temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'pasystolic', 'padiastolic', 'pamean']]
    # for values in a same hour, we take the average
    ## do not need to pivot
    
    vital_data = vital_data.groupby(['patienthealthsystemstayid', 'vital_time_by_admit_in_hour']).mean().reset_index()
    # save vital_wide_data
    print('Saving vital wide data...')
    vital_data.to_csv('vital_wide_data.csv', index=False)
    return vital_data

def get_hourly_demo_values(unit_info):
    # adding demo_time_by_admit_in_hour
    unit_info['demo_time_by_admit_in_hour'] = (- unit_info.hospitaladmitoffset) / 60
    unit_info['demo_time_by_admit_in_hour'] = unit_info['demo_time_by_admit_in_hour'].astype(int)
    # filter unit_info with demo_time_by_admit_in_hour >= 0 and <= 35
    unit_info = unit_info[(unit_info.demo_time_by_admit_in_hour >= 0) & (unit_info.demo_time_by_admit_in_hour <= 35)]
    # change Male and Female to 0 and 1 in the gender column
    unit_info['gender'] = unit_info['gender'].replace({
        'Male': 0,
        'Female': 1
    })
    # now age is string, for age >89, we set it to 90, then set age to int
    unit_info['age'] = unit_info['age'].replace('> 89', 90)
    unit_info['age'] = unit_info['age'].astype(float)
    # get BMI using admissionweight and admissionheight
    unit_info['bmi'] = unit_info['admissionweight'] / ((unit_info['admissionheight'] / 100) ** 2)
    
    
    # only left patienthealthsystemstayid, demo_time_by_admit_in_hour, age, gender, bmi, dischargeoffsetbyhospitaladmitandhour, event_indicator
    unit_info = unit_info[['patienthealthsystemstayid', 'demo_time_by_admit_in_hour', 'age', 'gender', 'bmi', 'dischargeoffsetbyhospitaladmitandhour', 'event_indicator']]
    # save demo_wide_data
    unit_info.to_csv('demo_wide_data.csv', index=False)
    return unit_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_lab', action='store_true')
    parser.add_argument('--do_vital', action='store_true')
    parser.add_argument('--do_demo', action='store_true')
    parser.add_argument('--eicu_dir', type=str, required=True, help='Path to eICU-CRD dataset directory')
    parser.add_argument('--unit_info_file', type=str, default='./unit_info.csv')
    parser.add_argument('--top_labs_file', type=str, default='top_labs.pkl')
    args = parser.parse_args()
    
    if args.do_lab:
        lab_data = pd.read_csv(f"{args.eicu_dir}/lab.csv")
        unit_info = pd.read_csv(args.unit_info_file)
        with open(args.top_labs_file, 'rb') as f:
            top_labs = pickle.load(f)
        lab_data = get_hourly_top_lab_values(lab_data, top_labs, unit_info)
    
    if args.do_vital:
        vital_data = pd.read_csv(f"{args.eicu_dir}/vitalPeriodic.csv")
        unit_info = pd.read_csv(args.unit_info_file)
        vital_data = get_hourly_vital_values(vital_data, unit_info)
    
    
    if args.do_demo:
        unit_info = pd.read_csv(args.unit_info_file)
        unit_info = get_hourly_demo_values(unit_info)
