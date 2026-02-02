import pandas as pd
import ahocorasick
from tqdm import tqdm
import pickle
import argparse


itemid2colname_dict = {
    50809: 'glucose',   # lab
    50931: 'glucose',   # lab
    51221: 'hematocrit',    # lab
    50810: 'hematocrit',    # lab
    50971: 'potassium', # lab
    50983: 'sodium',    # lab
    50902: 'chloride',  # lab
    50811: 'hemoglobin',    # lab
    51222: 'hemoglobin',    # lab
    50912: 'creatinine',    # lab
    51006: 'urea_nitrogen', # lab
    50882: 'bicarbonate',   # lab
    50868: 'anion_gap', # lab
    51265: 'platelet',  # lab
    50960: 'magnesium', # lab
    51300: 'wbc',   # lab
    51301: 'wbc',   # lab
    51249: 'mchc',  # lab
    51279: 'red_blood_cells',   # lab
    51248: 'mch',   # lab
    51250: 'mcv',   # lab
    51277: 'rdw',   # lab
    50970: 'phosphate', # lab
    50893: 'calcium_total', # lab
    50820: 'ph',    # lab
    51275: 'ptt',   # lab
    51237: 'inr',   # lab
    51274: 'pt',    # lab
    50804: 'calculated_total_co2',  # lab
    50802: 'base_excess',   # lab
    50821: 'po2',   # lab
    50818: 'pco2',  # lab
    50813: 'lactate',   # lab
    50808: 'free_calcium',  # lab
    50885: 'bilirubin_total', # lab
    50861: 'alanine_aminotransferase',  # lab
    50878: 'asparate_aminotransferase', # lab
    50863: 'alkaline_phosphatase',  # lab
    50822: 'potassium_whole_blood', # lab
    51244: 'lymphocytes',   # lab
    51254: 'monocytes', # lab
    51200: 'eosinophils',   # lab
    51146: 'basophils', # lab
    220045: 'heart_rate',   # chart
    220050: 'abp_systolic', # chart
    220179: 'abp_systolic', # chart
    220051: 'abp_diastolic',    # chart
    220180: 'abp_diastolic',    # chart
    220052: 'abp_mean', # chart
    220181: 'abp_mean', # chart
    220059: 'pap_systolic', # chart
    220060: 'pap_diastolic',    # chart
    220061: 'pap_mean', # chart
    220074: 'cvp',  # chart
    220210: 'respiratory_rate', # chart
    224690: 'respiratory_rate', # chart
    223762: 'temperature',  # chart
    226512: 'admission_weight', # chart
    226730: 'height',   # chart
    220277: 'oxygen_saturation',   # chart
}

with open('cohort_ad.pkl', 'rb') as f:
    cohort_ad = pickle.load(f)


admissions = pd.read_csv("./physionet.org/files/mimiciii/1.4/ADMISSIONS.csv")
icustays = pd.read_csv("./physionet.org/files/mimiciii/1.4/ICUSTAYS.csv")
cohort_ad = ahocorasick.Automaton()
# join the two tables
# select hadm_id if icu.dbsource is metavision, admission type is not newborn, and the duration between discharge and admission is longer than 36 hours
cohort = admissions.merge(icustays, on='HADM_ID')
for index, row in tqdm(cohort.iterrows()):
    time_diff = pd.to_datetime(row['DISCHTIME']) - pd.to_datetime(row['ADMITTIME'])
    time_diff_hour = time_diff.total_seconds() / 3600
    if row['DBSOURCE'] == 'metavision' and row['ADMISSION_TYPE'] != 'NEWBORN' and time_diff_hour > 36:
        cohort_ad.add_word(str(row['HADM_ID']), (row['ADMITTIME'], row['DISCHTIME']))
cohort_ad.make_automaton()
print(len(cohort_ad))

# save to pickle
with open('cohort_ad.pkl', 'wb') as f:
    pickle.dump(cohort_ad, f)



def process_lab():
    print('Processing labevents')
    table_long = []
    labevents = pd.read_csv(lab_file)
    for index, row in tqdm(labevents.iterrows()):
        itemid = row['ITEMID']
        if itemid != itemid:
            continue
        itemid = int(itemid)
        if row['ITEMID'] not in itemid2colname_dict:
            continue
        col_name = itemid2colname_dict[row['ITEMID']]
        # if the hadm_id is in the cohort
        hadm_id = str(int(row['HADM_ID']))
        # get the interval_id
        admittime, dischtime = cohort_ad.get(hadm_id)
        admittime = pd.to_datetime(admittime)
        dischtime = pd.to_datetime(dischtime)
        charttime = pd.to_datetime(row['CHARTTIME'])
        time_diff = charttime - admittime
        time_diff_hour = time_diff.total_seconds() / 3600
        if time_diff_hour < 36 and time_diff_hour >= 0:
            interval_id = int(time_diff_hour) + 1
            table_long.append([int(row['HADM_ID']), interval_id, col_name, row['VALUENUM']])

    table_long = pd.DataFrame(table_long, columns=['hadm_id', 'interval_id', 'feature', 'value'])
    table_long.to_csv('table_long_lab.csv', index=False)
    table_wide = table_long.pivot_table(index=['hadm_id', 'interval_id'], columns='feature', values='value', aggfunc='mean').reset_index()

    table_wide.to_csv('table_wide_lab.csv', index=False)
    
def process_chart(i, chartevents_dir='./'):
    print(f'Processing chartevents{i}')
    chunksize = 10**6
    table_long = []
    chartevents_file = f"{chartevents_dir}/chartevents{i}.csv"
    for chunk in tqdm(pd.read_csv(chartevents_file, chunksize=chunksize)):
        for index, row in tqdm(chunk.iterrows()):
            hadm_id = row['HADM_ID']
            if hadm_id != hadm_id:
                continue
            hadm_id = str(int(hadm_id))
            if hadm_id not in cohort_ad:
                continue
            itemid = row['ITEMID']
            if itemid != itemid:
                continue
            itemid = int(itemid)
            if itemid not in itemid2colname_dict:
                continue
            col_name = itemid2colname_dict[itemid]
            admittime, dischtime = cohort_ad.get(hadm_id)
            admittime = pd.to_datetime(admittime)
            dischtime = pd.to_datetime(dischtime)
            charttime = pd.to_datetime(row['CHARTTIME'])
            time_diff = charttime - admittime
            time_diff_hour = time_diff.total_seconds() / 3600
            if time_diff_hour < 36 and time_diff_hour >= 0:
                interval_id = int(time_diff_hour) + 1
                table_long.append([int(row['HADM_ID']), interval_id, col_name, row['VALUENUM']])

    table_long = pd.DataFrame(table_long, columns=['hadm_id', 'interval_id', 'feature', 'value'])
    table_long.to_csv(f'table_long_chart{i}.csv', index=False)
    table_wide = table_long.pivot_table(index=['hadm_id', 'interval_id'], columns='feature', values='value', aggfunc='mean').reset_index()
    table_wide.to_csv(f'table_wide_chart{i}.csv', index=False)
    

def main(args):
    if args.data_type == 'chartevents':
        process_chart(args.index, args.chartevents_dir)
    else:
        process_lab(args.lab_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='chartevents', choices=['chartevents', 'labevents'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--lab_file', type=str, default='./lab_of_interest.csv', help='Path to lab events file')
    parser.add_argument('--chartevents_dir', type=str, default='./', help='Directory containing chartevents files')
    args = parser.parse_args()
    main(args)
