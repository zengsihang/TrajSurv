import pandas as pd
import argparse

def main(args):
    dat1 = pd.read_csv(args.chart0_file)
    dat2 = pd.read_csv(args.chart1_file)
    dat3 = pd.read_csv(args.lab_file)

    # concat then horizontally
    dat = pd.concat([dat1, dat2, dat3], axis=0)

    # pivot
    dat_wide = dat.pivot_table(index=['hadm_id', 'interval_id'], columns='feature', values='value', aggfunc='mean').reset_index()

    admissions = pd.read_csv(args.admissions_file)
    # dat wide left join admissions to get the age, gender, admittime, dischtime, deathtime, event time, event indicator
    # change the column name of hadm_id to HADM_ID
    dat_wide = dat_wide.rename(columns={'hadm_id': 'HADM_ID'})
    wide_columns = dat_wide.columns
    dat_wide = dat_wide.merge(admissions, on='HADM_ID', how='left')

    patients = pd.read_csv(args.patients_file)
    dat_wide = dat_wide.merge(patients, on='SUBJECT_ID', how='left')

    dat_wide = dat_wide[wide_columns.tolist() + ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'GENDER', 'DOB', 'ETHNICITY']]
    # if deathtime is not null, then event_indicator = 1, event time = deathtime
    # if deathtime is null, then event_indicator = 0, event time = dischtime
    dat_wide['event_time'] = dat_wide['DEATHTIME'].fillna(dat_wide['DISCHTIME'])
    dat_wide['event_indicator'] = dat_wide['DEATHTIME'].notnull().astype(int)
    dat_wide['event_in_hours'] = (pd.to_datetime(dat_wide['event_time']) - pd.to_datetime(dat_wide['ADMITTIME'])).dt.total_seconds() / 3600
    dat_wide['gender'] = dat_wide['GENDER'].map({'M': 0, 'F': 1})
    dat_wide['age'] = (pd.to_datetime(dat_wide['ADMITTIME']).dt.year - pd.to_datetime(dat_wide['DOB']).dt.year)

    dat_wide.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart0_file', type=str, default='./table_long_chart0.csv')
    parser.add_argument('--chart1_file', type=str, default='./table_long_chart1.csv')
    parser.add_argument('--lab_file', type=str, default='./table_long_lab.csv')
    parser.add_argument('--admissions_file', type=str, required=True, help='Path to MIMIC ADMISSIONS.csv')
    parser.add_argument('--patients_file', type=str, required=True, help='Path to MIMIC PATIENTS.csv')
    parser.add_argument('--output_file', type=str, default='table_wide_lab_chart.csv')
    args = parser.parse_args()
    main(args)
