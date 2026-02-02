import pandas as pd
from tqdm import tqdm
import pickle
import argparse

def process_admissions(patient_file, output_admission_file='admission_info.csv', output_unit_file='unit_info.csv'):
    # Load the patient data
    patient = pd.read_csv(patient_file)

    # Define columns with lower case names
    columns = ['patientunitstayid', 'patienthealthsystemstayid', 'gender', 'age', 'hospitaladmitoffset', 'hospitaldischargeoffset', 'admissionheight', 'admissionweight', 
               'hospitaldischargestatus']

    # Rename columns to lower case
    patient.columns = [col.lower() for col in patient.columns]

    # Select specified columns
    patient = patient[columns]

    # Create admission info DataFrame
    admission_info = patient[['patienthealthsystemstayid', 'hospitaldischargeoffset', 'hospitaldischargestatus', 'hospitaladmitoffset']]

    # Calculate discharge offset by hospital admit (in hours)
    admission_info['dischargeoffsetbyhospitaladmitandhour'] = (admission_info['hospitaldischargeoffset'] - admission_info['hospitaladmitoffset']) / 60.0

    # Remove rows with missing values
    admission_info = admission_info.dropna()

    # Create event indicator
    admission_info['event_indicator'] = admission_info['hospitaldischargestatus'].map({'Alive': 0, 'Expired': 1})

    # Retain specific columns
    admission_info = admission_info[['patienthealthsystemstayid', 'dischargeoffsetbyhospitaladmitandhour', 'event_indicator']]

    # deduplicate
    admission_info = admission_info.drop_duplicates()

    # see if really deduplicated for patienthealthsystemstayid
    print(admission_info['patienthealthsystemstayid'].value_counts().max())

    # only retain those live longer than 36 hours
    admission_info = admission_info[admission_info['dischargeoffsetbyhospitaladmitandhour'] > 36]

    # print the event rate
    print(admission_info['event_indicator'].value_counts())

    # Save DataFrames to CSV
    admission_info.to_csv(output_admission_file, index=False)

    # inner join the two tables to exclude those not in the cohort
    print(patient.shape)
    patient = patient.merge(admission_info, on='patienthealthsystemstayid')
    print(patient.shape)

    patient.to_csv(output_unit_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient_file', type=str, required=True, help='Path to eICU patient.csv file')
    parser.add_argument('--output_admission_file', type=str, default='admission_info.csv')
    parser.add_argument('--output_unit_file', type=str, default='unit_info.csv')
    args = parser.parse_args()
    
    process_admissions(args.patient_file, args.output_admission_file, args.output_unit_file)
