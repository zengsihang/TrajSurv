import pandas as pd
import numpy as np
from typing import Dict, List
import argparse

def preprocess_eicu_for_mimic_validation(eicu_file_path: str, mimic_file_path: str, output_file_path: str = None) -> pd.DataFrame:
    """
    Preprocess eICU data to match MIMIC data format for external validation.
    
    Args:
        eicu_file_path: Path to eICU CSV file
        mimic_file_path: Path to MIMIC CSV file (used for calculating mean values for imputation)
        output_file_path: Optional path to save the processed data
    
    Returns:
        Preprocessed eICU DataFrame with MIMIC feature names
    """
    
    # Define feature mapping from eICU to MIMIC
    feature_mapping = {
        # Direct mappings
        'HADM_ID': 'HADM_ID',
        'interval_id': 'interval_id',
        'event_indicator': 'event_indicator',
        'event_in_hours': 'event_in_hours',
        'age': 'age',
        'bmi': 'bmi',
        
        # Lab values
        'ALT (SGPT)': 'alanine_aminotransferase',
        'AST (SGOT)': 'asparate_aminotransferase',
        'alkaline phos.': 'alkaline_phosphatase',
        'anion gap': 'anion_gap',
        'bicarbonate': 'bicarbonate',
        'HCO3': 'calculated_total_co2',  # Similar measure
        'calcium': 'calcium_total',
        'chloride': 'chloride',
        'creatinine': 'creatinine',
        'glucose': 'glucose',
        'bedside glucose': 'glucose',  # Will take max/mean if both present
        'magnesium': 'magnesium',
        'pH': 'ph',
        'paCO2': 'pco2',
        'paO2': 'po2',
        'phosphate': 'phosphate',
        'potassium': 'potassium',
        'sodium': 'sodium',
        'total bilirubin': 'bilirubin_total',
        'BUN': 'urea_nitrogen',
        
        # Blood counts
        'Hct': 'hematocrit',
        'Hgb': 'hemoglobin',
        'MCH': 'mch',
        'MCHC': 'mchc',
        'MCV': 'mcv',
        'RBC': 'red_blood_cells',
        'RDW': 'rdw',
        'WBC x 1000': 'wbc',
        'platelets x 1000': 'platelet',
        'PT - INR': 'inr',
        'PT': 'pt',
        
        # Differentials (convert from percentages to absolute if needed)
        '-basos': 'basophils',
        '-eos': 'eosinophils', 
        '-lymphs': 'lymphocytes',
        '-monos': 'monocytes',
        
        # Vital signs
        'temperature': 'temperature',
        'sao2': 'oxygen_saturation',
        'heartrate': 'heart_rate',
        'respiration': 'respiratory_rate',
        'cvp': 'cvp',
        
        # Blood pressure - systemic maps to abp (arterial blood pressure)
        'systemicsystolic': 'abp_systolic',
        'systemicdiastolic': 'abp_diastolic', 
        'systemicmean': 'abp_mean',
        
        # Pulmonary artery pressure
        'pasystolic': 'pap_systolic',
        'padiastolic': 'pap_diastolic',
        'pamean': 'pap_mean'
    }
    
    # Read the datasets
    print("Loading datasets...")
    eicu_df = pd.read_csv(eicu_file_path)
    mimic_df = pd.read_csv(mimic_file_path)
    
    print(f"eICU data shape: {eicu_df.shape}")
    print(f"MIMIC data shape: {mimic_df.shape}")
    
    # Create a new DataFrame for processed eICU data
    processed_eicu = pd.DataFrame()
    
    # Map existing features
    print("Mapping existing features...")
    for eicu_col, mimic_col in feature_mapping.items():
        if eicu_col in eicu_df.columns:
            processed_eicu[mimic_col] = eicu_df[eicu_col].copy()
            print(f"Mapped: {eicu_col} -> {mimic_col}")
        else:
            print(f"Warning: {eicu_col} not found in eICU data")
    
    # Handle special cases where multiple eICU columns might map to one MIMIC column
    # For glucose, take the first non-null value between 'glucose' and 'bedside glucose'
    if 'glucose' in processed_eicu.columns and 'bedside glucose' in eicu_df.columns:
        glucose_combined = processed_eicu['glucose'].fillna(eicu_df['bedside glucose'])
        processed_eicu['glucose'] = glucose_combined
        print("Combined glucose and bedside glucose values")
    
    # Get all MIMIC columns to identify missing features
    mimic_columns = set(mimic_df.columns)
    processed_columns = set(processed_eicu.columns)
    missing_features = mimic_columns - processed_columns
    
    print(f"\nFeatures missing in eICU data: {len(missing_features)}")
    print("Missing features:", sorted(missing_features))
    
    # Calculate mean values from MIMIC data for imputation
    print("\nCalculating mean values for imputation...")
    mimic_means = {}
    for col in missing_features:
        if col in mimic_df.columns:
            mean_val = mimic_df[col].mean()
            if not pd.isna(mean_val):
                mimic_means[col] = mean_val
                print(f"Mean for {col}: {mean_val:.4f}")
    
    # Add missing features with mean imputation
    print("\nImputing missing features...")
    for col, mean_val in mimic_means.items():
        processed_eicu[col] = mean_val
        print(f"Imputed {col} with mean value: {mean_val:.4f}")
    
    # Ensure column order matches MIMIC data
    mimic_column_order = list(mimic_df.columns)
    processed_eicu = processed_eicu.reindex(columns=mimic_column_order)
    
    print(f"\nProcessed eICU data shape: {processed_eicu.shape}")
    print(f"Number of features: {len(processed_eicu.columns)}")
    
    # Verify all MIMIC columns are present
    missing_cols = set(mimic_column_order) - set(processed_eicu.columns)
    if missing_cols:
        print(f"Warning: Still missing columns: {missing_cols}")
    else:
        print("All MIMIC columns are present in processed eICU data")
    
    # Save processed data if output path provided
    if output_file_path:
        processed_eicu.to_csv(output_file_path, index=False)
        print(f"Processed data saved to: {output_file_path}")
    
    return processed_eicu

def get_feature_mapping_summary() -> Dict[str, str]:
    """Return the feature mapping dictionary for reference"""
    return {
        'HADM_ID': 'HADM_ID',
        'interval_id': 'interval_id', 
        'event_indicator': 'event_indicator',
        'event_in_hours': 'event_in_hours',
        'age': 'age',
        'bmi': 'bmi',
        'ALT (SGPT)': 'alanine_aminotransferase',
        'AST (SGOT)': 'asparate_aminotransferase',
        'alkaline phos.': 'alkaline_phosphatase',
        'anion gap': 'anion_gap',
        'bicarbonate': 'bicarbonate',
        'HCO3': 'calculated_total_co2',
        'calcium': 'calcium_total',
        'chloride': 'chloride',
        'creatinine': 'creatinine',
        'glucose': 'glucose',
        'magnesium': 'magnesium',
        'pH': 'ph',
        'paCO2': 'pco2',
        'paO2': 'po2',
        'phosphate': 'phosphate',
        'potassium': 'potassium',
        'sodium': 'sodium',
        'total bilirubin': 'bilirubin_total',
        'BUN': 'urea_nitrogen',
        'Hct': 'hematocrit',
        'Hgb': 'hemoglobin',
        'MCH': 'mch',
        'MCHC': 'mchc',
        'MCV': 'mcv',
        'RBC': 'red_blood_cells',
        'RDW': 'rdw',
        'WBC x 1000': 'wbc',
        'platelets x 1000': 'platelet',
        'PT - INR': 'inr',
        'PT': 'pt',
        '-basos': 'basophils',
        '-eos': 'eosinophils',
        '-lymphs': 'lymphocytes', 
        '-monos': 'monocytes',
        'temperature': 'temperature',
        'sao2': 'oxygen_saturation',
        'heartrate': 'heart_rate',
        'respiration': 'respiratory_rate',
        'cvp': 'cvp',
        'systemicsystolic': 'abp_systolic',
        'systemicdiastolic': 'abp_diastolic',
        'systemicmean': 'abp_mean',
        'pasystolic': 'pap_systolic',
        'padiastolic': 'pap_diastolic',
        'pamean': 'pap_mean'
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess eICU data to match MIMIC format')
    parser.add_argument('--eicu_file', type=str, required=True, help='Path to eICU CSV file')
    parser.add_argument('--mimic_file', type=str, required=True, help='Path to MIMIC preprocessed CSV file')
    parser.add_argument('--output_file', type=str, default='eicu_sampled_data_map_to_mimic.csv', help='Output file path')
    args = parser.parse_args()
    
    try:
        processed_data = preprocess_eicu_for_mimic_validation(
            eicu_file_path=args.eicu_file,
            mimic_file_path=args.mimic_file, 
            output_file_path=args.output_file
        )
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE!")
        print("="*50)
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Output saved to: {args.output_file}")
        
        # Display first few rows
        print("\nFirst 5 rows of processed data:")
        print(processed_data.head())
        
        # Show feature mapping summary
        print("\nFeature mapping summary:")
        mapping = get_feature_mapping_summary()
        for eicu_feat, mimic_feat in list(mapping.items())[:10]:
            print(f"  {eicu_feat} -> {mimic_feat}")
        print(f"  ... and {len(mapping)-10} more mappings")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please provide valid file paths using --eicu_file and --mimic_file arguments.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
