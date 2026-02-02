import pandas as pd


def sofa1_func(po2, fio2, onMV_flag):
    """
    Calculate SOFA score for respiratory system based on PaO2/FiO2 ratio.
    
    Args:
        po2: Partial pressure of oxygen
        fio2: Fraction of inspired oxygen (as a percentage, e.g., 21 for room air)
        onMV_flag: Flag indicating if patient is on mechanical ventilation (1=yes, 0=no)
        
    Returns:
        int: SOFA score for respiratory system (0-4)
    """
    # 1. Compute P/F ratio
    if pd.isna(po2) or pd.isna(fio2):  # if one value is not available
        PF_ratio = float('nan')
    else:
        PF_ratio = (po2 / (fio2 / 100))  # convert FIO2 to actual percentage
    
    # 2. Conditions
    if pd.isna(PF_ratio):
        highest_score = 0
    else:
        if PF_ratio >= 400:
            scoreA = 0
        else:
            scoreA = float('nan')
        
        if PF_ratio < 400:
            scoreB = 1
        else:
            scoreB = float('nan')
        
        if PF_ratio < 300:
            scoreC = 2
        else:
            scoreC = float('nan')
        
        if PF_ratio < 200 and onMV_flag == 1:
            scoreD = 3
        else:
            scoreD = float('nan')
            
        if PF_ratio < 100 and onMV_flag == 1:
            scoreE = 4
        else:
            scoreE = float('nan')
        
        all_possible_scores = [scoreA, scoreB, scoreC, scoreD, scoreE]
        # Remove NaN values before finding max
        valid_scores = [score for score in all_possible_scores if not pd.isna(score)]
        highest_score = max(valid_scores) if valid_scores else 0
    
    return highest_score


def sofa2_func(GCS_value):
    """
    Calculate SOFA score for central nervous system based on Glasgow Coma Scale.
    
    Args:
        GCS_value: Glasgow Coma Scale value (3-15)
        
    Returns:
        int: SOFA score for central nervous system (0-4)
    """
    if pd.isna(GCS_value):  # if value is not available, treat it as 15
        highest_score = 0
    else:
        # make it int
        GCS_value = int(GCS_value)
        
        if GCS_value == 15:
            scoreA = 0
        else:
            scoreA = float('nan')
        
        if 13 <= GCS_value <= 14:
            scoreB = 1
        else:
            scoreB = float('nan')
            
        if 10 <= GCS_value <= 12:
            scoreC = 2
        else:
            scoreC = float('nan')
            
        if 6 <= GCS_value <= 9:
            scoreD = 3
        else:
            scoreD = float('nan')
            
        if GCS_value < 6:
            scoreE = 4
        else:
            scoreE = float('nan')
        
        all_possible_scores = [scoreA, scoreB, scoreC, scoreD, scoreE]
        # Remove NaN values before finding max
        valid_scores = [score for score in all_possible_scores if not pd.isna(score)]
        highest_score = max(valid_scores) if valid_scores else 0
    
    return highest_score


def sofa3_func(MAP_value, Use_DDM, UseENPV, UseENPVHard):
    """
    Calculate SOFA score for cardiovascular system based on MAP and vasopressors.
    
    Args:
        MAP_value: Mean Arterial Pressure
        Use_DDM: Flag indicating use of dopamine ≤ 5 μg/kg/min or dobutamine (any dose)
        UseENPV: Flag for dopamine > 5 μg/kg/min or epinephrine ≤ 0.1 μg/kg/min or norepinephrine ≤ 0.1 μg/kg/min
        UseENPVHard: Flag for dopamine > 15 μg/kg/min or epinephrine > 0.1 μg/kg/min or norepinephrine > 0.1 μg/kg/min
        
    Returns:
        int: SOFA score for cardiovascular system (0-4)
    """
    if pd.isna(MAP_value):  # if value is not available
        Score_MAP = 0
    else:
        if MAP_value >= 70:
            Score_MAP = 0
        elif MAP_value < 70:
            Score_MAP = 1
    
    if Use_DDM == 1:
        Score_DDM = 2
    else:
        Score_DDM = float('nan')
    
    if UseENPV == 1:
        Score_ENPV = 3
    else:
        Score_ENPV = float('nan')
    
    if UseENPVHard == 1:
        Score_ENPVHard = 4
    else:
        Score_ENPVHard = float('nan')
    
    all_possible_scores = [Score_MAP, Score_DDM, Score_ENPV, Score_ENPVHard]
    # Remove NaN values before finding max
    valid_scores = [score for score in all_possible_scores if not pd.isna(score)]
    highest_score = max(valid_scores) if valid_scores else 0
    
    return highest_score


def sofa4_func(bilirubin_value):
    """
    Calculate SOFA score for liver based on bilirubin value.
    
    Args:
        bilirubin_value: Total bilirubin level (mg/dL)
        
    Returns:
        int: SOFA score for liver (0-4)
    """
    if pd.isna(bilirubin_value):  # if value is not available
        sofa_score = 0
    else:
        if bilirubin_value < 1.2:
            sofa_score = 0
        elif 1.2 <= bilirubin_value < 2:
            sofa_score = 1
        elif 2 <= bilirubin_value < 6:
            sofa_score = 2
        elif 6 <= bilirubin_value < 12:
            sofa_score = 3
        elif bilirubin_value >= 12:
            sofa_score = 4
    
    return sofa_score


def sofa5_func(Platelets_value):
    """
    Calculate SOFA score for coagulation based on platelet count.
    
    Args:
        Platelets_value: Platelet count (×10³/μL)
        
    Returns:
        int: SOFA score for coagulation (0-4)
    """
    if pd.isna(Platelets_value):  # if value is not available
        highest_score = 0
    else:
        if Platelets_value >= 150:
            scoreA = 0
        else:
            scoreA = float('nan')
        
        if Platelets_value < 150:
            scoreB = 1
        else:
            scoreB = float('nan')
        
        if Platelets_value < 100:
            scoreC = 2
        else:
            scoreC = float('nan')
        
        if Platelets_value < 50:
            scoreD = 3
        else:
            scoreD = float('nan')
        
        if Platelets_value < 20:
            scoreE = 4
        else:
            scoreE = float('nan')
        
        all_possible_scores = [scoreA, scoreB, scoreC, scoreD, scoreE]
        # Remove NaN values before finding max
        valid_scores = [score for score in all_possible_scores if not pd.isna(score)]
        highest_score = max(valid_scores) if valid_scores else 0
    
    return highest_score


def sofa6_func(sCr_Value):
    """
    Calculate SOFA score for renal system based on serum creatinine.
    
    Args:
        sCr_Value: Serum creatinine level (mg/dL)
        
    Returns:
        int: SOFA score for renal system (0-4)
    """
    if pd.isna(sCr_Value):  # if value is not available
        sofa_score = 0
    else:
        if sCr_Value < 1.2:
            sofa_score = 0
        elif 1.2 <= sCr_Value < 2:
            sofa_score = 1
        elif 2 <= sCr_Value < 3.5:
            sofa_score = 2
        elif 3.5 <= sCr_Value < 5:
            sofa_score = 3
        elif sCr_Value >= 5:
            sofa_score = 4
    
    return sofa_score