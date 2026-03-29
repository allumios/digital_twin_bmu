"""
config.py — central configuration for the digital twin pipeline.
Edit this file for your structure. All other scripts read from here.
"""

# --- Structure ---
MASSES = [5.36, 5.36, 5.36]          # kg, per floor (weighed)
E_MATERIAL = 200e9                    # Pa, mild steel
COLUMN_DEPTH = 3.45e-3                # m, bending direction
COLUMN_WIDTH = 0.0106                 # m, back-calculated from freq match
N_COLUMNS = 4
STOREY_HEIGHTS = [0.1845, 0.1525, 0.1520]  # m, base-to-midpoint

# --- Data files ---
SESSION_1_FILE = 'data/2026_01_12.xlsx'       # calibration dataset
SESSION_2_FILE = 'data/Input_signal_50.xlsx'  # validation dataset
SAMPLING_RATE = 2048.0                         # Hz

# Session 1 sheets (2026_01_12.xlsx)
IMPACT_SHEET_S1 = 'Impact Test'
FREE_VIB_SHEET_S1 = 'Free Vibration'
MODE1_SHEET_S1 = '1st Frequency'
MODE2_SHEET_S1 = '2nd Frequency'
MODE3_SHEET_S1 = '3rd Frequency'
EQ_SHEET_S1 = 'EQ'

# Session 2 sheets (Input_signal_50.xlsx)
IMPACT_SHEET_S2 = 'Impact Test'
FREE_VIB_SHEET_S2 = 'Free_Vibration'
MODE1_SHEET_S2 = '1st Frequency'
MODE2_SHEET_S2 = '2nd Frequency'
MODE3_SHEET_S2 = '3rd Frequency'
EQ_SHEET_S2 = 'EQ'

# --- BMU settings ---
K_PRIOR_LOW = 30000.0       # N/m
K_PRIOR_HIGH = 100000.0     # N/m
SIGMA_FRACTION = 0.02       # 2% of f_meas as measurement noise
NSAMPLES = 1000             # per TMCMC stage
TMCMC_BETA = 0.2            # proposal scaling
RANDOM_SEED = 42

# --- Measured frequencies from Session 1 FFT ---
F_MEASURED = [7.2, 21.0, 30.5]  # Hz
