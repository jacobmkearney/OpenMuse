import pandas as pd
import os


TARGET_RATES = ({"EEG": 256.0, "IMU": 52.0, "PPG": 64.0, "OPTICS": 64.0},)
CAND_CHANNELS = {
    "EEG": [4, 5, 8],
    "IMU": [6],
    "PPG": [4, 6, 8],
    "OPTICS": [4, 5, 8, 16],
}
CAND_BIT_WIDTHS = {"EEG": 14, "IMU": 16, "PPG": 20, "OPTICS": 20}


files = [f for f in os.listdir("data_raw") if f.endswith(".txt")]

# Define decoding functions here


# Loop through files and analyze
