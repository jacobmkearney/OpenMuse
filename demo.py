# OpenMuse record --address 00:55:DA:B9:FA:20 --duration 90 --outfile test_eeg_quality.txt
# OpenMuse stream --address 00:55:DA:B9:FA:20 --preset p1035
# OpenMuse stream --address 00:55:DA:B9:FA:20
# OpenMuse view

import numpy as np
import pandas as pd

import OpenMuse
import matplotlib.pyplot as plt

with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)

data["ACCGYRO"]["time"] = data["ACCGYRO"]["time"] - data["ACCGYRO"]["time"].iloc[0]

fig = data["ACCGYRO"].plot(
    x="time",
    y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    subplots=True,
    title="ACC + GYRO Movement Data",
)

plt.savefig("media/example_accgyro.png")

fig = data["EEG"].plot(
    x="time",
    y=["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"],
    subplots=True,
    title="EEG Data",
)


import OpenMuse

# 90-second test raw data with ACCGYRO and EEG
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
data = OpenMuse.decode_rawdata(messages)
print(f"EEG timestamp: {min(data['EEG']['time'])} - {max(data['EEG']['time'])}")
print(
    f"ACCGYRO timestamp: {min(data['ACCGYRO']['time'])} - {max(data['ACCGYRO']['time'])}"
)
print(f"Optics timestamp: {min(data['Optics']['time'])} - {max(data['Optics']['time'])}")
# > EEG timestamp: 0.0 - 90.828125
# > ACCGYRO timestamp: 0.0 - 90.65384615384616
# > Optics timestamp: 0.0 - 90.71875