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

plt.savefig("example_accgyro.png")
