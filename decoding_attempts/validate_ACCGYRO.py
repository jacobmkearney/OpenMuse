import numpy as np
import pandas as pd

from MuseLSL3.decode import parse_message


with open("../tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
packets = [parse_message(m, parse_leftovers=True) for m in messages]
print(f"Total packets: {len(packets)}")


# Extract battery data - now each packet can have multiple battery readings
battery_data = []
for pkt in packets:
    for p in pkt:
        if p["pkt_type"] == "Battery" and p["data_battery"]:
            # Each data_battery element is a 1D array [time, battery_percent]
            for bat_reading in p["data_battery"]:
                battery_data.append(bat_reading)

if battery_data:
    battery = np.array(battery_data)
    battery = pd.DataFrame(battery, columns=["time", "battery_percent"])
    battery.plot(x="time", y="battery_percent", title="Battery Level")
    print(f"Battery readings: {len(battery)}")
else:
    print("No battery data found")

# Extract ACCGYRO data - now each packet can have multiple ACCGYRO arrays
accgyro_data = []
for pkt in packets:
    for p in pkt:
        if p["pkt_type"] == "ACCGYRO" and p["data_accgyro"]:
            # Each data_accgyro element is a 2D array (3 samples, 7 columns)
            for acc_array in p["data_accgyro"]:
                accgyro_data.append(acc_array)

# Concatenate all arrays vertically (stack samples)
accgyro = np.vstack(accgyro_data)
accgyro = pd.DataFrame(
    accgyro,
    columns=["time", "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
)
accgyro.plot(
    x="time",
    y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    subplots=True,
    title="ACCGYRO Data",
)
print(f"ACCGYRO samples: {len(accgyro)}")
print(f"Effective ACCGYRO sampling rate: {len(accgyro) / 90:.2f} Hz")


# Investigate timestamps
# Time elapsed
accgyro["time"].iloc[-1] - accgyro["time"].iloc[0]
accgyro["time"].diff().hist(bins=50)
accgyro.iloc[100:150].plot(
    x="time", y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True
)

# Investigate leftovers
leftovers = [sp["leftover"] for pkt in packets for sp in pkt]
