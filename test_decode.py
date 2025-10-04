import MuseLSL3
import json
import pandas as pd

# --------------------------------
# Raw data
# --------------------------------

with open("data.txt") as f:
    lines = f.readlines()

data = MuseLSL3.decode_rawdata(lines)
acc = data["ACC"]
acc["time"] = acc["time"] - acc["time"].iloc[0]
acc.plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)

gyro = data["GYRO"]
gyro["time"] = gyro["time"] - gyro["time"].iloc[0]
gyro.plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)


for i, line in enumerate(lines[:5]):
    parts = line.split("\t")
    if len(parts) >= 3:
        hexstr = parts[2].strip()
        print(f"\nLine {i}: UUID={parts[1][:20]}..., length={len(hexstr)} chars")
        print(f"  Contains 0x47: {hexstr.count('47')}")

        result = MuseLSL3.decode.decode_message(line)
        if result:
            print(f"  ACC: {len(result.get('ACC', []))} samples")
            print(f"  GYRO: {len(result.get('GYRO', []))} samples")
            print(f"  EEG: {len(result.get('EEG', []))} samples")


# --------------------------------
# LSL
# --------------------------------


# read data.json
with open("data.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
acc = pd.DataFrame(data["ACC"])
acc["time"] = acc["time"] - acc["time"].iloc[0]
acc["time_lsl"] = acc["time_lsl"] - acc["time_lsl"].iloc[0]
acc.plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)

acc.plot(x="time", y="time_lsl")

gyro = pd.DataFrame(data["GYRO"])
gyro["time"] = gyro["time"] - gyro["time"].iloc[0]
gyro["time_lsl"] = gyro["time_lsl"] - gyro["time_lsl"].iloc[0]
gyro.plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)
