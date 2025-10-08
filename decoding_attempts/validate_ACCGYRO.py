import numpy as np
import pandas as pd

from MuseLSL3.decode import parse_message


with open("../tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
packets = [parse_message(m) for m in messages]
print(f"Total packets: {len(packets)}")

battery = [
    p["data_battery"] for pkt in packets for p in pkt if p["pkt_type"] == "Battery"
]
battery = np.concatenate(battery)
battery = pd.DataFrame(battery, columns=["time", "battery_percent"])
battery.plot(x="time", y="battery_percent")


accgyro = [
    p["data_accgyro"] for pkt in packets for p in pkt if p["pkt_type"] == "ACCGYRO"
]
accgyro = np.concatenate(accgyro, axis=1)[0, :, :]
accgyro = pd.DataFrame(
    accgyro, columns=["time", "ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
)
accgyro.plot(
    x="time", y=["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True
)

leftovers = [a["leftover"] for a in accgyro if a["pkt_valid"] and a["leftover"]]
