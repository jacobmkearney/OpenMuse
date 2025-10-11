import numpy as np
import pandas as pd

from MuseLSL3.decode import parse_message

# These two  test  files were recorded as follows:
# ~10 seconds against black desk
# ~30 seconds against forehead (pulse-like bloodflow signal expected)
# ~10 seconds against red screen
# ~10 seconds against blue screen
# ~10 seconds against white screen
# ~10 seconds against black desk again


# OPTICS 4 ----------------------------------------------------
with open("../tests/test_data/test_optics4.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
packets = [parse_message(m, parse_leftovers=True) for m in messages]
print(f"Total packets: {len(packets)}")

# Extract OPTICS data
optics_data = []
for pkt in packets:
    for p in pkt:
        if p["pkt_type"] == "Optics4" and p["data_optics"]:
            # Each data_optics element is a 2D array (3 samples, 4 columns)
            for optics_array in p["data_optics"]:
                optics_data.append(optics_array)

# Concatenate all arrays vertically (stack samples)
optics = np.vstack(optics_data)
channels = ["OPTICS_" + str(i + 1) for i in range(optics.shape[1] - 1)]
optics = pd.DataFrame(
    optics,
    columns=["time"] + channels,
)
optics["time"] = optics["time"] - optics["time"].iloc[0]  # Start at 0
optics.plot(
    x="time",
    y=channels,
    subplots=True,
    title="Optics Data",
)
print(f"Optics samples: {len(optics)}")
print(f"Effective Optics sampling rate: {len(optics) / 90:.2f} Hz")

# OPTICS 16 ----------------------------------------------------
with open("../tests/test_data/test_optics16.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()
packets = [parse_message(m, parse_leftovers=True) for m in messages]
print(f"Total packets: {len(packets)}")

# Extract OPTICS data
optics_data = []
for pkt in packets:
    for p in pkt:
        if p["pkt_type"] == "Optics16" and p["data_optics"]:
            # Each data_optics element is a 2D array (3 samples, 4 columns)
            for optics_array in p["data_optics"]:
                optics_data.append(optics_array)

# Concatenate all arrays vertically (stack samples)
optics = np.vstack(optics_data)
channels = ["OPTICS_" + str(i + 1) for i in range(optics.shape[1] - 1)]
optics = pd.DataFrame(
    optics,
    columns=["time"] + channels,
)
optics["time"] = optics["time"] - optics["time"].iloc[0]  # Start at 0
optics.plot(
    x="time",
    y=channels,
    subplots=True,
    title="Optics Data",
)
print(f"Optics samples: {len(optics)}")
print(f"Effective Optics sampling rate: {len(optics) / 90:.2f} Hz")
