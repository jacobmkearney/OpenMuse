from .backends import BleakBackend


def find_devices(timeout=10, verbose=True):
    """Scan for Muse devices via Bluetooth Low Energy (BLE). Call 'find' from the terminal."""
    backend = BleakBackend()
    if verbose:
        print(f"Searching for Muses (max. {timeout} seconds)...")
    devices = backend.scan(timeout=timeout)
    muses = [d for d in devices if d.get("name") and "Muse" in d["name"]]

    if verbose:
        if muses:
            for m in muses:
                print(f'Found device {m["name"]}, MAC Address {m["address"]}')
        else:
            print("No Muses found. Ensure the device is on and Bluetooth is enabled.")

    return muses
