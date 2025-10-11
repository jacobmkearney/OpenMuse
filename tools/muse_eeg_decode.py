import argparse
import numpy as np

from muse_eeg import decode_file, PRESET_DEFAULTS


def main():
    p = argparse.ArgumentParser(description='Decode EEG from .bin into NPZ (samples×channels)')
    p.add_argument('--preset', required=True)
    p.add_argument('--infile', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    samples, ch = decode_file(args.infile, args.preset)
    arr = np.stack([np.asarray(samples[i], dtype=np.int32) for i in range(ch)], axis=1)
    params = PRESET_DEFAULTS.get(args.preset, {}).copy()
    np.savez(args.out, eeg=arr, preset=args.preset, channels=ch, params=params)
    print(f'wrote {args.out} with shape {arr.shape}')


if __name__ == '__main__':
    main()


