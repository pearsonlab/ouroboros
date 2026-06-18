"""Held-out per-vocalization window loaders for autonomy evaluation.

Both helpers carry the int16-normalize fix (divide by 32768 on int16 input) so val/test
inputs are on the same scale as the (float-converted) training windows. The pre-fix
loaders were the source of the real-finch autonomy bug noted in
docs/finch_blk445_syllC_arneodo (MEMORY: finch-blk445-syllC-arneodo).

- `load_voc_windows`: legacy mid-voc window starting `start_offset_ms` after onset.
  Used by the rescaled-autonomy selection mode.
- `load_voc_windows_coldstart`: silence lead-in + full vocalization. The IC at the
  segment start is near-silence, so the autonomy_score exercises the cold-start
  ignition path the deployed pipeline cares about. Used by `--cold-start-selection`.
"""

import glob
import os

import numpy as np
from scipy.io import wavfile


def load_voc_windows(data_dir, n_vocs, start_offset_ms, n):
    """Held-out sustained vocalization windows (start `start_offset_ms` after onset)."""
    segs = []
    sr = None
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        if af.dtype == np.int16:
            af = af / -np.iinfo(af.dtype).min
        af = af.astype(np.float64)
        onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        s = int(onoffs[0][0] * sr) + int(start_offset_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs, sr


def load_voc_windows_coldstart(data_dir, n_vocs, silence_pad_samples):
    """Held-out cold-start windows: `silence_pad_samples` lead-in + full vocalization.

    Matches the 50 ms lead-in convention (2000 samples at 40 kHz) used by the existing
    finchsim/scan_seed_amp_coldstart pipeline. Per-voc lengths vary, so segments are
    trimmed to the shortest common length so they stack uniformly.
    """
    raw = []
    sr = None
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        if af.dtype == np.int16:
            af = af / -np.iinfo(af.dtype).min
        af = af.astype(np.float64)
        onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        on_i = int(round(onoffs[0][0] * sr))
        off_i = int(round(onoffs[0][1] * sr))
        start = max(0, on_i - silence_pad_samples)
        raw.append(af[start:off_i])
    if not raw:
        return [], sr
    L = min(len(s) for s in raw)
    segs = [s[:L] for s in raw]
    return segs, sr
