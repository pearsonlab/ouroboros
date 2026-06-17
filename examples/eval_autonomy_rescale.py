"""
Evaluate autonomous reconstruction with the rescale baseline made explicit.

For each labeled poly checkpoint dir, runs DETERMINISTIC autonomous integration on held-out
gabo_p9 windows and reports three numbers:

  - raw autonomy   = spec_corr - amp_pen - pitch_pen        (train.eval.autonomy_score's metric)
  - rescale bound  = spec_corr - pitch_pen                   (amplitude rescaled to target RMS;
                     amp_pen -> 0, spec_corr is scale-invariant). This is the cheap baseline the
                     spectral/envelope fine-tune must BEAT to be worth it.
  - env match      = corr + relL1 of the time-varying loudness envelope, AFTER rescaling auto to
                     target RMS -- i.e. the loudness SHAPE that a single global rescale CANNOT fix.
                     This is where a successful envelope fine-tune should show a distinctive gain.

    python -m examples.eval_autonomy_rescale --data-dir data500/gabo_p9 --n-vocs 6 \
        --models "pre=poly_alpha_ms/alpha_off_seed0/poly" "post=poly_ftspec_seed0/poly"
"""

import argparse
import glob
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

from train.train import load_model
from train.eval import integrate_poly_autonomous, correct


def load_voc_windows(data_dir, n_vocs, off_ms, n):
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
        s = int(on * sr) + int(off_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs, sr


def envelope(x, sr, env_ms=2.0):
    """Gaussian low-pass of |x| (numpy)."""
    dt = 1.0 / sr
    sigma = (env_ms / 1e3) / dt
    radius = max(1, int(round(3 * sigma)))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    xr = np.pad(np.abs(x), radius, mode="reflect")
    return np.convolve(xr, k, mode="valid")


def score_model(model, segs, sr, fmax=8000.0):
    """
    Deterministic autonomous reconstruction metrics on held-out windows. Returns a dict:
      raw     = spec_corr - amp_pen - pitch_pen                  (autonomy_score metric)
      rescale = spec_corr - pitch_pen                            (amplitude gauge-fixed; bar to beat)
      spec, amp_pen, pitch_pen                                   (means)
      envcorr, envL1                                             (loudness-trajectory match after rescale)
    """
    dt = model.tau

    def logpsd(x):
        f, Pxx = welch(x - np.mean(x), fs=sr, nperseg=min(1024, len(x)))
        m = f <= fmax
        return np.log(Pxx[m] + 1e-20)

    def peak(x):
        f, Pxx = welch(x - np.mean(x), fs=sr, nperseg=min(1024, len(x)))
        Pxx[0] = 0
        return float(f[np.argmax(Pxx)])

    raws, resc, specs, amps, pits, ecorr, eL1 = ([] for _ in range(7))
    for seg in segs:
        tgt = correct(np.asarray(seg, dtype=np.float64))
        auto = integrate_poly_autonomous(model, seg, dt, noise_sd=0.0, detrend=True, verbose=False)
        n = min(len(tgt), len(auto))
        tgt, auto = tgt[:n], auto[:n]
        st, sa = np.nanstd(tgt), np.nanstd(auto)
        if (not np.isfinite(auto).all()) or sa < 1e-9:
            raws.append(-5.0); resc.append(-5.0)
            continue
        sc = float(np.corrcoef(logpsd(tgt), logpsd(auto))[0, 1])
        amp = abs(np.log((sa + 1e-12) / (st + 1e-12)))
        pit = abs(np.log((peak(auto) + 1e-9) / (peak(tgt) + 1e-9)))
        raws.append(sc - amp - pit); resc.append(sc - pit)
        specs.append(sc); amps.append(amp); pits.append(pit)
        ar = auto * (st / (sa + 1e-12))
        et, ea = envelope(tgt, sr), envelope(ar, sr)
        ecorr.append(float(np.corrcoef(et, ea)[0, 1]))
        eL1.append(float(np.mean(np.abs(ea - et)) / (np.mean(et) + 1e-9)))
    return {"raw": float(np.mean(raws)), "rescale": float(np.mean(resc)),
            "spec": float(np.mean(specs)) if specs else float("nan"),
            "amp_pen": float(np.mean(amps)) if amps else float("nan"),
            "pitch_pen": float(np.mean(pits)) if pits else float("nan"),
            "envcorr": float(np.mean(ecorr)) if ecorr else float("nan"),
            "envL1": float(np.mean(eL1)) if eL1 else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data500/gabo_p9")
    p.add_argument("--n-vocs", type=int, default=6)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--models", nargs="+", required=True, help='label=dir entries')
    args = p.parse_args()

    segs, sr = load_voc_windows(args.data_dir, args.n_vocs, args.start_offset_ms, args.auto_n)
    print(f"{len(segs)} held-out vocs from {args.data_dir} (n={args.auto_n}, sr={sr})\n", flush=True)

    print(f"{'label':14s} {'raw':>7s} {'rescale':>8s} {'spec':>6s} {'amp_pen':>8s} "
          f"{'pitch':>6s} {'envcorr':>8s} {'envL1':>7s}", flush=True)
    for entry in args.models:
        label, d = entry.split("=", 1)
        model, _, _, _ = load_model(d)
        model.eval()
        r = score_model(model, segs, sr, fmax=args.fmax)
        print(f"{label:14s} {r['raw']:+7.3f} {r['rescale']:+8.3f} {r['spec']:6.3f} "
              f"{r['amp_pen']:8.3f} {r['pitch_pen']:6.3f} {r['envcorr']:8.3f} {r['envL1']:7.3f}",
              flush=True)


if __name__ == "__main__":
    main()
