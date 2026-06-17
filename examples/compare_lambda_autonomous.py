"""
Compare lambda-sweep poly Ouroboros models on AUTONOMOUS generation (not teacher-forced R^2).

For each lambda checkpoint, runs closed-loop integration on one or more held-out vocalizations
(deterministic and noise-driven) and plots, vs lambda:
  - amplitude (std): deterministic & noise-driven, vs target
  - decay ratio (2nd-half / 1st-half std): >1 grows, <1 decays, ~1 self-sustained
  - pitch: deterministic resonance & noise-driven, vs target

Use --n-vocs 1 for a quick draft, larger for a robust (averaged) comparison.

Run from the repo root:
    python -m examples.compare_lambda_autonomous --sweep-dir poly_lp_sweep \
        --data-glob 'data500/gabo_p*' --n-vocs 5 --n 2500 --noise-sd 2.4e-4
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

from train.train import load_model
from train.eval import integrate_poly_autonomous, correct

plt.rcParams["text.usetex"] = False


def pk(x, sr):
    x = np.nan_to_num(x)
    f, P = welch(x - x.mean(), fs=sr, nperseg=min(1024, len(x)))
    P[0] = 0
    return float(f[np.argmax(P)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", default="poly_lp_sweep")
    p.add_argument("--data-glob", default="data500/gabo_p*")
    p.add_argument("--n-vocs", type=int, default=1)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--n", type=int, default=2500, help="samples to integrate per voc")
    p.add_argument("--noise-sd", type=float, default=2.4e-4)
    p.add_argument("--out-dir", default="poly_lp_sweep")
    args = p.parse_args()

    lam_dirs = sorted(glob.glob(os.path.join(args.sweep_dir, "lam_*")),
                      key=lambda d: float(d.split("lam_")[-1]))
    data_dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    # one held-out vocalization per data dir (gabo_artificial_0), up to n_vocs
    vocs = []
    for d in data_dirs[: args.n_vocs]:
        wav = os.path.join(d, "gabo_artificial_0.wav")
        seg_txt = os.path.join(d, "gabo_artificial_0.txt")
        if os.path.isfile(wav):
            vocs.append((wav, seg_txt))
    print(f"{len(lam_dirs)} lambdas x {len(vocs)} vocs; n={args.n} samples", flush=True)

    def load_seg(wav, seg_txt, sr_hint=40000):
        sr, af = wavfile.read(wav)
        af = af.astype(float)
        on = np.atleast_2d(np.loadtxt(seg_txt))[0][0]
        s = int(on * sr) + int(args.start_offset_ms / 1e3 * sr)
        return sr, af[s:s + args.n]

    lams, det_std, det_decay, det_res, nz_std, nz_pitch, tgt_std, tgt_pitch = ([] for _ in range(8))
    for ld in lam_dirs:
        lam = float(ld.split("lam_")[-1])
        model, _, _, _ = load_model(ld); model.eval(); dt = model.tau
        ds, dd, dr, ns, npi, ts, tp = [], [], [], [], [], [], []
        for wav, seg_txt in vocs:
            sr, seg = load_seg(wav, seg_txt)
            tgt = correct(seg)
            det = integrate_poly_autonomous(model, seg, dt, noise_sd=0.0, detrend=True, verbose=False)
            nz = integrate_poly_autonomous(model, seg, dt, noise_sd=args.noise_sd, seed=0, detrend=True, verbose=False)
            h1, h2 = det[: len(det) // 2], det[len(det) // 2:]
            ds.append(np.nanstd(det)); dd.append(np.nanstd(h2) / (np.nanstd(h1) + 1e-9))
            dr.append(pk(h1, sr)); ns.append(np.nanstd(nz)); npi.append(pk(nz, sr))
            ts.append(tgt.std()); tp.append(pk(tgt, sr))
        lams.append(lam)
        det_std.append(ds); det_decay.append(dd); det_res.append(dr)
        nz_std.append(ns); nz_pitch.append(npi); tgt_std.append(ts); tgt_pitch.append(tp)
        print(f"lam={lam:.3f}: det std={np.mean(ds):.4f} decay={np.mean(dd):.2f} res={np.mean(dr):.0f}Hz "
              f"| noise std={np.mean(ns):.4f} pitch={np.mean(npi):.0f}Hz", flush=True)

    lams = np.array(lams)
    mean = lambda L: np.array([np.mean(v) for v in L])
    sem = lambda L: np.array([np.std(v) / max(1, np.sqrt(len(v))) for v in L])
    tgt_s = np.mean([np.mean(v) for v in tgt_std]); tgt_p = np.mean([np.mean(v) for v in tgt_pitch])

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    a1.errorbar(lams, mean(det_std), sem(det_std), marker="o", color="tab:blue", label="deterministic")
    a1.errorbar(lams, mean(nz_std), sem(nz_std), marker="s", color="tab:green", label=f"noise-driven (σ={args.noise_sd:g})")
    a1.axhline(tgt_s, ls="--", color="tab:orange", label="target")
    a1.set_ylabel("autonomous amplitude (std)"); a1.set_yscale("log"); a1.legend(fontsize=8)
    a1.set_title("Autonomous generation vs kernel-weight λ (low-pass poly)")

    a2.errorbar(lams, mean(det_decay), sem(det_decay), marker="o", color="tab:blue")
    a2.axhline(1.0, ls="--", color="0.5", label="self-sustained (=1)")
    a2.set_ylabel("decay ratio (2nd/1st half)"); a2.legend(fontsize=8)

    a3.errorbar(lams, mean(det_res), sem(det_res), marker="o", color="tab:blue", label="det. resonance")
    a3.errorbar(lams, mean(nz_pitch), sem(nz_pitch), marker="s", color="tab:green", label="noise-driven pitch")
    a3.axhline(tgt_p, ls="--", color="tab:orange", label="target pitch")
    a3.set_ylabel("pitch (Hz)"); a3.set_xlabel("kernel-weight λ"); a3.legend(fontsize=8)

    fig.tight_layout()
    tag = "draft" if args.n_vocs == 1 else f"{args.n_vocs}voc"
    out = os.path.join(os.path.abspath(args.out_dir), f"lambda_autonomous_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out}", flush=True)


if __name__ == "__main__":
    main()
