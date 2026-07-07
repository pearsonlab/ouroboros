"""One-shot diagnostic for the live spectral-rollout training run.

Reads the TB events file, checks for a new checkpoint, runs cold-start
autonomy_score on the latest checkpoint if it hasn't been scored yet, and emits
a single line to stdout if there is something actionable to report.

Signals (priority order):
- DEAD: training process gone.
- NAN_SPIKE: nan_skip counter jumped by > 50 since last poll.
- LOSS_SPIKE: spec_recent5 / spec_prev5 > 2.0.
- NEW_CKPT: a fresh checkpoint was found; ran autonomy_score on it.
- PLATEAU: val_autonomy hasn't improved by > 0.02 across the last 2 checkpoints.
           Falls back to "best_spec hasn't improved in 8 polls" before any
           checkpoint exists.
- HEARTBEAT: nothing actionable, but every HEARTBEAT_EVERY polls.
- UNCHANGED: not printed; monitor silent.

The autonomy check is done in a SUBPROCESS so the main monitor process never holds
a CUDA context (avoids contention with the live training run -- subprocess will
queue waiting for a GPU slot for ~30-60s during a forward of the autonomous
integrator, finish, and release).
"""

import glob
import json
import argparse
import os
import signal
import subprocess
import sys

import numpy as np


_DEFAULT_STATE = "/home/pearson/.claude/jobs/712703d8/tmp/spectral_monitor/state.json"
_DEFAULT_SEED = "/home/pearson/code/ouroboros-spectral/poly_spectral_day85/seed0"
_DEFAULT_TRAIN_PATTERN = "train_poly_spectral_blk445"

# These three module-level names are populated at __main__ time from CLI args so the
# rest of the script can keep referring to them as constants.
STATE_PATH = _DEFAULT_STATE
SEED_DIR = _DEFAULT_SEED
TRAIN_PATTERN = _DEFAULT_TRAIN_PATTERN
HEARTBEAT_EVERY = 4              # heartbeat every 4 polls = 1 hour
PLATEAU_NO_IMPROVE_POLLS = 8     # 8 polls = 2 hours with no new spec best
PLATEAU_IMPROVE_MARGIN = 0.02    # "new best" requires beating prior best by 2%
AUTONOMY_PLATEAU_CKPTS = 2       # 2 consecutive ckpts with no autonomy improvement
AUTONOMY_IMPROVE_MARGIN = 0.02   # required autonomy improvement per ckpt


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_nan_skips": 0, "poll_count": 0, "best_spec_recent5": None,
            "best_poll": 0, "last_ckpt_epoch": -1, "last_val_autonomy": None,
            "best_val_autonomy": None, "ckpts_since_best_autonomy": 0,
            "last_plateau_emit_poll": 0, "history": []}


def save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def process_alive():
    # The CLI lets the user disambiguate two concurrent runs by --train-pattern --
    # the live run matches on its --out-dir basename, the env run matches on a
    # different basename. Falls back to the legacy "train_poly_spectral_blk445"
    # pattern if the user didn't override (single-run case).
    out = subprocess.run(
        ["pgrep", "-f", TRAIN_PATTERN],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def latest_event_file():
    """Find the trainer's events file. Multiple events files can live under SEED_DIR
    (the monitor itself opens a SummaryWriter to write audio/figures/Val scalars per
    ckpt). We want the file with the training scalar curves -- iterate newest to
    oldest and pick the first one whose tag list includes 'Loss/spec'."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    evs = sorted(glob.glob(os.path.join(SEED_DIR, "events.out.tfevents.*")))
    if not evs:
        return None
    for ev_path in reversed(evs):
        try:
            ea = EventAccumulator(ev_path, size_guidance={"scalars": 1})
            ea.Reload()
            if "Loss/spec" in ea.Tags().get("scalars", []):
                return ev_path
        except Exception:
            continue
    return evs[-1]


_BPE_CACHE = {}

def detect_batches_per_epoch(fallback: int = 750) -> int:
    """Find 'batches_per_epoch=N' in <run_dir>_train.log (the sibling-named log the
    entry script writes). Cached per SEED_DIR so we only parse the log once. Falls
    back to `fallback` if anything goes wrong (legacy runs that predate the print)."""
    if SEED_DIR in _BPE_CACHE:
        return _BPE_CACHE[SEED_DIR]
    bpe = fallback
    try:
        run_dir = os.path.dirname(SEED_DIR.rstrip("/"))
        parent = os.path.dirname(run_dir)
        # The launch script writes <run_dir>/train.log; older runs used a sibling
        # <parent>/<runname>_train.log. Try both.
        candidates = [
            os.path.join(run_dir, "train.log"),
            os.path.join(parent, os.path.basename(run_dir) + "_train.log"),
        ]
        import re
        for log_path in candidates:
            if not os.path.exists(log_path):
                continue
            found = False
            with open(log_path) as f:
                # the line is small + near the top; scan up to first ~200 lines.
                for i, line in enumerate(f):
                    if i > 200:
                        break
                    m = re.search(r"batches_per_epoch=(\d+)", line)
                    if m:
                        bpe = int(m.group(1))
                        found = True
                        break
            if found:
                break
    except Exception:
        pass
    _BPE_CACHE[SEED_DIR] = bpe
    return bpe


def read_loss(ev_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(ev_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in ("Loss/spec", "Loss/tf", "Train/H"):
        if t in tags:
            sc = ea.Scalars(t)
            out[t] = np.array([s.value for s in sc])
    nan = ea.Scalars("Loss/nan_skip") if "Loss/nan_skip" in tags else []
    out["nan_skips"] = int(sum(s.value for s in nan))
    return out


def window_means(vals, w=100, n_windows=20):
    if len(vals) < w * n_windows:
        n_windows = len(vals) // w
    if n_windows == 0:
        return np.array([])
    means = []
    for i in range(n_windows):
        end = len(vals) - i * w
        start = end - w
        if start < 0:
            break
        means.append(vals[start:end].mean())
    return np.array(list(reversed(means)))


def _ckpt_epoch(p):
    return int(os.path.basename(p).split("_")[1].split(".")[0])


def latest_checkpoint():
    # Sort by parsed epoch number, NOT lexicographically -- checkpoint_9.tar would
    # otherwise sort after checkpoint_10.tar and silently mask all double-digit ckpts.
    ckpts = glob.glob(os.path.join(SEED_DIR, "checkpoint_*.tar"))
    if not ckpts:
        return None, -1
    ckpts.sort(key=_ckpt_epoch)
    latest = ckpts[-1]
    return latest, _ckpt_epoch(latest)


def unscored_checkpoints(last_scored_epoch):
    """Return [(path, epoch), ...] for ckpts whose epoch > last_scored_epoch,
    sorted oldest-first. Used so that if multiple ckpts arrived between polls,
    each one gets its own val score and spectrogram figure on TB."""
    ckpts = glob.glob(os.path.join(SEED_DIR, "checkpoint_*.tar"))
    eps = sorted({_ckpt_epoch(p) for p in ckpts})
    eps = [e for e in eps if e > last_scored_epoch]
    return [(os.path.join(SEED_DIR, f"checkpoint_{e}.tar"), e) for e in eps]


AUTONOMY_SNIPPET = r"""
import sys, os, glob, json, warnings
import numpy as np
# train.* / model.* resolve from PYTHONPATH (the sidecar driver sets it to THIS branch's
# root) so the monitor scores with this branch's drive_noise-aware load_model + tract/
# envelope model, rather than a sibling worktree's pre-drive_noise code.
import torch
from scipy.io import wavfile
from train.train import load_model
from train.eval import autonomy_score

DATA = os.environ.get('MONITOR_DATA',
                      os.path.expanduser('~/ouroboros_data/blk445_syllC/day85'))
STRATIFY_SEP = os.environ.get('MONITOR_STRATIFY_SEP', '')
SR = 44100
DT = 1.0 / SR
N_VAL = int(os.environ.get('MONITOR_N_VAL', '8'))
# Number of vocs we render full spectrograms+audio for in TB. Bump via env when
# val coverage grows (e.g. multi-syllable runs want ~1 per (bird, syllable)).
N_TB_VOCS = int(os.environ.get('MONITOR_N_TB_VOCS', '4'))
TB_BATCHES_PER_EPOCH = None  # set from env so the step on the audio/figure aligns
                              # with the train scalars' x-axis (which is global batch idx)

wavs = sorted(glob.glob(os.path.join(DATA, '*.wav')))
rng = np.random.default_rng(1234)
idx = np.arange(len(wavs))
rng.shuffle(idx)
n_test = max(1, int(round(0.1 * len(wavs))))
n_val = max(1, int(round(0.1 * len(wavs))))
val_i = idx[n_test:n_test + n_val]
val_wavs = [wavs[i] for i in val_i]
# Stratify val selection by `{prefix}<sep>` so the scored vocs cover every group
# (e.g. each (bird, syllable) prefix in the multi-syllable run).
if STRATIFY_SEP:
    by_g = {}
    for w in val_wavs:
        stem = os.path.basename(w)
        g = stem.split(STRATIFY_SEP, 1)[0] if STRATIFY_SEP in stem else '__none__'
        by_g.setdefault(g, []).append(w)
    groups = sorted(by_g)
    per = max(1, N_VAL // max(1, len(groups)))
    picked = []
    for g in groups:
        picked.extend(by_g[g][:per])
    val_wavs = picked[:N_VAL]

# Coldstart voc span: MONITOR_COLDSTART_DURATION_MS extends each voc to cover
# the SECOND/THIRD/... annotation rather than stopping at the first offset, so
# the val task includes multi-syllable continuation. Each voc spans from
# (first-onset - 2000) to the offset of the LAST annotation whose end is within
# `target_ms` of the first onset. Defaults to 0 (legacy: single-syllable).
TARGET_MS = float(os.environ.get('MONITOR_COLDSTART_DURATION_MS', '0'))
raw, sr = [], None
for wav in val_wavs[:N_VAL]:
    sr, af = wavfile.read(wav)
    if af.dtype == np.int16:
        af = af / -np.iinfo(af.dtype).min
    af = af.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        onoffs = np.atleast_2d(np.loadtxt(wav.replace('.wav', '.txt')))
    on_i = int(round(onoffs[0][0] * sr))
    if TARGET_MS > 0:
        # Fixed-length window from (first_onset - 2000) through target_ms past
        # first_onset, so all vocs end up the same length regardless of how
        # densely syllables are packed in the source recording. Silence/song-
        # structure within the window is part of what the model should produce.
        target_len = 2000 + int(round(TARGET_MS / 1e3 * sr))
        start = max(0, on_i - 2000)
        raw.append(af[start:start + target_len])
    else:
        off_i = int(round(onoffs[0][1] * sr))
        raw.append(af[max(0, on_i - 2000):off_i])
# Keep each voc at its natural length (different syllables have different
# durations); autonomy_score handles per-voc lengths via min(len(tgt), len(auto))
# per integration. The plot code below uses max(len) across vocs to set a
# common x-axis so short syllables aren't visually stretched.
val_vocs = list(raw)

ckpt_path = sys.argv[1]
ckpt_dir = os.path.dirname(ckpt_path)
# Honor MONITOR_DEVICE env var so the polling loop can ask for CPU evaluation when
# the trainer is holding all of GPU memory and the monitor would otherwise OOM.
_dev = os.environ.get("MONITOR_DEVICE", "cuda")
model, _, _, _ = load_model(ckpt_dir, device=_dev)
model.eval()
# Noise-forcing models make sound only WITH the OU forcing on; the deterministic (noise-off)
# reconstruction is the dead backbone. So render/score at a noise_gain that matches what the
# trainer used at this checkpoint. MONITOR_NOISE_GAIN overrides (float); "auto" (default)
# derives the per-checkpoint gain from the ramp schedule + this checkpoint's global step, so
# each epoch's spectrogram shows the actual training-time output. No-op for non-noise models.
_ng_env = os.environ.get("MONITOR_NOISE_GAIN", "auto")
if _ng_env != "auto":
    _noise_gain = float(_ng_env)
else:
    try:
        # step_override is ckpt_epoch*bpe (START of the epoch); the checkpoint is saved AFTER
        # that epoch trained, so add one epoch of steps to get the gain the model was actually
        # saved with (end of epoch). ckpt_4 -> step 3750 -> gain 0 (last deterministic epoch);
        # ckpt_5 -> 4500 -> gain 0.2 (noise ramping); ckpt_10 -> gain 1.0.
        _step = float(os.environ["MONITOR_STEP_OVERRIDE"]) + float(os.environ.get("MONITOR_BPE", "0"))
        _nstart = float(os.environ.get("MONITOR_NOISE_START_STEP", "0"))
        _nwarm = float(os.environ.get("MONITOR_NOISE_WARMUP_STEPS", "0"))
        _noise_gain = 0.0 if _step < _nstart else (
            min(1.0, (_step - _nstart) / _nwarm) if _nwarm > 0 else 1.0)
    except (KeyError, ValueError, TypeError):
        _noise_gain = 1.0
# Oscillator warmup gain: same ramp-from-step logic as noise, so warmup-epoch checkpoints are
# scored/rendered with the oscillator OFF (only rumble+noise), matching what was trained.
try:
    _ostep = float(os.environ["MONITOR_STEP_OVERRIDE"]) + float(os.environ.get("MONITOR_BPE", "0"))
    _ostart = float(os.environ.get("MONITOR_OSC_START_STEP", "0"))
    _owarm = float(os.environ.get("MONITOR_OSC_WARMUP_STEPS", "0"))
    _osc_gain = 0.0 if _ostep < _ostart else (
        min(1.0, (_ostep - _ostart) / _owarm) if _owarm > 0 else 1.0)
except (KeyError, ValueError, TypeError):
    _osc_gain = 1.0
with torch.no_grad():
    score, _, bd, trajs = autonomy_score(
        model, val_vocs, DT, rescale=False, cold_start=True, return_trajectories=True,
        noise_gain=_noise_gain, osc_gain=_osc_gain,
    )

# Persist signed amp_pen alongside the offline cache the loss panels reads.
# autonomy_score now puts signed_amp_per_voc + signed_amp_mean in the breakdown so this
# is essentially free -- the integration already happened above.
cache_path = os.path.join(ckpt_dir, 'signed_amp_pen.json')
cache = {}
if os.path.exists(cache_path):
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except Exception:
        cache = {}
ep = int(os.path.basename(ckpt_path).split('_')[1].split('.')[0])
cache[str(ep)] = {
    'per_voc': [float(v) for v in bd.get('signed_amp_per_voc', [])],
    'mean': float(bd.get('signed_amp_mean', float('nan'))),
}
with open(cache_path, 'w') as f:
    json.dump(cache, f, indent=2)

# Log val audio + spectrogram figures to the run's TB events file. We open a SECOND
# SummaryWriter pointed at ckpt_dir (the trainer's writer is the first); TB merges
# the events files in a logdir, so audio and scalars appear under the same run.
# Step value uses an estimated global-batch index so audio/figures align with the
# train-loop scalar curves on the x-axis (rather than landing at step=0 every poll).
try:
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa  # mel filterbank for the spectrogram panels (already a codebase dep)
    bpe_env = os.environ.get('MONITOR_BPE')
    bpe = int(bpe_env) if (bpe_env and bpe_env.isdigit()) else 1
    # epoch -> step alignment: trainer's writer.add_scalar uses idx = global batch index,
    # so checkpoint_N (saved at the END of epoch N) sits at global step (N + 1) * bpe - 1.
    # Use that so the audio/figure/Val points land exactly on the train-loss curve for the
    # same model. (Resume runs complicate this -- idx resets to 0 even though the epoch
    # counter starts at start_epoch -- but from-scratch runs are exact.)
    # MONITOR_LOG_DIR overrides where Val/* scalars + spectrograms are written. Needed
    # when the sidecar stages an inflight save into a temp dir (ckpt_dir then points at
    # the temp dir that gets cleaned up after subprocess exit, taking the SummaryWriter
    # output with it). For per-epoch saves the override is omitted and TB lands in the
    # ckpt's own seed dir as before.
    _log_dir = os.environ.get("MONITOR_LOG_DIR", ckpt_dir)
    sw = SummaryWriter(log_dir=_log_dir)
    # MONITOR_STEP_OVERRIDE lets the sidecar set an arbitrary TB step (used for the
    # inflight ckpts which don't have a clean epoch number — the sidecar reads the
    # trainer's latest Loss/spec step and passes it here so the Val/ curves align).
    _step_override = os.environ.get("MONITOR_STEP_OVERRIDE")
    # step_override is ckpt_epoch*bpe (start of epoch); add bpe-1 to land on the epoch's LAST
    # batch index = (ckpt_epoch+1)*bpe-1, matching the noise-gain calc above and the trainer's
    # end-of-epoch scalar step. (No override -> same formula from the parsed ep.)
    step = (int(_step_override) + bpe - 1) if _step_override else (ep + 1) * bpe - 1
    # Val metrics (cold-start, rescale=False -- the same numbers the seed-CV uses).
    # Putting them on the same x-axis as the train scalars lets you compare e.g.
    # LossW/logm directly to Val/signed_amp_mean.
    sw.add_scalar("Val/autonomy",        float(score),                          step)
    sw.add_scalar("Val/spec_corr",       float(bd.get('spec_corr', float('nan'))),     step)
    sw.add_scalar("Val/amp_pen",         float(bd.get('amp_pen', float('nan'))),       step)
    sw.add_scalar("Val/pitch_pen",       float(bd.get('pitch_pen', float('nan'))),     step)
    sw.add_scalar("Val/bounded_frac",    float(bd.get('bounded_frac', float('nan'))),  step)
    sw.add_scalar("Val/signed_amp_mean", float(bd.get('signed_amp_mean', float('nan'))), step)
    # Shared x-axis upper limit (max voc duration across the rendered set) so
    # short syllables aren't visually stretched to fill the same plot width as
    # long ones. Each per-voc panel still shows its own content; the extra space
    # on the right of short ones is intentional.
    _max_ms = max((len(t[0]) for t in trajs[:N_TB_VOCS] if len(t) >= 1), default=1) / SR * 1000
    _psd = []  # per-voc (target, auto, noise, rumble) for the mel-PSD panel built after the loop
    for i in range(min(N_TB_VOCS, len(trajs))):
        # autonomy_score returns (tgt, auto, env, src, drives) when return_trajectories=True;
        # tolerate older 2/3/4-tuple shapes so this script works against in-flight runs
        # that haven't restarted yet. src = pre-tract, pre-envelope RK4 oscillator output.
        # drives = {omega, gamma, alpha} per timestep (alpha = constant forcing term).
        traj = trajs[i]
        if len(traj) == 5:
            tgt_n, auto_n, env_n, src_n, drives = traj
        elif len(traj) == 4:
            tgt_n, auto_n, env_n, src_n = traj
            drives = None
        elif len(traj) == 3:
            tgt_n, auto_n, env_n = traj
            src_n = drives = None
        else:
            tgt_n, auto_n = traj
            env_n = src_n = drives = None
        rumble_n = drives.get('rumble') if drives is not None else None
        s_tgt = float(np.nanstd(tgt_n) + 1e-12)
        s_auto = float(np.nanstd(auto_n) + 1e-12)
        auto_rescaled = auto_n * (s_tgt / s_auto)  # match target RMS for listening / display
        # Same gauge factor for the envelope so it sits on the rescaled-auto axis.
        env_rescaled = env_n * (s_tgt / s_auto) if env_n is not None else None
        # Audio (TB SummaryWriter expects (N,) or (1, T) float in [-1, 1])
        sw.add_audio(f"audio/voc{i}_target", torch.tensor(tgt_n / (np.max(np.abs(tgt_n)) + 1e-12),
                                                          dtype=torch.float32), step, sample_rate=SR)
        sw.add_audio(f"audio/voc{i}_auto",   torch.tensor(auto_rescaled / (np.max(np.abs(auto_rescaled)) + 1e-12),
                                                          dtype=torch.float32), step, sample_rate=SR)
        # Specgram figure rows:
        #   row 0: target waveform | target spectrogram
        #   row 1: auto (post-tract) waveform | auto spectrogram
        #   row 2: source (pre-tract, pre-env) waveform | source spectrogram (if src_n)
        #   row 3: rumble (band-limited LF branch) waveform | spectrogram (if rumble_n)
        #   row 4: drives panel -- omega^2 / gamma / alpha time series (if drives)
        n_rows = (2 + (1 if src_n is not None else 0)
                  + (1 if rumble_n is not None else 0) + (1 if drives is not None else 0))
        fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2 * n_rows),
                                 gridspec_kw={'width_ratios': [1, 2]})
        # Mel-spaced spectrograms (default): the low/mid vocal structure is what matters
        # (birdsong harmonics live < ~8 kHz), so map onto a mel filterbank via librosa (already
        # a codebase dep -- visualization/model_vis.py). n_fft=1024 gives enough low-frequency
        # resolution that the bottom mel bands aren't empty (n_fft=512 striped). Colour is dB
        # relative to the TARGET's peak MEL POWER (one shared reference for all rows), so the
        # target peaks at 0 dB and quieter rows sit visibly lower on the same faithful scale.
        _n_mels, _fmax, _mel_nfft, _mel_hop = 128, 16000.0, 1024, 256
        def _mel_pow(x):
            return librosa.feature.melspectrogram(
                y=np.ascontiguousarray(x, dtype=np.float32), sr=SR, n_fft=_mel_nfft,
                hop_length=_mel_hop, n_mels=_n_mels, fmax=_fmax, power=2.0)  # (nmels, ntime)
        _mel_hz = librosa.mel_frequencies(n_mels=_n_mels, fmax=_fmax)
        _spec_ref = float(np.nanmax(_mel_pow(tgt_n)) + 1e-20)
        _spec_db_floor = -80.0
        # Lock all waveform panels to the target's y-range so each panel is
        # directly comparable in scale. The envelope shape (positive) is rescaled
        # to fit the same range so its time-course is visible against the carrier.
        tgt_peak = float(np.nanmax(np.abs(tgt_n)) + 1e-12)
        wf_ylim = (-1.05 * tgt_peak, 1.05 * tgt_peak)
        # Panels show RAW signals. The y-axis is locked to the target's peak so
        # the auto waveform's quietness vs the target is directly visible; the
        # spectrogram colormap is fixed (vmin=-8, vmax=-2) so quiet auto rollouts
        # also dim in the spectrogram. No rescaling anywhere -- you read the real
        # amplitude gap directly. Source row is locked to the same target y-range
        # too; that lets you see immediately how much amplitude the tract+envelope
        # add or subtract.
        row_specs = [("target", tgt_n,  tgt_n,  "tab:orange"),
                     ("auto",   auto_n, auto_n, "tab:green")]
        if src_n is not None:
            # Gate the source DISPLAY by osc_gain too, so during warmup (osc_gain=0) the source row
            # reads ~0 -- reflecting that the oscillator isn't contributing to the output yet.
            src_g = src_n * _osc_gain
            row_specs.append(("source", src_g, src_g, "tab:purple"))
        if rumble_n is not None:
            row_specs.append(("rumble", rumble_n, rumble_n, "tab:blue"))
        for row, (label, wf_x, spec_x, color) in enumerate(row_specs):
            t_ms = np.arange(len(wf_x)) / SR * 1000
            axes[row, 0].plot(t_ms, wf_x, color=color, lw=0.6); axes[row, 0].set_ylabel(label)
            axes[row, 0].set_xlim([0, _max_ms])
            axes[row, 0].set_ylim(wf_ylim)
            # Overlay the envelope on the auto panel as +/- bounds. Rescale env_n to
            # peak at the target waveform's peak so its shape is visible in the
            # target-locked y-axis (just a visual gauge — the absolute envelope value
            # is meaningless here; only shape matters).
            if row == 1 and env_n is not None and len(env_n) == len(wf_x):
                env_peak = float(np.nanmax(np.abs(env_n)) + 1e-12)
                env_shape = env_n * (tgt_peak / env_peak)
                axes[row, 0].plot(t_ms,  env_shape, color="k", lw=0.6, linestyle="--", label="env (shape only)")
                axes[row, 0].plot(t_ms, -env_shape, color="k", lw=0.6, linestyle="--")
                axes[row, 0].legend(loc="upper right", fontsize=7, framealpha=0.6)
            Mp = _mel_pow(spec_x)                                     # (nmels, ntime) mel power
            M_db = 10.0 * np.log10(np.maximum(Mp / _spec_ref, 1e-8))  # dB re target peak mel power
            axes[row, 1].imshow(M_db, aspect='auto', origin='lower',
                                 extent=[0, t_ms[-1] if len(t_ms) else 1, 0, _n_mels],
                                 vmin=_spec_db_floor, vmax=0.0, cmap='magma')
            axes[row, 1].set_xlim([0, _max_ms])
            _yt = np.linspace(0, _n_mels - 1, 6).astype(int)          # mel-spaced Hz tick labels
            axes[row, 1].set_yticks(_yt)
            axes[row, 1].set_yticklabels([str(int(_mel_hz[k])) for k in _yt], fontsize=7)
        # Drives panel: time series of omega^2, gamma, alpha drawn across BOTH
        # columns of the next row, with the constant terms ALPHA and GAMMA on the
        # left y-axis and OMEGA^2 on a twin right axis (it's on a very different
        # scale). Helps see what the encoded dynamics are doing at each timestep.
        if drives is not None:
            drives_row = len(row_specs)
            ax_d = axes[drives_row, 0]
            ax_d2 = axes[drives_row, 1]
            ms_axis = np.arange(len(drives['omega'])) / SR * 1000
            # Left panel: for a harmonic-plus-noise model, show the FILTERED-NOISE waveform (the
            # term added to the tract output to produce the final waveform), y-locked to the
            # TARGET scale like the other left-column waveforms. The drives are still shown on the
            # right panel. For models without the noise branch, fall back to the drives here.
            if drives.get('noise') is not None:
                nz = drives['noise']
                t_ms_n = np.arange(len(nz)) / SR * 1000
                ax_d.plot(t_ms_n, nz, color='tab:gray', lw=0.6)
                ax_d.set_ylabel('filtered noise')
                ax_d.set_xlim([0, _max_ms])
                ax_d.set_ylim(wf_ylim)          # lock to target scale (as with the other left-col waveforms)
            else:
                # No noise branch: this panel is reserved for the filtered-noise waveform, so
                # with noise off leave it blank + annotated (the drives are on the right panel).
                ax_d.text(0.5, 0.5, 'no noise branch', transform=ax_d.transAxes,
                          ha='center', va='center', color='gray', fontsize=9, style='italic')
                ax_d.set_xticks([]); ax_d.set_yticks([])
            # Right panel: gamma + alpha on the left axis, omega^2 on a twin, and sigma on its
            # OWN twin (offset spine). sigma (the noise gate g) is ~1000x smaller than gamma, so
            # sharing gamma's axis squashes it to a flat line -- its own axis makes its structure
            # visible.
            ax_d2.plot(ms_axis, drives['gamma'], color='tab:red', lw=0.8, label=r'$\gamma$')
            ax_d2.plot(ms_axis, drives['alpha'], color='tab:purple', lw=0.8, label=r'$\alpha$')
            ax_d2.set_xlim([0, _max_ms])
            ax_d2.set_ylabel(r'$\gamma$, $\alpha$')
            ax_d2t = ax_d2.twinx()
            ax_d2t.plot(ms_axis, drives['omega'] ** 2, color='tab:blue', lw=0.8,
                        label=r'$\omega^2$', alpha=0.7)
            ax_d2t.set_ylabel(r'$\omega^2$', color='tab:blue')
            ax_d2t.tick_params(axis='y', labelcolor='tab:blue')
            if drives.get('sigma') is not None:
                ax_d2s = ax_d2.twinx()
                ax_d2s.spines['right'].set_position(('outward', 44))  # offset so it clears omega^2's axis
                ax_d2s.plot(ms_axis, drives['sigma'], color='tab:green', lw=0.8, label=r'$g=\sigma$')
                ax_d2s.set_ylabel(r'$g=\sigma$', color='tab:green')
                ax_d2s.tick_params(axis='y', labelcolor='tab:green')
            ax_d2.legend(loc='upper left', fontsize=7, framealpha=0.6)
        bottom = n_rows - 1
        axes[bottom, 0].set_xlabel('ms'); axes[bottom, 1].set_xlabel('ms')
        # spec-row right-column labels (skip the drives row which has its own ylabel)
        last_spec_row = len(row_specs) - 1
        for r in range(last_spec_row + 1):
            axes[r, 1].set_ylabel('mel (Hz)')
        # MONITOR_CKPT_LABEL overrides the title's epoch tag — used by the inflight
        # scorer to display the real epoch / step rather than the temp-dir stub of "0".
        _label = os.environ.get("MONITOR_CKPT_LABEL", str(ep))
        fig.suptitle(f"voc{i}  ckpt {_label}  noise_gain={_noise_gain:.2f}  osc_gain={_osc_gain:.2f}   "
                     f"(spec: mel, dB re target peak, [-80, 0])", fontsize=10)
        plt.tight_layout()
        sw.add_figure(f"specgram/voc{i}", fig, step)
        plt.close(fig)
        _dv = drives or {}
        _psd.append((np.asarray(tgt_n, dtype=np.float64), np.asarray(auto_n, dtype=np.float64),
                     _dv.get('noise'), _dv.get('rumble')))

    # Per-epoch mel-PSD panel: target vs tract output (= auto - noise - rumble) vs filtered noise
    # vs rumble, for the same TB vocs. Mel-band power (time-averaged), mel-warped frequency axis.
    if _psd:
        _nm, _fm, _nf, _hop = 128, 16000.0, 1024, 256
        _mhz = librosa.mel_frequencies(n_mels=_nm, fmax=_fm)
        def _mpsd(y):
            y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=_nf, hop_length=_hop,
                                               n_mels=_nm, fmax=_fm, power=2.0)
            return 10.0 * np.log10(S.mean(axis=1) + 1e-12)
        pfig, paxes = plt.subplots(1, len(_psd), figsize=(5 * len(_psd), 4.2),
                                   sharey=True, squeeze=False)
        _xt = np.linspace(0, _nm - 1, 7).astype(int)
        for j, (tg, au, nz, ru) in enumerate(_psd):
            ax = paxes[0, j]
            n = min(len(tg), len(au))
            tract = au[:n].copy()
            if nz is not None:
                tract = tract - np.asarray(nz, dtype=np.float64)[:n]
            if ru is not None:
                tract = tract - np.asarray(ru, dtype=np.float64)[:n]
            ax.plot(np.arange(_nm), _mpsd(tg[:n]), color='tab:orange', lw=1.2, label='target')
            ax.plot(np.arange(_nm), _mpsd(tract), color='tab:green', lw=1.2, label='tract out')
            if nz is not None:
                ax.plot(np.arange(_nm), _mpsd(np.asarray(nz)[:n]), color='tab:gray', lw=1.0, label='noise')
            if ru is not None:
                ax.plot(np.arange(_nm), _mpsd(np.asarray(ru)[:n]), color='tab:blue', lw=1.0, label='rumble')
            ax.set_title(f"voc{j}", fontsize=8); ax.set_xlabel('mel freq (Hz)'); ax.grid(alpha=0.3)
            ax.legend(fontsize=7); ax.set_xticks(_xt)
            ax.set_xticklabels([str(int(_mhz[k])) for k in _xt], fontsize=7)
        paxes[0, 0].set_ylabel('mel-band power (dB)')
        _plabel = os.environ.get("MONITOR_CKPT_LABEL", str(ep))
        pfig.suptitle(f"mel-PSD  ckpt {_plabel}   (target / tract out / noise / rumble)", fontsize=10)
        plt.tight_layout(); sw.add_figure("psd/mel", pfig, step); plt.close(pfig)
    sw.close()
except Exception as _tb_e:
    # Don't let TB rendering errors fail the autonomy score itself.
    print(f"# TB log failed: {type(_tb_e).__name__}: {_tb_e}", file=sys.stderr)

# JSON-safe breakdown: drop the list, keep the scalar mean alongside everything else.
safe_bd = {}
for k, v in bd.items():
    if isinstance(v, bool):
        safe_bd[k] = bool(v)
    elif isinstance(v, list):
        continue  # per-voc lists go to the cache, not the stdout summary
    else:
        safe_bd[k] = float(v)
print(json.dumps({'autonomy': float(score), 'breakdown': safe_bd}))
"""


def _trainer_gpu_pid():
    """PID of the live trainer's python process on the GPU, or None. nvidia-smi gives
    the GPU-using PIDs; filter to ones whose /proc cmdline matches the train pattern."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
        for pid in pids:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
                if TRAIN_PATTERN in cmd:
                    return pid
            except (FileNotFoundError, ProcessLookupError):
                continue
    except Exception:
        pass
    return None


def run_autonomy_on_checkpoint(ckpt_path, step_override=None, log_dir=None, label=None):
    # Pass bpe through env so the snippet writes audio/figures at the same global-batch
    # index the trainer used for its scalar curves -- aligns the x-axis in TB.
    env = os.environ.copy()
    env["MONITOR_BPE"] = str(detect_batches_per_epoch())
    if step_override is not None:
        env["MONITOR_STEP_OVERRIDE"] = str(int(step_override))
    if log_dir is not None:
        env["MONITOR_LOG_DIR"] = str(log_dir)
    if label is not None:
        env["MONITOR_CKPT_LABEL"] = str(label)
    # Stage the specific ckpt into a temp dir as checkpoint_0.tar so the snippet's
    # load_model() (which globs the dir and picks the highest epoch number) actually
    # loads the ckpt we requested. Without this, re-scoring an older ckpt in a dir
    # containing newer ones silently scored the newest. The default log_dir falls
    # back to the original ckpt's dir so per-epoch TB scalars/figures still land
    # alongside the real run output.
    import shutil
    import tempfile
    real_dir = os.path.dirname(ckpt_path)
    stage_dir = tempfile.mkdtemp(prefix="score_ckpt_")
    shutil.copy(ckpt_path, os.path.join(stage_dir, "checkpoint_0.tar"))
    staged_path = os.path.join(stage_dir, "checkpoint_0.tar")
    if log_dir is None:
        env["MONITOR_LOG_DIR"] = str(real_dir)
    # Briefly SIGSTOP the trainer so scoring gets the GPU without contention. Without this
    # the autonomy subprocess queues behind every training batch and times out at the 180s
    # cap. SIGCONT in finally so a crash inside subprocess.run can't leave the trainer
    # frozen. Cost: ~80s of paused training per scored checkpoint (= one epoch save), or
    # roughly 5% wall-clock on a 30-min-epoch run.
    #   CPU scoring mode: skip the pause entirely -- scoring never touches the GPU, so there's
    #   no reason to freeze GPU training. The sidecar then runs fully decoupled (CPU scoring
    #   alongside uninterrupted training), at the cost of some CPU contention during a score.
    _cpu_mode = os.environ.get("MONITOR_DEVICE", "cuda").lower() == "cpu"
    trainer_pid = None if _cpu_mode else _trainer_gpu_pid()
    if trainer_pid is not None:
        try:
            os.kill(trainer_pid, signal.SIGSTOP)
        except (ProcessLookupError, PermissionError):
            trainer_pid = None
    try:
        try:
            r = subprocess.run(
                ["/home/pearson/code/ouroboros/.venv/bin/python", "-c", AUTONOMY_SNIPPET, staged_path],
                capture_output=True, text=True, timeout=180, env=env,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(stage_dir, ignore_errors=True)
            return None, {"error": "scoring subprocess timed out (even with trainer paused)"}
        shutil.rmtree(stage_dir, ignore_errors=True)
    finally:
        if trainer_pid is not None:
            try:
                os.kill(trainer_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    if r.returncode != 0:
        last_err = r.stderr.strip().splitlines()[-1] if r.stderr else "unknown"
        return None, {"error": last_err}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
        return out["autonomy"], out["breakdown"]
    except Exception as e:
        return None, {"error": f"parse: {e}; stdout='{r.stdout[:200]}'"}


def emit(parts):
    status = parts.pop("status")
    fields = [f"status={status}"] + [f"{k}={v}" for k, v in parts.items()]
    print(" ".join(fields), flush=True)


def main():
    state = load_state()
    state["poll_count"] = state.get("poll_count", 0) + 1

    alive = process_alive()
    ev_path = latest_event_file()
    if ev_path is None:
        emit({"status": "DEAD", "reason": "no_events_file"})
        save_state(state); return

    loss = read_loss(ev_path)
    spec = loss.get("Loss/spec", np.array([]))
    tf = loss.get("Loss/tf", np.array([]))
    H_arr = loss.get("Train/H", np.array([]))
    nan_skips = loss["nan_skips"]
    cur_H = int(H_arr[-1]) if len(H_arr) else -1
    epoch_est = int(len(spec) // max(1, detect_batches_per_epoch()))

    spec_w = window_means(spec, w=100, n_windows=20)
    spec_recent5 = float(spec_w[-5:].mean()) if len(spec_w) >= 5 else float("nan")
    spec_prev5 = float(spec_w[-10:-5].mean()) if len(spec_w) >= 10 else float("nan")
    spec_ratio = spec_recent5 / spec_prev5 if (spec_prev5 and not np.isnan(spec_prev5)) else float("nan")
    tf_w = window_means(tf, w=100, n_windows=5)
    tf_recent5 = float(tf_w.mean()) if len(tf_w) else float("nan")

    # Best-spec tracking with margin
    if not np.isnan(spec_recent5):
        prev_best = state.get("best_spec_recent5")
        if prev_best is None or spec_recent5 < prev_best * (1 - PLATEAU_IMPROVE_MARGIN):
            state["best_spec_recent5"] = spec_recent5
            state["best_poll"] = state["poll_count"]
        elif prev_best is None or spec_recent5 < prev_best:
            state["best_spec_recent5"] = spec_recent5

    best_spec_for_fmt = state.get("best_spec_recent5")
    if best_spec_for_fmt is None:
        best_spec_for_fmt = float("nan")
    base = {
        "poll": state["poll_count"], "epoch": epoch_est, "H": cur_H,
        "spec_recent5": f"{spec_recent5:.3g}",
        "spec_ratio": f"{spec_ratio:.3f}",
        "tf_recent5": f"{tf_recent5:.3f}",
        "nan_skips": nan_skips,
        "best_spec": f"{best_spec_for_fmt:.3g}",
        "polls_since_best": state["poll_count"] - state.get("best_poll", 0),
    }

    # 1. Process death
    if not alive:
        emit({"status": "DEAD", **base})
        save_state(state); return

    # 2. NaN spike
    nan_delta = nan_skips - state.get("last_nan_skips", 0)
    state["last_nan_skips"] = nan_skips
    if nan_delta > 50:
        emit({"status": "NAN_SPIKE", "nan_delta": nan_delta, **base})
        save_state(state); return

    # 3. Loss explosion
    if not np.isnan(spec_ratio) and spec_ratio > 2.0:
        emit({"status": "LOSS_SPIKE", **base})
        save_state(state); return

    # 4. New checkpoints -> run autonomy_score on EACH unscored ckpt in epoch order
    # so every ckpt lands its own spectrogram + Val/* scalar on TB. If multiple ckpts
    # have arrived since the last poll, this loop processes them all (oldest first).
    pending = unscored_checkpoints(state.get("last_ckpt_epoch", -1))
    autonomy_line = None
    status = None
    _bpe = detect_batches_per_epoch()
    for ckpt_path, ckpt_epoch in pending:
        # Stage-into-temp-as-checkpoint_0 makes the snippet always parse ep=0,
        # so the snippet's default step = 0*bpe = 0 stacks every per-epoch
        # Val/* scalar at x=0. Pass the real ckpt_epoch through step_override
        # + label so the scalars land at the right global-batch index on TB
        # and the spectrogram title shows the actual epoch number.
        score, bd = run_autonomy_on_checkpoint(
            ckpt_path,
            step_override=ckpt_epoch * _bpe,
            label=str(ckpt_epoch),
        )
        if score is None:
            emit({"status": "CKPT_ERR", "ckpt_epoch": ckpt_epoch,
                  "err": bd.get("error", "?"), **base})
            # Mark this ckpt as attempted so we don't retry on subsequent polls.
            state["last_ckpt_epoch"] = ckpt_epoch
            save_state(state)
            continue

        prev_best_val = state.get("best_val_autonomy")
        if prev_best_val is None or score > prev_best_val + AUTONOMY_IMPROVE_MARGIN:
            state["best_val_autonomy"] = score
            state["ckpts_since_best_autonomy"] = 0
        else:
            state["ckpts_since_best_autonomy"] = state.get("ckpts_since_best_autonomy", 0) + 1
        state["last_val_autonomy"] = score
        state["last_ckpt_epoch"] = ckpt_epoch
        save_state(state)  # persist progress between ckpts so a mid-loop crash doesn't redo work

        autonomy_line = {
            "ckpt_epoch": ckpt_epoch,
            "val_autonomy": f"{score:+.3f}",
            "val_spec_corr": f"{bd.get('spec_corr', float('nan')):+.3f}",
            "val_amp_pen": f"{bd.get('amp_pen', float('nan')):.2f}",
            "val_pitch_pen": f"{bd.get('pitch_pen', float('nan')):.2f}",
            "val_bounded": f"{bd.get('bounded_frac', float('nan')):.2f}",
            "best_val_autonomy": f"{state['best_val_autonomy']:+.3f}",
            "ckpts_since_best": state["ckpts_since_best_autonomy"],
        }
        status = "PLATEAU" if state["ckpts_since_best_autonomy"] >= AUTONOMY_PLATEAU_CKPTS else "NEW_CKPT"
        # Emit per-ckpt so the user sees each one land
        emit({"status": status, **autonomy_line, **base})
    # 4b. Intra-epoch save (--save-minutes): inflight_latest.tar is overwritten by the
    # trainer every N minutes. Score it on a separate cadence keyed on its mtime so the
    # user gets val curves every half hour instead of every 2-hour-epoch. Staged into a
    # temp dir as `checkpoint_0.tar` because load_model accepts a directory of those.
    inflight_path = os.path.join(SEED_DIR, "inflight_latest.tar")
    if os.path.exists(inflight_path):
        cur_mtime = int(os.path.getmtime(inflight_path))
        # +30s slack so identical-mtime polls don't re-score the same file
        if cur_mtime > state.get("last_inflight_mtime", 0) + 30:
            import shutil
            import tempfile
            tmp = tempfile.mkdtemp(prefix="inflight_score_")
            try:
                shutil.copy(inflight_path, os.path.join(tmp, "checkpoint_0.tar"))
                # Synthetic step = latest train scalar step (so Val/ curves align with
                # the train scalars' x-axis). Falls back to spec.size if Loss/spec wasn't
                # logged yet (very early in training).
                if len(spec):
                    # spec_w / read_loss read by-step; the underlying spec array is indexed
                    # by event order. The trainer logs at step = epoch*bpe + batch, so the
                    # length of `spec` IS the latest logged step + 1.
                    step_override = len(spec) - 1
                else:
                    step_override = 0
                score, bd = run_autonomy_on_checkpoint(
                    os.path.join(tmp, "checkpoint_0.tar"),
                    step_override=step_override,
                    log_dir=SEED_DIR,  # write Val/* + specgrams to the real run dir
                    label=f"inflight-step-{step_override}",
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if score is not None:
                state["last_inflight_mtime"] = cur_mtime
                save_state(state)
                emit({
                    "status": "INFLIGHT",
                    "inflight_step": step_override,
                    "val_autonomy": f"{score:+.3f}",
                    "val_spec_corr": f"{bd.get('spec_corr', float('nan')):+.3f}",
                    "val_amp_pen": f"{bd.get('amp_pen', float('nan')):.2f}",
                    "val_pitch_pen": f"{bd.get('pitch_pen', float('nan')):.2f}",
                    "val_bounded": f"{bd.get('bounded_frac', float('nan')):.2f}",
                    **base,
                })
                return

    if pending:
        # Already emitted per-ckpt in the loop above; the loss-side plateau / heartbeat
        # checks below would be redundant noise in the same poll.
        save_state(state); return

    # 5. Loss-side plateau (before any ckpt exists, or as a fallback)
    polls_since_best = state["poll_count"] - state.get("best_poll", 0)
    if polls_since_best >= PLATEAU_NO_IMPROVE_POLLS:
        last_emit = state.get("last_plateau_emit_poll", 0)
        if state["poll_count"] - last_emit >= PLATEAU_NO_IMPROVE_POLLS:
            emit({"status": "PLATEAU_LOSS", **base})
            state["last_plateau_emit_poll"] = state["poll_count"]
            save_state(state); return

    # 6. Heartbeat
    if state["poll_count"] % HEARTBEAT_EVERY == 0:
        emit({"status": "HEARTBEAT", **base})

    state["history"].append({"poll": state["poll_count"], "epoch": epoch_est,
                             "spec_recent5": spec_recent5,
                             "tf_recent5": tf_recent5})
    if len(state["history"]) > 200:
        state["history"] = state["history"][-200:]
    save_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", default=_DEFAULT_SEED,
                        help="run/seed directory to watch (default: live run).")
    parser.add_argument("--state-path", default=_DEFAULT_STATE,
                        help="JSON file to persist monitor state across polls.")
    parser.add_argument("--train-pattern", default=_DEFAULT_TRAIN_PATTERN,
                        help="pgrep -f pattern that identifies the training process. "
                             "Distinguish concurrent runs by their out-dir basename.")
    args = parser.parse_args()
    SEED_DIR = args.seed_dir
    STATE_PATH = args.state_path
    TRAIN_PATTERN = args.train_pattern
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    main()
