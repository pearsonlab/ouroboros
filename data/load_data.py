import numpy as np
import warnings

import os
import glob
from scipy.io import wavfile


from typing import Tuple, Union


def get_audio_training(
    audio_files: list[str],
    seg_files: list[str],
    max_vocs: int = 5000,
    context_len: float = 0.3,
    extend: bool = True,
):
    """
    Takes a list of wav files, containing audio, and a list
    of txt files, containing the onsets and offsets of vocalizations.
    Using these, splits the audio files into chunks of
    `context_len` long vocalizations, returning at most
    `max_vocs` vocalizations. Used for generating model train/test data

    Inputs
    -----
        - audio_files: a list of .wav files
        - seg_files: a list of .txt files, containing onsets and offsets of vocalizations
        - max_vocs: maximum number of vocalizations to grab from the audio, total
        - context_len: length of segmented audio in seconds, when full_vocs = False
        - extend: whether to extend the onset of vocalizations to avoid cutting off the
        end when chunking

    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    audio = []

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = wavfile.read(a)  # here, we assume
            # all loaded files have the same sample rate
            if aud.dtype == np.int16:  # ints are TOO BIG!! turn into floats
                aud = aud / -np.iinfo(aud.dtype).min

            chunk_len = int(round(context_len * sr))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg_onoffs = np.loadtxt(s, usecols=(0, 1))

            if len(seg_onoffs) == 0:
                continue
            if len(seg_onoffs.shape) == 1:
                seg_onoffs = seg_onoffs[None, :]

            for on, off in seg_onoffs:
                difference = (off - on) - context_len
                if (difference <= 0) and extend:
                    # extend from the beginning, if shorter than context_len
                    on += difference
                on_ind, off_ind = int(round(on * sr)), int(round(off * sr))
                aud_sample = aud[on_ind:off_ind]

                cut_len = np.mod(len(aud_sample), chunk_len)
                if cut_len > 0:
                    aud_sample = aud_sample[:-cut_len]

                aud_sample = aud_sample.reshape(-1, chunk_len, 1)

                audio += list(aud_sample)

                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr

    return audio, sr


def get_audio_analysis(
    audio_files: list[str],
    seg_files: list[str],
    max_vocs: int = 5000,
    padding: float = 0.1,
):
    """
    Takes a list of wav files, containing audio, and a list
    of txt files, containing the onsets and offsets of vocalizations.
    Using these, extracts vocalizations, returning at most
    `max_vocs` of them. Used for generating data for analysis

    Inputs
    -----
        - audio_files: a list of .wav files
        - seg_files: a list of .txt files, containing onsets and offsets of vocalizations
        - max_vocs: maximum number of vocalizations to grab from the audio, total
        - padding: amount of time (in seconds) to pad the onsets of vocalizations wtih

    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    audio = []

    for a, s in zip(audio_files, seg_files):
        if os.path.isfile(a) and os.path.isfile(s):
            sr, aud = wavfile.read(a)  # here, we assume
            # all loaded files have the same sample rate

            if aud.dtype == np.int16:  # ints are TOO BIG!! turn into floats
                aud = aud / -np.iinfo(aud.dtype).min

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seg_onoffs = np.loadtxt(s, usecols=(0, 1))
            
            if len(seg_onoffs) == 0:
                continue
            if len(seg_onoffs.shape) == 1:
                seg_onoffs = seg_onoffs[None, :]

            for on, off in seg_onoffs:
                on = max(0.0, on - padding)
                on_ind, off_ind = int(round(on * sr)), int(round(off * sr))
                audio.append(aud[on_ind:off_ind][None, :, None])

                if len(audio) >= max_vocs:
                    return audio[:max_vocs], sr

    return audio, sr


def get_segmented_audio(
    audio_path: str,
    seg_path: str,
    audio_id: str = ".wav",
    max_vocs: int = 5000,
    context_len: float = 0.3,
    seed: Union[None, int] = None,
    training: bool = False,
    extend: bool = True,
    padding: float = 0.0,
    shuffle_order: bool = True,
) -> Tuple[list, int]:
    """
    Takes a path to audio files and a path to segment files.
    returns segmented audio and the sample rate of the audio.

    Inputs
    -----
        - audio_path: path to a set of audio files
        - seg_path: path to a set of .txt files with vocalization onsets and offsets
        - audio_id: common ending to all audio files. Used to get
            .txt filenames
        - max_vocs: number of vocalizations to extract
        - context_len: length of vocal chunks. used for training
        - seed: random seed used when shuffling filenames
        - training: whether to generate audio for training or analysis
        - extend: whether to extend onsets for training chunks
        - padding: how much to pad onsets of vocalizations for analysis chunks
        - shuffle_order: whether to shuffle filenames before extracting vocalizations
    Returns
    -----
        - a list of collected audio
        - the sample rate of the audio
    """

    gen = np.random.default_rng(seed=seed)

    audio_files = glob.glob(os.path.join(audio_path, "*" + audio_id))

    if shuffle_order:
        order = gen.choice(len(audio_files), len(audio_files), replace=False)
        audio_files = [audio_files[o] for o in order]
        # seg_files = [seg_files[o] for o in order]

    audio_tags = [a.split("/")[-1].split(audio_id)[0] for a in audio_files]
    seg_files = [os.path.join(seg_path, a + ".txt") for a in audio_tags]

    if training:
        audio_segments, sr = get_audio_training(
            audio_files,
            seg_files,
            max_vocs=max_vocs,
            context_len=context_len,
            extend=extend,
        )

    else:
        audio_segments, sr = get_audio_analysis(
            audio_files, seg_files, max_vocs=max_vocs, padding=padding
        )

    return audio_segments, sr


# --- Edge-biased segment sampler for spectral-rollout training -----------------------

ONSET = 0
OFFSET = 1
MID = 2


def get_audio_training_edge_weighted(
    audio_files: list[str],
    seg_files: list[str],
    *,
    context_len: float,
    edge_ms: float = 10.0,
    silence_prefix_ms: float = 25.0,
    silence_suffix_ms: float = 25.0,
    ratio=(0.4, 0.4, 0.2),
    max_segs: int = 5000,
    seed: int = 0,
    int16_norm: bool = True,
    silence_ratio: float = 1.0,
):
    """
    Categorized segment sampler that heavily oversamples syllable onsets and offsets.

    For each (wav, txt) pair, three pools of fixed-length (context_len seconds) windows
    are built:
      - ONSET  (cat=0): window starts at on - silence_prefix_ms  (real pre-onset audio
                        forms the silence-noise prefix; first context_len samples).
      - OFFSET (cat=1): window ends   at off + silence_suffix_ms (real post-offset audio
                        forms the silence tail; last context_len samples).
      - MID    (cat=2): non-overlapping windows of length context_len lying strictly inside
                        [on + edge_ms, off - edge_ms].

    The pools are then sampled to `max_segs` segments with category-counts proportional to
    `ratio` (with replacement only if a pool is exhausted). Returns
    (segments_list, categories_array, sr).

    The cold-start training step gets the cold-start noise IC only on ONSET examples
    (categories == ONSET); OFFSET/MID examples use the data IC. See
    train/spectral_rollout.py.

    `silence_ratio` (default 1.0 = no filter, legacy) controls a quietness check for
    candidate ONSET windows: an ONSET candidate is kept only if
        RMS(pre-onset region) <= silence_ratio * RMS(syllable body).
    Otherwise the same window is reclassified into MID (the prefix isn't really a
    silence-to-onset transition; it's mid-song activity from other syllables). Same
    rule mirrored on OFFSET (RMS(post-offset region) vs syllable body). Useful for
    continuous-song datasets where the annotated onset is often surrounded by other
    annotated/unannotated syllable activity.
    """
    rng = np.random.default_rng(seed)
    pools = {ONSET: [], OFFSET: [], MID: []}
    sr = None

    for a, s in zip(audio_files, seg_files):
        if not (os.path.isfile(a) and os.path.isfile(s)):
            continue
        sr, aud = wavfile.read(a)
        if aud.dtype == np.int16 and int16_norm:
            aud = aud / -np.iinfo(aud.dtype).min
        aud = aud.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            on_off = np.loadtxt(s, usecols=(0, 1))
        if on_off.size == 0:
            continue
        if on_off.ndim == 1:
            on_off = on_off[None, :]

        L_seg = int(round(context_len * sr))
        pre = int(round(silence_prefix_ms / 1000.0 * sr))
        suff = int(round(silence_suffix_ms / 1000.0 * sr))
        edge = int(round(edge_ms / 1000.0 * sr))

        for on_s, off_s in on_off:
            on_i = int(round(on_s * sr))
            off_i = int(round(off_s * sr))

            # ONSET: window starts pre samples before onset
            start = on_i - pre
            end = start + L_seg
            if start >= 0 and end <= len(aud) and end > on_i:
                win = aud[start:end]
                # Quietness check: pre-onset region must be silence_ratio*body or quieter.
                # body = annotated syllable interval inside the window. Skip the check
                # (and accept as ONSET) when silence_ratio >= 1.0 (legacy behaviour) or
                # when the body has too few samples to measure RMS reliably.
                if silence_ratio < 1.0:
                    prefix = aud[start:on_i]
                    body_end = min(off_i, end)
                    body = aud[on_i:body_end]
                    pref_rms = float(np.sqrt(np.mean(prefix.astype(np.float64) ** 2))) if len(prefix) > 0 else 0.0
                    body_rms = float(np.sqrt(np.mean(body.astype(np.float64) ** 2))) if len(body) > 32 else 0.0
                    is_silent_prefix = body_rms > 0 and pref_rms <= silence_ratio * body_rms
                else:
                    is_silent_prefix = True
                if is_silent_prefix:
                    pools[ONSET].append(win)
                else:
                    pools[MID].append(win)

            # OFFSET: window ends suff samples after offset
            end = off_i + suff
            start = end - L_seg
            if start >= 0 and end <= len(aud) and start < off_i:
                win = aud[start:end]
                if silence_ratio < 1.0:
                    suffix = aud[off_i:end]
                    body_start = max(on_i, start)
                    body = aud[body_start:off_i]
                    suff_rms = float(np.sqrt(np.mean(suffix.astype(np.float64) ** 2))) if len(suffix) > 0 else 0.0
                    body_rms = float(np.sqrt(np.mean(body.astype(np.float64) ** 2))) if len(body) > 32 else 0.0
                    is_silent_suffix = body_rms > 0 and suff_rms <= silence_ratio * body_rms
                else:
                    is_silent_suffix = True
                if is_silent_suffix:
                    pools[OFFSET].append(win)
                else:
                    pools[MID].append(win)

            # MID: non-overlapping windows strictly inside (on+edge, off-edge)
            mid_start = on_i + edge
            mid_end = off_i - edge
            if mid_end - mid_start >= L_seg:
                for ms in range(mid_start, mid_end - L_seg + 1, L_seg):
                    pools[MID].append(aud[ms:ms + L_seg])

    # If max_segs is 0 use the whole pool; otherwise sample to that target with the ratio.
    pool_sizes = {c: len(pools[c]) for c in (ONSET, OFFSET, MID)}
    if sum(pool_sizes.values()) == 0:
        return [], np.array([], dtype=np.int64), sr

    if max_segs <= 0:
        # Return everything, no resampling.
        out_segs, out_cats = [], []
        for cat in (ONSET, OFFSET, MID):
            for seg in pools[cat]:
                out_segs.append(seg.reshape(-1, 1).astype(np.float32))
                out_cats.append(cat)
        order = rng.permutation(len(out_segs))
        return ([out_segs[i] for i in order],
                np.array([out_cats[i] for i in order], dtype=np.int64),
                sr)

    ratio = np.asarray(ratio, dtype=np.float64)
    ratio = ratio / ratio.sum()
    counts = np.round(ratio * max_segs).astype(int)
    counts[-1] = max_segs - counts[:-1].sum()  # exact total

    out_segs, out_cats = [], []
    for cat, n in zip((ONSET, OFFSET, MID), counts):
        pool = pools[cat]
        if not pool or n <= 0:
            continue
        replace = n > len(pool)
        idx = rng.choice(len(pool), size=n, replace=replace)
        for i in idx:
            out_segs.append(pool[i].reshape(-1, 1).astype(np.float32))
            out_cats.append(cat)

    order = rng.permutation(len(out_segs))
    return ([out_segs[i] for i in order],
            np.array([out_cats[i] for i in order], dtype=np.int64),
            sr)


def get_segmented_audio_edge_weighted(
    audio_path: str,
    seg_path: str,
    *,
    audio_id: str = ".wav",
    context_len: float,
    edge_ms: float = 10.0,
    silence_prefix_ms: float = 25.0,
    silence_suffix_ms: float = 25.0,
    ratio=(0.4, 0.4, 0.2),
    max_segs: int = 5000,
    seed: int = 0,
    shuffle_files: bool = True,
):
    """Convenience wrapper: glob audio files from audio_path, pair with seg_path/*.txt,
    then call get_audio_training_edge_weighted. Returns (segments, categories, sr).
    """
    gen = np.random.default_rng(seed)
    audio_files = glob.glob(os.path.join(audio_path, "*" + audio_id))
    if shuffle_files:
        order = gen.permutation(len(audio_files))
        audio_files = [audio_files[o] for o in order]
    audio_tags = [a.split("/")[-1].split(audio_id)[0] for a in audio_files]
    seg_files = [os.path.join(seg_path, a + ".txt") for a in audio_tags]
    return get_audio_training_edge_weighted(
        audio_files, seg_files,
        context_len=context_len,
        edge_ms=edge_ms,
        silence_prefix_ms=silence_prefix_ms,
        silence_suffix_ms=silence_suffix_ms,
        ratio=ratio,
        max_segs=max_segs,
        seed=seed,
    )
