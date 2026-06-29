"""Stage org545's annotations (CAGbirds layout) into a flat directory:
~/ouroboros_data/org545_multi/. Each (wav, txt) pair becomes:
    syll{X}__{stem}.wav
    syll{X}__{stem}.txt
matching the `--stratify-sep "__"` convention so the multi-syllable trainer
sees 5 groups (syllA, syllB, syllC, syllD, syllE).

org545's layout differs from muscimol/blk445 et al.:
    .../org545/data/<YYYYMMDD>/denoised/<stem>.wav            # plain stem (no _cleaned)
    .../org545/data/<YYYYMMDD>/denoised_segments_syllables<X>/<stem>.txt
    <stem>.txt has a leading "# ..." comment line (np.loadtxt skips those).

The same recording stem appears under 5 different syllable directories with
different annotations; each gets its own (syll{X}__{stem}.wav, syll{X}__{stem}.txt)
pair so the loader can stratify.
"""

import os
import sys

SRC = os.path.expanduser("~/isilon/All_Staff/birds/mooney/CAGbirds/org545/data")
DST = os.path.expanduser("~/ouroboros_data/org545_multi")
SYLLABLES = ["A", "B", "C", "D", "E"]


def main():
    if not os.path.isdir(SRC):
        print(f"source root missing: {SRC}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(DST, exist_ok=True)

    n_link = 0
    n_skip_empty = 0
    n_skip_no_wav = 0
    by_prefix = {}

    for day in sorted(os.listdir(SRC)):
        day_dir = os.path.join(SRC, day)
        if not os.path.isdir(day_dir):
            continue
        wav_dir = os.path.join(day_dir, "denoised")
        if not os.path.isdir(wav_dir):
            continue
        for syl in SYLLABLES:
            seg_dir = os.path.join(day_dir, f"denoised_segments_syllables{syl}")
            if not os.path.isdir(seg_dir):
                continue
            prefix = f"syll{syl}"
            for txt_name in sorted(os.listdir(seg_dir)):
                if not txt_name.endswith(".txt"):
                    continue
                stem = txt_name[:-4]
                txt_src = os.path.join(seg_dir, txt_name)
                if os.path.getsize(txt_src) == 0:
                    n_skip_empty += 1
                    continue
                wav_src = os.path.join(wav_dir, f"{stem}.wav")
                if not os.path.isfile(wav_src):
                    n_skip_no_wav += 1
                    continue
                wav_dst = os.path.join(DST, f"{prefix}__{stem}.wav")
                txt_dst = os.path.join(DST, f"{prefix}__{stem}.txt")
                for src, dst in [(wav_src, wav_dst), (txt_src, txt_dst)]:
                    if os.path.lexists(dst):
                        continue
                    os.symlink(src, dst)
                    n_link += 1
                by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    print(f"linked {n_link} files ({n_link // 2} (wav, txt) pairs)")
    print(f"skipped {n_skip_empty} empty txts, {n_skip_no_wav} txts with no matching wav")
    print(f"\nper-prefix pair counts:")
    for p, n in sorted(by_prefix.items()):
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main()
