"""Stage every (bird, syllable) annotation under
~/isilon/.../muscimol/Microdialysis/Muscimol/ into a single flat directory
~/ouroboros_data/finch_multi/ so the multi-syllable trainer can mix them in
one pool.

Naming convention: each pair is exposed as
    {bird}_syll{X}__{stem}.wav
    {bird}_syll{X}__{stem}.txt
where {prefix} = "{bird}_syll{X}" identifies the group, and {stem} is the
original recording stem (one recording may carry two syllable annotations
under different prefixes -- e.g. org666 has both syllable_A and syllable_B
on the same wav -- and gets a symlink per syllable).

Wavs come from each bird's double_denoised/{stem}_cleaned.wav (matching
stage_finch_blk445_syllC.py's convention).
"""

import os
import sys

SRC = os.path.expanduser(
    "~/isilon/All_Staff/birds/mooney/muscimol/Microdialysis/Muscimol"
)
DST = os.path.expanduser("~/ouroboros_data/finch_multi")
os.makedirs(DST, exist_ok=True)


def main():
    if not os.path.isdir(SRC):
        print(f"source root missing: {SRC}", file=sys.stderr)
        sys.exit(1)

    n_link = 0
    n_skip_empty = 0
    n_skip_no_wav = 0
    by_prefix = {}

    for bird in sorted(os.listdir(SRC)):
        segs_root = os.path.join(SRC, bird, "segs")
        if not os.path.isdir(segs_root):
            continue
        for day in sorted(os.listdir(segs_root)):
            day_dir = os.path.join(segs_root, day)
            if not os.path.isdir(day_dir):
                continue
            denoised_dir = os.path.join(SRC, bird, day, "double_denoised")
            if not os.path.isdir(denoised_dir):
                continue
            for syl_name in sorted(os.listdir(day_dir)):
                if not syl_name.startswith("syllable_"):
                    continue
                syl_dir = os.path.join(day_dir, syl_name)
                if not os.path.isdir(syl_dir):
                    continue
                syl_letter = syl_name.split("_", 1)[1]
                prefix = f"{bird}_syll{syl_letter}"
                for txt_name in sorted(os.listdir(syl_dir)):
                    if not txt_name.endswith(".txt"):
                        continue
                    stem = txt_name[:-4]
                    txt_src = os.path.join(syl_dir, txt_name)
                    if os.path.getsize(txt_src) == 0:
                        n_skip_empty += 1
                        continue
                    wav_src = os.path.join(denoised_dir, f"{stem}_cleaned.wav")
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
