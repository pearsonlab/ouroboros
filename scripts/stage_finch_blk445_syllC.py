"""Stage blk445 / syllable_C real-finch data for the Ouroboros training pipeline.

The pipeline (data/load_data.py::get_segmented_audio + examples/run_lambda_pipeline.py)
expects each shard directory to contain paired `{stem}.wav` and `{stem}.txt` files
side by side. The real data lives in two split locations with mismatched suffixes:

    audio:       ~/isilon/All_Staff/.../blk445/{day}/double_denoised/{stem}_cleaned.wav
    annotations: ~/isilon/.../blk445/segs/{day}/syllable_C/{stem}.txt

This script symlinks (no copies) non-empty annotations and their matching cleaned
wavs into ~/ouroboros_data/blk445_syllC/day{day}/ with `{stem}.wav` / `{stem}.txt`
filenames the pipeline can glob. Skips annotations whose corresponding wav is
missing. Idempotent.
"""

import os
import sys

DAYS = [84, 85, 86]
SRC_ROOT = os.path.expanduser(
    "~/isilon/All_Staff/birds/mooney/muscimol/Microdialysis/Muscimol/blk445"
)
DST_ROOT = os.path.expanduser("~/ouroboros_data/blk445_syllC")


def stage_day(day):
    seg_dir = os.path.join(SRC_ROOT, "segs", str(day), "syllable_C")
    wav_dir = os.path.join(SRC_ROOT, str(day), "double_denoised")
    dst_dir = os.path.join(DST_ROOT, f"day{day}")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(seg_dir):
        print(f"day{day}: missing seg_dir {seg_dir}", file=sys.stderr)
        return 0, 0
    if not os.path.isdir(wav_dir):
        print(f"day{day}: missing wav_dir {wav_dir}", file=sys.stderr)
        return 0, 0

    staged = 0
    skipped_empty = 0
    skipped_no_wav = 0
    for fn in sorted(os.listdir(seg_dir)):
        if not fn.endswith(".txt"):
            continue
        seg_path = os.path.join(seg_dir, fn)
        if os.path.getsize(seg_path) == 0:
            skipped_empty += 1
            continue
        stem = fn[:-len(".txt")]
        wav_src = os.path.join(wav_dir, f"{stem}_cleaned.wav")
        if not os.path.isfile(wav_src):
            skipped_no_wav += 1
            continue

        wav_link = os.path.join(dst_dir, f"{stem}.wav")
        txt_link = os.path.join(dst_dir, f"{stem}.txt")
        for link, target in [(wav_link, wav_src), (txt_link, seg_path)]:
            if os.path.lexists(link):
                if os.readlink(link) == target:
                    continue
                os.remove(link)
            os.symlink(target, link)
        staged += 1

    print(
        f"day{day}: staged={staged} (skipped {skipped_empty} empty annotations, "
        f"{skipped_no_wav} with missing wav) -> {dst_dir}",
        flush=True,
    )
    return staged, skipped_no_wav


def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    total_staged = 0
    for day in DAYS:
        n, _ = stage_day(day)
        total_staged += n
    print(f"TOTAL staged pairs: {total_staged}", flush=True)


if __name__ == "__main__":
    main()
