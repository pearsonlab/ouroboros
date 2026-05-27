"""Generate a shard of synthetic Mindlin/gabo vocalizations (for parallel data gen)."""
import argparse
import jax.random as jr
from data.generate_data_gabo import generate_vocal_dataset

p = argparse.ArgumentParser()
p.add_argument("--out-dir", required=True)
p.add_argument("--n-vocs", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--noise-sd", type=float, default=0.0002)
a = p.parse_args()

generate_vocal_dataset(
    jr.PRNGKey(a.seed), n_vocs=a.n_vocs, noise_sd=a.noise_sd,
    audio_loc=a.out_dir, seg_loc=a.out_dir, func_loc=a.out_dir,
)
print(f"shard {a.out_dir}: wrote {a.n_vocs} vocs (seed {a.seed})")
