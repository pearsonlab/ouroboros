from data.load_data import get_segmented_audio
from data.data_utils import get_loaders

from train.model_cv import model_cv_lambdas

from typing import Union
import os


def train_model(
    audio_dirs: list[str],
    seg_dirs: list[str],
    model_dir: str,
    max_vocs: int = 5000,
    context_len=0.25,
    seed: Union[None, int] = 1234,
    shuffle_order=False,
    max_jobs: int = 4,
    batch_size: int = 32,
    n_epochs: int = 100,
):

    n_cpu = os.cpu_count()

    n_jobs = min(max_jobs, n_cpu if n_cpu is not None else 0)

    assert len(audio_dirs) == len(seg_dirs), print(
        "Need the same number of audio dirs as segment dirs!"
    )
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    chunks_per_dir = max_vocs // len(audio_dirs)
    chunks = []

    for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
        audio, sr = get_segmented_audio(
            audio_dir,
            seg_dir,
            max_vocs=chunks_per_dir,
            context_len=context_len,
            seed=seed,
            training=True,
            extend=True,
            shuffle_order=shuffle_order,
        )
        chunks += audio
    dt = 1 / sr
    dataloaders = get_loaders(
        chunks,
        num_workers=n_jobs,
        batch_size=batch_size,
        train_size=0.6,
        cv=True,
        seed=seed,
        dt=dt,
    )

    best_model = model_cv_lambdas(
        dls=dataloaders,
        dt=dt,
        n_epochs=n_epochs,
        lr=1e-3,
        n_kernels=15,
        expand_factor=10,
        n_layers=3,
        d_state=1,
        d_conv=4,
        tau=dt,
        model_path=model_dir,
        save_freq=5,
    )

    return best_model


if __name__ == "__main__":
    pass
