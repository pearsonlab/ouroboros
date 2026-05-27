from data.load_data import get_segmented_audio
from data.data_utils import get_loaders

from train.model_cv import model_cv_lambdas, train_arneodo

from typing import Union
import os
import torch
import numpy as np
from tqdm import tqdm


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
    save_freq: int = 5,
    parameterization: str = "poly",
    lr: float = 1e-3,
    n_layers: int = 3,
    d_state: int = 1,
    d_conv: int = 4,
    expand_factor: int = 10,
    drive_lowpass_ms: float = 1.0,
) -> torch.nn.Module:
    """
    function for training a model. takes audio from
    audio_dirs, onsets and offsets from segmentation files
    in seg_dirs. cross-validates over the one hyperparameter of this model

    inputs
    --------
            audio_dirs: list of folders with audio
            seg_dirs: list of folders with segmentation decisions
            model_dir: location to save model checkpoints and plots
            max_vocs: max number of vocal chunks to train on
            context_len: context window to train model on
            seed: random seed for reproducibility
            shuffle_order: whether to shuffle audio files for gathering train data
            max_jobs: max number of jobs for dataloader
            batch_size: batch size during training
            n_epochs: max number of passes through the data during training
            save_freq: how often (in epochs) we want to checkpoint model
            parameterization: "poly" for the full-polynomial Ouroboros (with lambda
                cross-validation), or "arneodo" for the Arneodo 2021 syrinx ODE
                parameterization (single fit, no regularization CV)
            lr: learning rate
            n_layers: number of mamba layers in each encoder
            d_state: internal SSM state size of the mamba encoders. The default (1) is
                small; bumping it (e.g. 4) substantially improves fit -- with enough data
                the arneodo model reaches R^2 > 0.98 at d_state=4. Mind GPU memory: the
                parallel scan allocates ~batch * npo2(2*seq) * 2*expand_factor * d_state.
            d_conv: width of the mamba convolutional kernel
            expand_factor: channel expansion from audio to mamba input
            drive_lowpass_ms: (arneodo only) hard low-pass timescale (ms) on the
                alpha/beta/delta drives; default 1 ms gives slow, physiological drives and
                cold-start-stable autonomous dynamics. Set 0.0 for the unregularized model.
    returns
    --------
            best model after hyperparameter cross-validation
    """

    n_cpu = os.cpu_count()

    n_jobs = min(max_jobs, n_cpu if n_cpu is not None else 0)

    assert len(audio_dirs) == len(seg_dirs), print(
        "Need the same number of audio dirs as segment dirs!"
    )
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    chunks_per_dir = max_vocs // len(audio_dirs)
    chunks = []

    for audio_dir, seg_dir in tqdm(
        zip(audio_dirs, seg_dirs), desc="Gathering training data", total=len(audio_dirs)
    ):
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

    print(f"Gathered {len(chunks)}/{max_vocs} allowed vocalizations")
    dt = 1 / sr
    dataloaders = get_loaders(
        np.stack(chunks, axis=0),
        num_workers=n_jobs,
        batch_size=batch_size,
        train_size=0.6,
        cv=True,
        seed=seed,
        dt=dt,
    )

    if parameterization == "arneodo":
        best_model = train_arneodo(
            dls=dataloaders,
            dt=dt,
            n_epochs=n_epochs,
            lr=lr,
            expand_factor=expand_factor,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            tau=dt,
            model_path=model_dir,
            save_freq=save_freq,
            drive_lowpass_ms=drive_lowpass_ms,
        )
    else:
        best_model = model_cv_lambdas(
            dls=dataloaders,
            dt=dt,
            n_epochs=n_epochs,
            lr=lr,
            n_kernels=15,
            expand_factor=expand_factor,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            tau=dt,
            model_path=model_dir,
            save_freq=save_freq,
        )

    return best_model


if __name__ == "__main__":
    pass
