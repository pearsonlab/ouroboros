from data.real_data import *
from model.kernels import fullPolyModule
from data.data_utils import get_loaders
from train.train import train, load_model, save_model
from train.eval import eval_model_error
from model.constrained_model import rkhs_ouroboros
from utils import sse
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visualization.model_vis import loss_plot


def test_eval_error():

    audio_path = "/home/miles/isilon/All_Staff/marmosets"
    seg_path = audio_path
    model_path = "/home/miles/isilon/All_Staff/miles/models/ouroboros/cv_eval_test"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    audio_subdir = "s*/wavfiles/synchro_cleaned"
    seg_subdir = "s*/wavfiles"
    audio_filetype = "_cleaned.wav"
    seed = None
    d_conv = 4
    expand_factor = 10
    n_layers = 4
    n_kernels = 15
    smooth_len = 0.005
    d_state = 1
    tau = 1000
    nEpochs = 25

    audios, sr = get_segmented_audio(
        audio_path,
        seg_path,
        audio_subdir=audio_subdir,
        seg_subdir=seg_subdir,
        envelope=False,
        context_len=0.15,
        audio_type=audio_filetype,
        seg_type=".txt",
        max_pairs=250,
        seed=seed,
    )

    dls = get_loaders(np.vstack(audios), cv=True, train_size=0.6, seed=seed)

    model_path_full_poly = model_path + "/kernelborous_poly_end_to_end_lambda_0.75"
    save_loc_poly = model_path_full_poly + "/checkpoint_100.tar"

    kernel = fullPolyModule(
        nTerms=n_kernels,
        device="cuda",
        x_dim=1,
        z_dim=2,
        activation=lambda x: x,
        lam=0.75,
        trend_filtering=True,
    )
    reg_weights = True
    full_model_poly = rkhs_ouroboros(
        d_data=1,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand_factor=expand_factor,
        tau=tau,
        smooth_len=smooth_len,
        kernel=kernel,
    )

    full_opt_poly = Adam(full_model_poly.parameters(), lr=1e-3)
    full_scheduler_poly = ReduceLROnPlateau(
        full_opt_poly, factor=0.5, patience=max(nEpochs // 25, 2), min_lr=1e-10
    )

    tl, vl, full_model_poly, full_opt_poly = train(
        full_model_poly,
        full_opt_poly,
        loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
        loaders=dls,
        scheduler=full_scheduler_poly,
        nEpochs=nEpochs,
        val_freq=1,
        runDir=model_path_full_poly,
        dt=1 / sr,
        vis_freq=100,
        smoothing=False,
        reg_weights=reg_weights,
    )

    loss_plot(tl, vl, save_loc=model_path_full_poly, show=False)

    save_model(
        full_model_poly,
        full_opt_poly,
        save_loc_poly,
        n_layers=n_layers,
        d_state=d_state,
        expand_factor=expand_factor,
        d_conv=d_conv,
    )
    print("Testing consistency of eval trained model")
    for repeat1 in range(10):
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly, dt=1 / sr
        )
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly, dt=1 / sr, comparison="test"
        )

    print("Testing consistency of model loading")
    full_model_poly_loaded, full_opt_poly_loaded, full_scheduler_poly = load_model(
        save_loc_poly, kernel_type="full_poly"
    )

    for repeat1 in range(10):
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly_loaded, dt=1 / sr
        )
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly_loaded, dt=1 / sr, comparison="test"
        )

    print("also testing consistency of model_loading")
    for repeat1 in range(10):
        full_model_poly_loaded, full_opt_poly_loaded, full_scheduler_poly = load_model(
            save_loc_poly, kernel_type="full_poly"
        )
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly_loaded, dt=1 / sr
        )
        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model_poly_loaded, dt=1 / sr, comparison="test"
        )


if __name__ == "__main__":
    test_eval_error()
