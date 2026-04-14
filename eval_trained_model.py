from data.real_data import *
from data.data_utils import get_loaders
from train.train import load_model
from model.constrained_model import rkhs_ouroboros
from model.kernels import *
import gc
import torch
import fire
from train.eval import eval_model_error
from visualization.model_vis import r2_plot


def eval_model(
    model_path="",
    context_len=0.3,
    max_pairs=1000,
    kernel_type="gauss",
    n_kernels=10,
    n_layers=2,
    expand_factor=4,
    d_state=1,
    tau=1000,
    smooth_len=0.005,
    trainbird="org545",
):

    use_trend = True
    if kernel_type == "gauss":
        kernel = simpleGaussModule(
            nTerms=n_kernels,
            device="cuda",
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            trend_filtering=use_trend,
        )
        reg_weights = False
    elif kernel_type == "constant_gauss":
        kernel = constantGaussModule(
            nTerms=n_kernels,
            device="cuda",
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            trend_filtering=use_trend,
        )
        reg_weights = False
    elif kernel_type == "full_poly":
        kernel = fullPolyModule(
            nTerms=n_kernels,
            device="cuda",
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            lam=1.5,
            trend_filtering=use_trend,
        )
        reg_weights = True

    else:
        kernel = polyModule(
            nTerms=n_kernels,
            device="cuda",
            x_dim=1,
            z_dim=2,
            activation=lambda x: x,
            lam=0.9,
            trend_filtering=use_trend,
        )
        reg_weights = False

    full_model = rkhs_ouroboros(
        d_data=1,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=4,
        expand_factor=expand_factor,
        tau=tau,
        smooth_len=smooth_len,
        kernel=kernel,
    )
    save_loc = model_path + "/checkpoint_100.tar"
    full_model, opt, sched = load_model(save_loc, kernel_type=kernel_type)

    birds = ["blk521", "org512", "org545", "pur567"]

    path = "/home/miles/isilon/All_Staff/birds/mooney/CAGbirds"

    mus, sds, labels = [], [], []
    for bird in birds:
        p = os.path.join(path, bird, "data")
        audios, sr = get_segmented_audio(
            p,
            p,
            envelope=False,
            context_len=context_len,
            audio_type="2*[0-9][0-9]/denoised/*.wav",
            seg_type="2*[0-9][0-9]/denoised_segments/*.txt",
            max_pairs=max_pairs,
        )
        dls = get_loaders(
            np.vstack(audios),
            cv=True,
            train_size=0.6,
            seed=None,
            oversample_prop=1,
            dt=1 / sr,
        )

        (train_mu, test_mu), (train_sd, test_sd) = eval_model_error(
            dls, full_model, dt=1 / sr
        )
        if bird != trainbird:
            mus.append(train_mu)
            sds.append(train_sd)
            labels.append(bird)
        else:
            mus.append(train_mu)
            mus.append(test_mu)
            sds.append(train_sd)
            sds.append(test_sd)
            labels.append(bird + " train")
            labels.append(bird + " test")

        del dls
        del audios
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    r2_plot(mus, sds, labels, saveloc=model_path, show=True)


if __name__ == "__main__":
    fire.Fire(eval_model)
