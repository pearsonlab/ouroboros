from data.real_data import *
from model.constrained_model import rkhs_ouroboros
from model.kernels import *
import torch


def analyze_model(
    model_path,
    audio_path,
    seg_path="",
    context_len=0.3,
    seg_filetype=".txt",
    max_pairs=1000,
    audio_type=".wav",
    kernel_type="gauss",
    n_kernels=10,
    plot_fig=True,
    plot_fncs=True,
    nPairs=20,
):

    if kernel_type == "gauss":
        kernel = simpleGaussModule(
            nTerms=n_kernels, device="cuda", xdim=1, z_dim=4, activation=lambda x: x
        )
    else:
        kernel = polyModule(
            nTerms=n_kernels, device="cuda", x_dim=1, z_dim=4, activation=lambda x: x
        )

    model = rkhs_ouroboros(
        d_data=1,
        n_layers=1,
        d_state=1,
        d_conv=4,
        expand_factor=1,
        tau=1000,
        smooth_len=0.005,
        kernel=kernel,
    )
    print("loading model...")
    model_checkpoint = os.path.join(model_path, "checkpoint_100.tar")

    state = torch.load(model_checkpoint, weights_only=True)
    model.load_state_dict(state["ouroboros"])
    print("done!")
    # opt.load_state_dict(state['opt'])

    print("getting data now...")
    if seg_path == "":
        seg_path = audio_path

    print("loading data")
    loaders_used = os.path.join(model_path, "loaders.pth")
    dls = torch.load(loaders_used)

    audios, sr = get_segmented_audio(
        audio_path,
        seg_path,
        envelope=False,
        context_len=context_len,
        audio_type=audio_type,
        seg_type=seg_filetype,
        max_pairs=max_pairs,
    )

    ### error plot, using original dataloaders

    #### plotting gifs

    #### plotting omega, gamma, kernel weights(?) for different syllable pairs

    #### deal with long audio files (for the budgie case)

    #### deal with no extra padding (for the marmoset case)
