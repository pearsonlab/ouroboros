from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm
from utils import sst, sse, euler_step_k
import matplotlib.pyplot as plt
import os
import glob
from model.model import Ouroboros, ArneodoOuroboros
from model.kernels import fullPolyModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from typing import Tuple

# filters removed


def save_model(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    location: str,
    n_layers: int = 2,
    d_state: int = 1,
    d_conv: int = 4,
    expand_factor: int = 4,
    max_saved: int = 5,
):
    """
    save a model. requires information about the structure of
    the model as well, so that it can be reproduced upon loading
    also, automatically removes older saves. It would probably be best to just keep
    the 'best' epoch, but whatevah

    inputs
    -----
        - model: Ouroboros model to be saved
        - opt: optimizer for said model
        - location: filename in which to save the model
        - n_layers: number of mamba layers in encoder
        - d_state: internal state dim of ssm
        - d_conv: width of mamba convolutional kernel
        - expand_factor: factor by which we expand input prior to feeding into ssm
        - max_saved: max number of checkpoint files. we only keep the 5 most recent, assuming
            they're saved with epoch number in the tag
    """

    current_saves = glob.glob(os.path.join("/".join(location.split("/")[:-1]), "*.tar"))
    if len(current_saves) >= max_saved:
        save_epochs = [
            int(s.split("/")[-1].split(".tar")[0].split("_")[-1]) for s in current_saves
        ]
        save_order = np.argsort(save_epochs)
        ordered_saves = [current_saves[o] for o in save_order]
        for ii in range(len(current_saves) - max_saved + 1):
            os.remove(ordered_saves[ii])
    # "poly" models carry a kernel module; the Arneodo parameterization does not.
    parameterization = "poly" if hasattr(model, "kernel") else "arneodo"
    sd = {
        "ouroboros": model.state_dict(),
        "opt": opt.state_dict(),
        "tau": model.tau,
        "smooth_len": model.smooth_len,
        "n_layers": n_layers,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand_factor": expand_factor,
        "parameterization": parameterization,
        "drive_lowpass_ms": getattr(model, "drive_lowpass_ms", 0.0),
        "keep_const": getattr(model, "keep_const", False),
    }
    try:
        sd["n_kernel"] = model.kernel.nTerms
    except (KeyError, AttributeError):
        pass

    torch.save(sd, location)


def load_model(
    location: str,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """
    load a model. requires that save files contained information about the structure of
    the model as well, so that it can be reproduced upon loading


    inputs
    -----
        - location: save file directory you wish to load from. Assumes files include checkpoint number
            and loads the most recent checkpoint

    returns
    -----
        - model: an Ouroboros with weights and structure specified by savefile
        - opt: optimizer for that Ouroboros
        - scheduler: learning rate scheduler for that optimizer
        - epoch: training epoch corresponding to this checkpoint
    """

    model_files = glob.glob(os.path.join(location, "*.tar"))
    epochs = [int(m.split("/checkpoint_")[-1].split(".tar")[0]) for m in model_files]
    most_recent = np.argsort(epochs)[-1]
    location = model_files[most_recent]
    print(f"loading from {location}")

    sd = torch.load(location, weights_only=False)
    try:
        n_layers = sd["n_layers"]
        d_state = sd["d_state"]
        d_conv = sd["d_conv"]
        expand_factor = sd["expand_factor"]
    except KeyError:
        n_layers = 2
        d_state = 1
        d_conv = 4
        expand_factor = 4
    parameterization = sd.get("parameterization", "poly")
    if parameterization == "arneodo":
        model = ArneodoOuroboros(
            d_data=1,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            tau=sd["tau"],
            smooth_len=sd["smooth_len"],
            drive_lowpass_ms=sd.get("drive_lowpass_ms", 0.0),
        )
    else:
        try:
            # since this is a trained model and we only use lambda during training, i set it to 1 here...
            # but probably should have saved it. oh well! we set to 1 for compatibility with all my saves.
            kernel = fullPolyModule(
                nTerms=sd["n_kernel"],
                device="cuda",
                x_dim=1,
                z_dim=2,
                activation=lambda x: x,
                lam=1,
            )

            model = Ouroboros(
                d_data=1,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                tau=sd["tau"],
                smooth_len=sd["smooth_len"],
                kernel=kernel,
                drive_lowpass_ms=sd.get("drive_lowpass_ms", 0.0),
                keep_const=sd.get("keep_const", False),
            )
        except:
            print("no kernel in savefile!")
            raise

    print(f"model tau: {model.tau}")
    opt = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(opt, factor=0.75, patience=5, min_lr=1e-10)
    model.load_state_dict(sd["ouroboros"])
    opt.load_state_dict(sd["opt"])

    return model, opt, scheduler, epochs[most_recent]


def train(
    model,
    optimizer,
    loss_fn,
    loaders,
    scheduler=None,
    nEpochs=100,
    val_freq=25,
    runDir=".",
    dt=1 / 44100,
    vis_freq=100,
    smoothing=False,
    reg_weights=False,
    start_epoch=0,
    model_info={},
    save_freq=0,
    k_rollout=0,
    lambda_k=0.3,
    k_rollout_units="rescaled",
) -> Tuple[
    list[float], list[Tuple[int, float, float]], nn.Module, torch.optim.Optimizer
]:
    """
    main train loop for an Ouroboros model. takes a model, an optimizer, loss function, and dataloaders;
    trains a model, and returns that model and train statistics

    inputs
    -----
        - model: an Ouroboros
        - optimizer: an optimizer for that Ouroboros
        - loss_fn: a training objective. In all cases, we used MSE
        - loaders: Dataloaders, one for the train set and one for the test set
        - scheduler: a learning rate scheduler for the optimizer. Optional, but we use this
        - nEpochs: number of passes through the entire training dataset
        - val_freq: frequency of looking at the validation set to test performance
        - runDir: directory to save tensorboard logs in
        - dt: sampling timestep of the data
        - vis_freq: frequency with which to visualize model reconstructions
        - smoothing: whether or not to smooth latents during training. we do not,but you can
        - reg_weights: whether or not to regularize weights, using procedure laid out in the paper. we always do
        - start_epoch: starting epoch for training. if training a model, this should be 0; if loading a trained model
            this might be higher
        - model_info: dictionary of model structure specification. used for saving models
        - save_freq: how often (in epochs) to save your model
        - k_rollout: if >=2, add a small-k Euler-step consistency loss to the train
            objective (see `utils.euler_step_k`). The model's predicted second
            derivative is rolled forward `k_rollout` samples and required to
            reproduce the ground-truth (y, dy/dt) trajectory at every intermediate
            step. k must be << one carrier cycle (~16 samples at sr=40 kHz) so
            pointwise MSE doesn't reward amplitude collapse via phase drift.
            See docs/small_k_rollout_plan.md. Default 0 disables the term and
            preserves the legacy single-step ẍ-MSE training objective.
        - lambda_k: relative weight on the k-step rollout loss
            (`total = single_step + lambda_k * k_rollout`). Ignored if
            k_rollout<2.
        - k_rollout_units: "rescaled" (default) or "physical". Selects the
            time-unit system the k-step rollout is computed in:
              * "rescaled": state = (y, dy/ds), step = ds = dt/τ. y and dy/ds
                have comparable scales (dy/ds ~ τ·2π·f·y; with τ=1/40000 and
                f≈2.5 kHz, ratio ~0.4), so MSE(y)+MSE(dy/ds) is
                well-conditioned and commensurable with the single-step ẍ MSE
                (also in rescaled units). No τ² division, no dxdt clone --
                the model's post-mutation dxdt is already dy/ds and yhat is
                already d²y/ds². This is the recommended choice.
              * "physical": state = (y, dy/dt), step = dt. dy/dt is ~10⁴×
                larger than y, so MSE(dy/dt) dominates MSE(y) by ~10⁸ and the
                whole k-step term is on a different scale from the single-step
                ẍ MSE. Useful only for reproducing the May-30 matrix runs in
                docs/small_k_rollout_plan.md §5 (which used this mode).

    returns
    ----
        - train_losses: train loss per epoch.
        - val losses: list of tuples, containing time point (in gradient updates) of val, val loss, and val regularization cost
        - trained Ouroboros
        - optimizer for that trained Ouroboros
    """

    writer = SummaryWriter(log_dir=runDir)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(start_epoch, nEpochs), desc="training model"):
        model.train()

        for idx, batch in enumerate(
            loaders["train"], start=epoch * len(loaders["train"])
        ):
            optimizer.zero_grad()
            x, dxdt, dx2dt2 = batch  # each is bsz x seq len x 1
            bsz, _, n = x.shape

            x = x.to("cuda").to(torch.float32)
            dxdt = dxdt.to("cuda").to(torch.float32)
            dx2 = (
                dx2dt2.to("cuda").to(torch.float32) / (dt**2) * model.tau**2
            )  # rescale dx2, rather than model output

            # model.forward mutates dxdt in-place (dxdt *= tau; dxdt /= dt -> dy/ds).
            # The physical-units k-rollout needs the ORIGINAL per-sample dxdt, so clone
            # before the model call. The rescaled-units k-rollout uses the post-mutation
            # dxdt (= dy/ds) directly, so no clone is needed.
            need_persamp = (k_rollout >= 2) and (k_rollout_units == "physical")
            dxdt_persamp = dxdt.clone() if need_persamp else None

            dx2hat, weights = model(x, dxdt, dt, smoothing)  # state: B x L x SD

            yhat = dx2hat

            y = dx2
            L = x.shape[1]

            if vis_freq > 0:
                if (idx % vis_freq) == 0:
                    sse_sample = sse(yhat[:1, :, :1], y[:1, :, :1])
                    sst_sample = sst(y[:1, :, :1])
                    r2_sample = (1 - sse_sample / sst_sample).item()

                    on = np.random.choice(L - 600)
                    resids = (
                        (y[0, :, 0] - yhat[0, :, 0]).detach().cpu().numpy()
                    )  # * dt**2
                    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
                        nrows=1, ncols=6, sharey=False, figsize=(20, 5)
                    )

                    ax1.plot(yhat[0, :, 0].detach().cpu().numpy(), label="model")
                    ax1.set_title("model")
                    ax1.set_ylabel("a.u.")
                    ax2.plot(
                        y[0, :, 0].detach().cpu().numpy(),
                        label="data",
                        color="tab:orange",
                    )
                    ax2.set_title("data")
                    ylims = ax2.get_ylim()
                    (l1,) = ax3.plot(
                        y[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                        label="data",
                        color="tab:orange",
                    )
                    (l2,) = ax3.plot(
                        yhat[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                        label="model",
                        color="tab:blue",
                    )
                    ax4.spines[["left", "right", "top", "bottom"]].set_visible(False)
                    ax4.set_xticks([])
                    ax4.set_yticks([])
                    ax4.legend([l1, l2], ["Data", "Model"])

                    ax5.plot(resids, label="res", color="tab:red")
                    ax5.set_title("residuals")
                    ax6.hist(resids, bins=100, density=True)
                    xlims = ax6.get_xlim()
                    sd = np.nanstd(resids)
                    def px(x):
                        return (1 / np.sqrt(2 * np.pi * sd**2)) * np.exp(-(x**2) / (2 * sd**2))
                    
                    xax = np.linspace(xlims[0], xlims[1], 1000)
                    yax = px(xax)
                    ax6.plot(xax, yax, color="tab:red")

                    ax1.set_ylim(ylims)
                    ax2.set_ylim(ylims)
                    ax3.set_ylim(ylims)
                    ax5.set_ylim(ylims)
                    ax6.set_xlim(xlims)

                    fig.suptitle(f"sample r2: {r2_sample: 0.4f}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(runDir, f"y_vs_yhat_batch_{idx}.svg"))
                    plt.close()

            ##################################

            train_loss = loss_fn(y, yhat[:, :L, :])

            # default objective is the data loss; the polynomial parameterization adds a
            # weight-complexity penalty when reg_weights=True. Parameterizations without
            # kernel weights (e.g. ArneodoOuroboros) train with reg_weights=False.
            total_loss = train_loss
            if reg_weights:
                B, L, P, P = weights.shape
                lam_mat = torch.arange(
                    P, dtype=torch.float32, device=model.kernel.device
                )[None, None, :, None].expand(B, L, -1, P)

                w = model.kernel.lam ** (lam_mat + lam_mat.transpose(-1, -2))  # new

                penalty = (
                    (w * weights**2).sum(dim=(-1, -2, -3)).mean()
                )  # new (sum over weights, time), average over samples. equivalent to squared L2 norm
                # we take mean over samples to match the loss fn we use (MSE, with mean over samples)
                total_loss = train_loss + penalty

            if k_rollout >= 2:
                # Small-k Euler-step rollout consistency (see docs/small_k_rollout_plan.md).
                # euler_step_k returns the ground-truth (y, dy) and the recurrently-stepped
                # predictions at every intermediate step 1..k, stacked on the last dim;
                # loss_fn is applied to both. Gradient flows through the model's d2y output.
                if k_rollout_units == "rescaled":
                    # rescaled time s = t/τ: dxdt (post-mutation) = dy/ds, yhat = d²y/ds².
                    # Step size = ds = dt/τ. Two MSE terms are commensurable with y.
                    ds = dt / model.tau
                    (y_gt, y_pred), (dy_gt, dy_pred) = euler_step_k(
                        x, dxdt, yhat, ds, k=k_rollout
                    )
                elif k_rollout_units == "physical":
                    # physical time t: dy/dt = dxdt_persamp/dt, d²y/dt² = yhat/τ².
                    # MSE(dy/dt) dominates MSE(y) by ~10⁸; lambda_k absorbs this.
                    d2y_phys = yhat / (model.tau ** 2)
                    dy_phys = dxdt_persamp / dt
                    (y_gt, y_pred), (dy_gt, dy_pred) = euler_step_k(
                        x, dy_phys, d2y_phys, dt, k=k_rollout
                    )
                else:
                    raise ValueError(
                        f"k_rollout_units must be 'rescaled' or 'physical', got "
                        f"{k_rollout_units!r}"
                    )
                k_loss = loss_fn(y_gt, y_pred) + loss_fn(dy_gt, dy_pred)
                total_loss = total_loss + lambda_k * k_loss

            total_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())
            # we should probably be adding val loss here too...ugh
            writer.add_scalar("Loss/train", train_loss.item(), idx)
            if reg_weights:
                writer.add_scalar("Penalty/train", penalty.item(), idx)
            if k_rollout >= 2:
                writer.add_scalar("Loss/k_rollout", k_loss.item(), idx)

        if epoch % val_freq == 0:
            model.eval()
            vl = 0.0
            vp = 0.0

            for idx, batch in enumerate(
                loaders["val"], start=epoch * len(loaders["train"])
            ):
                with torch.no_grad():
                    x, dxdt, dx2dt2 = batch  # each is bsz x seq len x 1
                    bsz, _, n = x.shape

                    x = x.to("cuda").to(torch.float32)
                    dxdt = dxdt.to("cuda").to(torch.float32)
                    dx2 = dx2dt2.to("cuda").to(torch.float32) / (dt**2) * model.tau**2

                    dx2hat, weights = model(x, dxdt, dt, smoothing)

                    yhat = dx2hat

                    y = dx2
                    L = y.shape[1]
                    if vis_freq > 0:
                        if idx == epoch * len(loaders["train"]):
                            sse_sample = sse(yhat[:1, :, :1], y[:1, :, :1])
                            sst_sample = sst(y[:1, :, :1])
                            r2_sample = (1 - sse_sample / sst_sample).item()

                            on = np.random.choice(L - 600)

                            resids = (
                                (y[0, on : on + 600, 0] - yhat[0, on : on + 600, 0])
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
                                nrows=1, ncols=6, sharey=False, figsize=(20, 5)
                            )

                            ax1.plot(
                                yhat[0, on : on + 600, 0].detach().cpu().numpy(),
                                label="model",
                            )
                            ax1.set_title("model")
                            ax1.set_ylabel("a.u.")
                            ax2.plot(
                                y[0, on : on + 600, 0].detach().cpu().numpy(),
                                label="data",
                                color="tab:orange",
                            )
                            ax2.set_title("data")
                            ylims = ax2.get_ylim()
                            (l1,) = ax3.plot(
                                y[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                                label="data",
                                color="tab:orange",
                            )
                            (l2,) = ax3.plot(
                                yhat[0, on + 300 : on + 350, 0].detach().cpu().numpy(),
                                label="model",
                                color="tab:blue",
                            )
                            ax4.spines[["left", "right", "top", "bottom"]].set_visible(
                                False
                            )
                            ax4.set_xticks([])
                            ax4.set_yticks([])
                            ax4.legend([l1, l2], ["Data", "Model"])

                            ax5.plot(resids, label="res", color="tab:red")
                            ax5.set_title("residuals")
                            ax6.hist(resids, bins=100, density=True)
                            xlims = ax6.get_xlim()
                            sd = np.nanstd(resids)
                            def px(x):
                                return (1 / np.sqrt(2 * np.pi * sd**2))* np.exp(-(x**2) / (2 * sd**2))
                            
                            xax = np.linspace(xlims[0], xlims[1], 1000)
                            yax = px(xax)
                            ax6.plot(xax, yax, color="tab:red")

                            ax1.set_ylim(ylims)
                            ax2.set_ylim(ylims)
                            ax3.set_ylim(ylims)
                            ax5.set_ylim(ylims)
                            ax6.set_xlim(xlims)
                            fig.suptitle(f"sample r2: {r2_sample: 0.4f}")
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(runDir, f"y_vs_yhat_batch_{idx}_test.svg")
                            )
                            plt.close()

                    val_loss = loss_fn(y, yhat[:, :L, :])
                    
                    if reg_weights:
                        B, L, P, P = weights.shape
                        lam_mat = torch.arange(
                            P, dtype=torch.float32, device=model.kernel.device
                        )[None, None, :, None].expand(B, L, -1, P)

                        exps = lam_mat + lam_mat.transpose(-1, -2)
                        w = model.kernel.lam**exps
                        penalty = (w * weights**2).sum(dim=(-1, -2)).mean()

                    vl += val_loss.item()

                    if reg_weights:
                        vp += penalty.item()

            if scheduler:
                scheduler.step(vl / len(loaders["val"]))
            val_losses.append(
                (
                    epoch * len(loaders["train"]),
                    vl / len(loaders["val"]),
                    vp / len(loaders["val"]),
                )
            )
            writer.add_scalar("Loss/validation", vl / len(loaders["val"]), idx)
            writer.add_scalar("Penalty/validation", vp / len(loaders["val"]), idx)

            if epoch % save_freq == 0:
                save_model(
                    model,
                    optimizer,
                    location=os.path.join(runDir, f"checkpoint_{epoch}.tar"),
                    n_layers=model_info["n layers"],
                    d_state=model_info["d state"],
                    d_conv=model_info["d conv"],
                    expand_factor=model_info["expand factor"],
                )
    writer.close()
    return train_losses, val_losses, model, optimizer
