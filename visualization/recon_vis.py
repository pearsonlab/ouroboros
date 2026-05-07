from utils import from_numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import get_spec


def test_pure_tones(
    model,
    sr,
    int_time=0.2,
    start_time=0.05,
    method="RK45",
    remove_dc_offset=True,
    scaled=True,
):

    dt = 1 / sr
    t = np.arange(0, int_time, 1 / sr)
    omegas = np.arange(1000, (sr - 1000) // 2, 1000)
    start = int(round(start_time * sr))
    for omega in omegas:
        x_tone = np.sin(2 * np.pi * t * omega)
        try:
            y_int, *_ = model.integrate(
                from_numpy(x_tone[None, :, None]).to(torch.float64),
                dt,
                st=start_time,
                method=method,
                scaled=scaled,
            )
        except:
            print(f"unable to integrate for omega = {omega},skipping")
            continue
        if remove_dc_offset:
            y_int = y_int - np.nanmean(y_int, axis=1, keepdims=True)
        ax = plt.gca()
        ax.plot(x_tone[start:].squeeze())
        ax.plot(y_int[0, :])
        plt.show()

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        s1, t1, f1, _ = get_spec(
            x_tone,
            sr,
            onset=0,
            offset=x_tone.shape[0] / sr,
            shoulder=0.0,
            interp=False,
            win_len=1028,
            normalize=False,
        )
        s2, t2, f2, _ = get_spec(
            y_int[0, :],
            sr,
            onset=0,
            offset=y_int[0, :].shape[0] / sr,
            shoulder=0.0,
            interp=False,
            win_len=1028,
            normalize=False,
        )
        # print(s1.shape)
        # print(s2.shape)
        vmin = min(np.amin(s1), np.amin(s2))
        vmax = max(np.amax(s1), np.amax(s2))
        # weights = s2[:,100:200].sum(axis=1)/s2[:,100:200].sum()
        # mf = (f2 * weights).sum()
        ax1.imshow(
            s1,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[t1[0], t1[-1], f1[0], f1[-1]],
            aspect="auto",
        )
        ax2.imshow(
            s2,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[t2[0], t2[-1], f2[0], f2[-1]],
            aspect="auto",
        )
        ax1.set_title(rf"true spec $\omega = ${omega}")
        ax2.set_title(r"integrated spec")
        plt.tight_layout()
        plt.show()
        plt.close()


def reconstruct_data(model, sr, audio, method="RK45", remove_dc_offset=True):

    pass
