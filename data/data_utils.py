from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline, make_smoothing_spline
from tqdm import tqdm
from scipy.signal import savgol_filter
from utils import deriv_approx_d2y, deriv_approx_dy


class aud_neur_ds(Dataset):
    """
    think about how to change this to accept lists, rather than arrays
    (probably just map, but make sure this works with arrays still)

    If `categories` is passed (np.ndarray of int, shape (N,)), __getitem__ returns a
    4-tuple (x, dxdt, dx2dt2, category) instead of the legacy 3-tuple. Used by the
    edge-biased sampler so the training step knows which examples are ONSET/OFFSET/MID
    (and should get the cold-start noise IC for the spectral rollout).
    """

    def __init__(self, data, deriv_approx="nine-point", dxdt=[], dx2dt2=[], categories=None):
        self.x = data
        if len(dxdt) > 0:
            self.dxdt = dxdt
            assert np.all(self.x.shape == self.dxdt.shape), print(
                "x and dx must have the same shapes"
            )
        else:
            if deriv_approx == "savgol":
                self.dxdt = savgol_filter(
                    self.x, window_length=5, polyorder=3, deriv=1, axis=1
                )
            elif deriv_approx == "nine-point":
                self.dxdt = deriv_approx_dy(self.x)
        if len(dx2dt2) > 0:
            self.dx2dt2 = dx2dt2
            assert np.all(self.x.shape == self.dx2dt2.shape), print(
                "x and d2x must have the same shapes"
            )

        else:
            if deriv_approx == "savgol":
                self.dx2dt2 = savgol_filter(
                    self.x, window_length=5, polyorder=3, deriv=2, axis=1
                )
            elif deriv_approx == "nine-point":
                self.dx2dt2 = deriv_approx_d2y(self.x)
        # Optional per-example category labels (0=ONSET, 1=OFFSET, 2=MID) used by the
        # edge-biased training loop to pick which examples get the cold-start noise IC.
        if categories is not None:
            assert len(categories) == self.x.shape[0], "categories must match data length"
            self.categories = np.asarray(categories, dtype=np.int64)
        else:
            self.categories = None

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        x = self.x[idx]
        dxdt = self.dxdt[idx]
        dx2dt2 = self.dx2dt2[idx]

        # Store batches as float32 -- the model runs in float32, so float64 here just
        # doubled the host->device transfer bytes and forced a post-copy GPU cast. The
        # derivative numpy arrays remain float64; only the per-item tensor is downcast.
        x, dxdt, dx2dt2 = (
            torch.from_numpy(x).type(torch.FloatTensor),
            torch.from_numpy(dxdt).type(torch.FloatTensor),
            torch.from_numpy(dx2dt2).type(torch.FloatTensor),
        )

        if self.categories is not None:
            return x, dxdt, dx2dt2, int(self.categories[idx])
        return x, dxdt, dx2dt2

    def interpolate_oversample(self, oversample_prop, dt):

        L = self.data.shape[1]
        new_data = []
        for d in self.data:
            old_t = np.arange(0, L * dt + dt / 2, dt)[:L]
            new_t = np.arange(0, L * dt + dt / 2, dt / oversample_prop)[
                : L * oversample_prop
            ]
            spl = make_interp_spline(old_t, d)
            new_data.append(spl(new_t))

        self.data = np.stack(new_data, axis=0)


def time_stretch(data, true_dt, fake_dt):

    L = len(data)
    T = fake_dt * L
    currTimes = np.arange(0, T + fake_dt / 2, fake_dt)
    newTimes = np.arange(0, T + true_dt / 2, true_dt)

    interp = np.interp(newTimes, currTimes, data)

    return interp


def euler_integrate(y0, dy, dt):

    return np.cumsum(dy * dt, axis=1) + y0


def adjusted_euler_integrate(y0, dy, d2y, dt=1):

    dy_adjusted = dy * dt + 1 / 2 * d2y * (dt**2)

    return y0 + np.cumsum(dy_adjusted, axis=1)


def get_loaders(
    data,
    num_workers=4,
    batch_size=32,
    train_size=0.8,
    cv=False,
    seed=None,
    oversample_prop=1,
    dt=1 / 44100,
):

    dls = {}
    if oversample_prop > 1:
        print("oversampling")
    else:
        pass
    test_size = 1 - train_size

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=seed)

    if cv:
        X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=seed)
        dsVal = aud_neur_ds(X_val)
        if oversample_prop > 1:
            dsVal.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
        dls["val"] = DataLoader(
            dsVal, num_workers=num_workers, batch_size=batch_size, shuffle=False,
            pin_memory=True,
        )
    dsTrain, dsTest = aud_neur_ds(X_train), aud_neur_ds(X_test)
    if oversample_prop > 1:
        dsTrain.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
        dsTest.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
    dls["train"] = DataLoader(
        dsTrain, num_workers=num_workers, batch_size=batch_size, shuffle=True,
        pin_memory=True,
    )
    dls["test"] = DataLoader(
        dsTest, num_workers=num_workers, batch_size=batch_size, shuffle=False,
        pin_memory=True,
    )

    return dls


def eval_interp(sample_x, sample_y, lam, n_samples=20):

    choices = np.random.choice(len(sample_x), n_samples, replace=False)
    var, intVar = 0, 0
    for c in choices:
        a, t = sample_y[c], sample_x[c]
        spl = make_smoothing_spline(t, a, lam=lam)
        int = spl(t)
        intVar += np.nanvar(int)
        var += np.nanvar(a)

    return var / n_samples, intVar / n_samples


def interp_samples(sample_x, sample_y, lam):

    new_y = []
    for y, x in tqdm(zip(sample_x, sample_y)):
        spl = make_smoothing_spline(x, y, lam=lam)
        new_y.append(spl(x))

    return np.stack(new_y, axis=0)


def get_loaders_interp(
    data,
    num_workers=4,
    batch_size=32,
    train_size=0.8,
    cv=False,
    seed=None,
    oversample_prop=1,
    dt=1 / 44100,
    starting_lam=1e-15,
):

    dls = {}
    if oversample_prop > 1:
        print("oversampling")
    else:
        pass
    test_size = 1 - train_size

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=seed)
    t = np.arange(0, X_train.shape[1] * dt, dt)[: X_train.shape[1]]
    t_train = np.tile(t[None, :], (X_train.shape[0], 1))

    X_train = interp_samples(t_train, X_train, lam=starting_lam)

    if cv:
        X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=seed)

        t_val = np.tile(t[None, :], (X_val.shape[0], 1))

        X_val = interp_samples(t_val, X_val, lam=starting_lam)
        dsVal = aud_neur_ds(X_val)
        if oversample_prop > 1:
            dsVal.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
        dls["val"] = DataLoader(
            dsVal, num_workers=num_workers, batch_size=batch_size, shuffle=False,
            pin_memory=True,
        )
    t_test = np.tile(t[None, :], (X_test.shape[0], 1))

    X_test = interp_samples(t_test, X_test, lam=starting_lam)
    dsTrain, dsTest = aud_neur_ds(X_train), aud_neur_ds(X_test)
    if oversample_prop > 1:
        dsTrain.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
        dsTest.interpolate_oversample(oversample_prop=oversample_prop, dt=dt)
    dls["train"] = DataLoader(
        dsTrain, num_workers=num_workers, batch_size=batch_size, shuffle=True,
        pin_memory=True,
    )
    dls["test"] = DataLoader(
        dsTest, num_workers=num_workers, batch_size=batch_size, shuffle=False,
        pin_memory=True,
    )

    return dls


# --- Edge-biased loaders for spectral-rollout training -------------------------------

def _stratified_split(N, categories, test_size, seed):
    """Per-category random split into (train_idx, test_idx); preserves the ratio."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cat in np.unique(categories):
        idx = np.where(categories == cat)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(test_size * len(idx)))) if len(idx) > 1 else 0
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def get_loaders_edge(
    data,
    categories,
    num_workers=4,
    batch_size=8,
    train_size=0.8,
    cv=True,
    seed=None,
):
    """
    DataLoaders for the edge-biased segment sampler. `data` is (N, L, 1) and `categories`
    is (N,) int (ONSET=0/OFFSET=1/MID=2). Returns a dict {train, [val,] test} of loaders
    whose datasets carry the matching per-example category labels so the training step
    can drive the cold-start IC selection.

    Stratified split: each category is split independently with the same train_size, so
    val/test never have zero ONSET examples even when the ratio is small.
    """
    data = np.asarray(data)
    categories = np.asarray(categories, dtype=np.int64)
    test_size = 1 - train_size

    train_i, holdout = _stratified_split(len(data), categories, test_size, seed)
    dls = {}
    if cv:
        # Stratified 50/50 split of the held-out tail into val and test (in local coords,
        # then mapped back to globals via the `holdout` index array).
        val_local, test_local = _stratified_split(
            len(holdout), categories[holdout], test_size=0.5, seed=seed
        )
        val_idx = holdout[val_local]
        test_idx = holdout[test_local]
        dsVal = aud_neur_ds(data[val_idx], categories=categories[val_idx])
        dls["val"] = DataLoader(
            dsVal, num_workers=num_workers, batch_size=batch_size, shuffle=False,
            pin_memory=True,
        )
    else:
        test_idx = holdout

    dsTrain = aud_neur_ds(data[train_i], categories=categories[train_i])
    dsTest = aud_neur_ds(data[test_idx], categories=categories[test_idx])
    dls["train"] = DataLoader(
        dsTrain, num_workers=num_workers, batch_size=batch_size, shuffle=True,
        pin_memory=True,
    )
    dls["test"] = DataLoader(
        dsTest, num_workers=num_workers, batch_size=batch_size, shuffle=False,
        pin_memory=True,
    )
    return dls
