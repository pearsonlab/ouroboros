import torch


def smooth(data: torch.FloatTensor, smooth_len: float) -> torch.FloatTensor:
    """
    Smooth, using a causal uniform moving average. Uses cumsum
    to speed performance. By default, pre-pads with zeros

    inputs
    -----
        - data: data to be smoothed
        - smooth_len: smooth length, in samples, of smoothing window

    returns
    -----
        - smoothed data
    """

    B, L, D = data.shape
    if smooth_len == 1:
        return data
    pad = torch.zeros((B, smooth_len, D), device=data.device)
    try:
        cumsum = torch.cumsum(torch.cat([pad, data], dim=1), dim=1)
    except:
        print(pad.shape, data.shape)
        raise
    return (cumsum[:, smooth_len:, :] - cumsum[:, :-smooth_len, :]) / float(smooth_len)
