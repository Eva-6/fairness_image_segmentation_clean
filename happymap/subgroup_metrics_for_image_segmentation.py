import numpy as np

from happymap.metrics import get_error_function


def subgroup_metrics(
    groups: np.ndarray,  # (n_samples, n_groups)           – group indicators
    y: np.ndarray,  # (n_samples, n_pixels)           – ground-truth masks
    h: np.ndarray,  # (n_samples, n_pixels)           – scores/logits
    f: np.ndarray,  # (n_samples,)                    – thresholds  λ(x)
    metric_name: str,
):  # "FNR", "Dice", ...
    """
    Returns a dict with two keys:
        agg['ERR']           – overall segmentation error on the whole cohort
        max['VIOLATION']     – max_{g,v}  P(f(x)=v, g) · |E[s|f=v,g]|   (exactly the
                                quantity _find_max_patch maximises)
    """

    # 1) overall pixel-level error replaces the old 'MSE'
    err_fn = get_error_function(metric_name)  # returns a function
    overall_error = err_fn(f, h, y)  # eg. 1 - global Dice

    # 2) worst calibration violation
    h.shape[0]
    worst_val = 0.0  # initialise max

    for g_idx in range(groups.shape[1]):
        idx = np.where(groups[:, g_idx] == 1)[0]
        violation = err_fn(f[idx], [h[i] for i in idx], [y[i] for i in idx])
        if violation > worst_val:
            worst_val = violation

    return {"agg": {"ERR": overall_error}, "max": {"VIOLATION": worst_val}}
