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
    Compute the global segmentation error and the worst group-level HappyMap violation.

    Returns
    -------
    dict
        A dictionary with:

        - agg["ERR"]: Overall segmentation error on the full cohort:
                E[s(lambda(x), h(x), y)]

        - max["VIOLATION"]: Maximum empirical group violation:
                max_A P(A) * E[s(lambda(x), h(x), y) | x in A]
            where A is one of the protected subgroups encoded in `groups`.

    Notes
    -----
    This quantity is the joint expectation:
        E[1_{x in A} * s(lambda(x), h(x), y)]

    not the conditional group error E[s | x in A] alone.

    This is the stopping criterion used by the HappyMap-style correction algorithm,
    and it must be consistent with the quantity maximized in `_find_max_patch`.

    """

    # 1) overall pixel-level error replaces the old 'MSE'
    err_fn = get_error_function(metric_name)  # returns a function
    overall_error = err_fn(f, h, y)  # eg. 1 - global Dice

    # 2) worst calibration violation
    worst_val = 0.0  # initialise max

    # for g_idx in range(groups.shape[1]):
    #     idx = np.where(groups[:, g_idx] == 1)[0]
    #     violation = err_fn(f[idx], [h[i] for i in idx], [y[i] for i in idx]) # Bug: E(S|A) and not P(A)·E(S|A)
    #     if violation > worst_val:
    #         worst_val = violation

    for g_idx in range(groups.shape[1]):
        idx = np.where(groups[:, g_idx] == 1)[0]
        if len(idx) == 0:
            continue
        group_size = len(idx) / h.shape[0]  # P(A)
        cond_err = err_fn(f[idx], [h[i] for i in idx], [y[i] for i in idx])
        violation = cond_err * group_size  # P(A)·E[s|A]
        if violation > worst_val:
            worst_val = violation

    return {"agg": {"ERR": overall_error}, "max": {"VIOLATION": worst_val}}
