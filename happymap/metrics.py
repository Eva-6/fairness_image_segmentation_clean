import numpy as np

# New version of the function which works even if the images don't have the same shape.
# Less efficient because no vectorization


def get_error_function(metric_name):
    """
    Returns a function s(lambda_vec, h, y) that calculates the average error
    of a segmentation metric over a batch of images.
    Works even if the images don't have the same shape.

    Parameters:
    metric_name: str, from {'FNR', 'FPR', 'IoU', 'Precision', 'Dice', 'PixelAccuracy'}

    Returns:
    A function (lambda_vec, h, y) -> float
    """

    def safe_div(numer, denom, fill_value=0.0):
        if denom==0:
            return None
        return np.divide(numer, denom)

    def error_function(lambda_vec, h, y):
        """
        Parameters:
        - lambda_vec: ndarray of shape (n_samples,) - Thresholds.
        - h: ndarray of shape (n_samples,) of arrays - Scoring functions.
        - y: ndarray of shape (n_samples,) of arrays - True labels.

        Returns:
        - float: The average error for the specified metric.
        """
        n_samples = len(h)
        errors = []

        for i in range(n_samples):
            threshold = lambda_vec[i]
            h_i = h[i]
            y_i = y[i]

            pred = h_i >= threshold
            gt = y_i == 1
            not_pred = ~pred
            not_gt = ~gt

            tp = np.sum(pred & gt)
            fp = np.sum(pred & not_gt)
            np.sum(not_pred & gt)
            tn = np.sum(not_pred & not_gt)

            total_pos = np.sum(gt)
            total_neg = np.sum(not_gt)
            total_pred = np.sum(pred)
            total_union = np.sum(gt | pred)
            total_pixels = h_i.size



            if metric_name == "FNR":
                recall = safe_div(tp, total_pos, fill_value=1.0)
                errors.append(1 - recall)

                # print("DEBUG: ",metric_name,"i=",i, "FNR=",1-recall, "\ttp=", tp, "\tfp:", fp, "\ttotal_pos:", total_pos, "\ttotal_neg:", total_neg, "\ttotal_pred", total_pred, "\ttotal_union:", total_union)


            elif metric_name == "FPR":
                errors.append(safe_div(fp, total_neg, fill_value=0.0))

            elif metric_name == "IoU":
                iou = safe_div(tp, total_union, fill_value=1.0)
                errors.append(1 - iou)

            elif metric_name == "Precision":
                precision = safe_div(tp, total_pred, fill_value=1.0)
                errors.append(1 - precision)

            elif metric_name == "Dice":
                dice = safe_div(2 * tp, total_pos + total_pred, fill_value=1.0)
                errors.append(1 - dice)

            elif metric_name == "PixelAccuracy":
                correct = tp + tn
                acc = safe_div(correct, total_pixels)
                errors.append(1 - acc)

            else:
                raise ValueError(f"Unsupported metric_name '{metric_name}'")

        return np.mean(errors)
    
    return error_function

import numpy as np

def compute_tpr(logits, thresholds, y):
    """
    Compute the True Positive Rate (TPR) for binary segmentation.
    """
    logits = np.asarray(logits)
    y = np.asarray(y)

    n_samples = logits.shape[0]
    tprs = []

    for i in range(n_samples):
        thr = thresholds if np.isscalar(thresholds) else thresholds[i]
        pred = logits[i] >= thr

        gt_pos = (y[i] == 1)  # true positives in GT
        tp = np.sum(pred & gt_pos)
        fn = np.sum(~pred & gt_pos)

        if (tp + fn) == 0:
            tpr_i = 0.0  # no positives in ground truth
        else:
            tpr_i = tp / (tp + fn)

        tprs.append(tpr_i)

    return np.mean(tprs)


def _flatten_if_needed(arr):
    """(N,H,W) -> (N, H*W). Laisse (N,P) inchangé."""
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    return arr

def get_group_sizes(groups, group_names):
    group_sizes = np.zeros(groups, dtype=float)

    for g in range(groups):
        mask_g = groups[:, g].astype(bool)
        p_g = mask_g.mean()
        group_sizes[g] = p_g

    return group_sizes

import numpy as np

def compute_fpr(logits, thresholds, y):
    """
    Compute the False Positive Rate (FPR) for binary segmentation.
    """
    logits = np.asarray(logits)
    y = np.asarray(y)

    n_samples = logits.shape[0]
    fprs = []

    for i in range(n_samples):
        thr = thresholds if np.isscalar(thresholds) else thresholds[i]
        pred = logits[i] >= thr

        gt_neg = (y[i] == 0)  # true negatives in GT
        fp = np.sum(pred & gt_neg)
        tn = np.sum(~pred & gt_neg)

        if (fp + tn) == 0:
            fpr_i = 0.0  # no negatives in ground truth
        else:
            fpr_i = fp / (fp + tn)

        fprs.append(fpr_i)

    return np.mean(fprs)

def compute_group_errors_weighted(
    masks, logits, groups, thresholds, metric_name, group_names
):
    """
    Calcule l'erreur moyenne par groupe (via get_error_function),
    puis la pondère par la taille du groupe (proportion d'images).
    Retourne:
      - err_weighted: np.ndarray shape (G,), erreur * proportion
      - err_unweighted: np.ndarray shape (G,), erreur non pondérée
      - group_sizes: np.ndarray shape (G,), proportion de chaque groupe
    """
    err_fn = get_error_function(metric_name)

    H = _flatten_if_needed(logits)
    Y = _flatten_if_needed(masks)
    N, G = groups.shape

    err_unweighted = np.zeros(G, dtype=float)
    group_sizes = np.zeros(G, dtype=float)

    for g in range(G):
        mask_g = groups[:, g].astype(bool)
        p_g = mask_g.mean()
        group_sizes[g] = p_g
        if p_g == 0:
            err_unweighted[g] = np.nan  # groupe vide
            continue

        f_g = thresholds[mask_g]
        H_g = H[mask_g]
        Y_g = Y[mask_g]
        err_unweighted[g] = float(err_fn(f_g, H_g, Y_g))

    err_weighted = err_unweighted * group_sizes  # ce que tu veux afficher
    return err_weighted, err_unweighted, group_sizes
