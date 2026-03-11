import math
from statistics import NormalDist

import numpy as np
from numpy.polynomial.hermite import hermgauss

# ASSUMING THE LOGITS WERE CREATED WITH THE CLASS FROM GENERATE_IMAGE


# MEAN AND VARIANCE
def _sigmoid_stable(t: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(t >= 0, 1.0 / (1.0 + np.exp(-t)), np.exp(t) / (1.0 + np.exp(t)))


def sigmoid_normal_moments(mu: float, sigma: float, n: int = 64) -> tuple[float, float]:
    """
    Return (mean, variance) of Y = sigmoid(X) with X ~ N(mu, sigma^2).

    Uses n-point Gauss–Hermite quadrature:
      E[f(mu + sigma Z)] = (1/sqrt(pi)) * sum_i w_i * f(mu + sigma * sqrt(2) * x_i)
    where (x_i, w_i) are Hermite nodes/weights for e^{-x^2}.

    Args:
        mu: mean of the normal.
        sigma: std dev (>= 0).
        n: number of quadrature nodes (32–128 is typical; default 64).

    Returns:
        (E[Y], Var[Y])
    """
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if sigma == 0:
        s = 1.0 / (1.0 + np.exp(-mu))
        return float(s), 0.0

    # Gauss–Hermite nodes/weights for ∫ e^{-x^2} f(x) dx
    x, w = hermgauss(n)

    # Transform to standard normal expectation
    z = np.sqrt(2.0) * x
    t = mu + sigma * z

    y = _sigmoid_stable(t)
    mean = float((w @ y) / np.sqrt(np.pi))

    y2 = y * y
    second_moment = float((w @ y2) / np.sqrt(np.pi))

    var = second_moment - mean * mean
    return mean, var


# THRESHOLDS
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def good_threshold(p_neg, p_pos):
    """Balanced (EER / Bayes) threshold in probability space."""
    t = 0.5 * (p_neg + p_pos)
    return sigmoid(t)


def threshold_for_target_tpr(p_pos, sigma, tpr_target):
    """Return f so that TPR ≈ tpr_target for rule (sigmoid(X) > f).

    TPR = TP / (TP +FN) = how much we get the true 1

    Small threshold = easy to be labeled 1 = Big TPR = We recognize all the true 1 =  Big FPR
    """
    z = NormalDist().inv_cdf(1 - tpr_target)  # Φ^{-1}(1 - TPR)
    t = p_pos + sigma * z
    return sigmoid(t)


def threshold_for_target_fpr(p_neg, sigma, fpr_target):
    """Return f so that FPR ≈ fpr_target for rule (sigmoid(X) > f).

    FPR = FP/(FP + TN) = how much we make errors on the true 0

    Big threshold = hard to be labeled 1 = Small FPR = We recognize all the true 0 =  Small TPR
    Small threshold = easy to
    """
    z = NormalDist().inv_cdf(1 - fpr_target)  # Φ^{-1}(1 - FPR)
    t = p_neg + sigma * z
    return sigmoid(t)


def rates_at_threshold(p_neg, p_pos, sigma, f):
    """Compute FPR, TPR given threshold f in probability space."""
    # Map back to X-space cutoff
    t = math.log(f / (1 - f))
    Phi = NormalDist().cdf
    FNR = Phi((t - p_pos) / sigma)
    FPR = 1 - Phi((t - p_neg) / sigma)
    return {"FPR": FPR, "TPR": 1 - FNR, "FNR": FNR, "threshold_prob": f}


if __name__ == "__main__":
    p_pos = 0.85
    p_neg = 0.3
    sigma = 0.2

    # Information on the distribution
    print("For y=0: let's look at the distribution of logits")
    mean, var = sigmoid_normal_moments(p_neg, sigma, n=128)
    print(f"Mean: {mean} \tVar={var}")

    print("\nFor y=1: let's look at the distribution of logits")
    mean, var = sigmoid_normal_moments(p_pos, sigma, n=128)
    print(f"Mean: {mean} \tVar={var}")

    # Definitions of thresholds
    print("\nThresholds:")
    print("If I want to have the good threshold")
    good_thr = good_threshold(p_neg, p_pos)
    print(rates_at_threshold(p_neg, p_pos, sigma, good_thr))

    print("\nIf I want to target FPR=0.8")
    thr_target_fpr = threshold_for_target_fpr(p_neg, sigma, fpr_target=0.8)
    print(rates_at_threshold(p_neg, p_pos, sigma, thr_target_fpr))

    print("\nIf I want to target FPR=0.6")
    thr_target_fpr = threshold_for_target_fpr(p_neg, sigma, fpr_target=0.6)
    print(rates_at_threshold(p_neg, p_pos, sigma, thr_target_fpr))

    print("If I want to target TPR=0.8")
    thr_target_tpr = threshold_for_target_tpr(p_pos, sigma, tpr_target=0.8)
    print(rates_at_threshold(p_neg, p_pos, sigma, thr_target_tpr))

    print("\nIf I want to target TPR=0.6")
    thr_target_tpr = threshold_for_target_tpr(p_pos, sigma, tpr_target=0.6)
    print(rates_at_threshold(p_neg, p_pos, sigma, thr_target_tpr))
