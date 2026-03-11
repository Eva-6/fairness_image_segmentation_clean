import numpy as np
from tqdm import tqdm

from happymap.metrics import get_error_function
from happymap.subgroup_metrics_for_image_segmentation import subgroup_metrics


class FairSegmentationBoost:
    ALLOWED_METRICS = {"FNR", "IoU", "Precision", "Dice", "PixelAccuracy"}

    def __init__(self, metric_name, step_size=0.1):
        assert (
            metric_name in self.ALLOWED_METRICS
        ), f"Unsupported metric_name: {metric_name}. Choose from {self.ALLOWED_METRICS}"
        self.metric_name = metric_name
        self.M = 1  # Clipping variant, same for all allowed metrics
        self.step_size = step_size  # Default step size for now
        self.adjustments = None

    def _find_max_patch(self, f, h, y, groups):
        """
        Find the group with maximum calibration error.
        Returns worst_g that maximizes E[s(v,h,y) * 1_{g(x)=1}]
        The clipping ensures that thresholds stay within bounds (0,1).

        Parameters:
        - f: ndarray of shape (n_samples, ) - Thresholds.
        - h: ndarray of shape (n_samples, n_pixels) - Scoring functions.
        - y: ndarray of shape (n_samples, n_pixels) - True labels.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators.

        Returns:
        - g_t: The group with maximum calibration error.
        """

        # Initialize variables
        n_samples = h.shape[0]
        max_value = -np.inf
        worst_g = None

        for g_id in range(groups.shape[1]):
            group_mask = groups[:, g_id].astype(bool)  # booléen (n_samples,)
            group_size = group_mask.sum() / n_samples

            if group_size == 0:
                continue

            group_thresholds = f[group_mask]  # (n_group,)
            group_h = h[group_mask]  # (n_group, n_pixels)
            group_y = y[group_mask]  # (n_group, n_pixels)

            # expected value of the error among the group : E[s(l,h,y) , g=1]
            error_group = (
                get_error_function(self.metric_name)(group_thresholds, group_h, group_y)
                * group_size
            )

            if error_group >= max_value:
                worst_g = g_id
                max_value = error_group

        return worst_g

    def _update_predictions(self, f_t, g_t, groups):
        # Copy the predictions to avoid modifying the original array
        f_t_new = f_t.copy()

        # Get boolean mask for the selected group
        group_mask = groups[:, g_t].astype(bool)  # (n_samples,)

        # Find the samples in the group
        matching_indices = np.where(group_mask)[0]

        # Add the patch to the matching rows
        f_t_new[matching_indices] -= self.step_size
        f_t_new = np.clip(
            f_t_new, 0, self.M
        )  # should be [-M,M] from the proof, but for our case it is [0,1]

        return f_t_new

    def fit(self, f, h, y, groups, alpha, tol=1e-4, max_iter=500):
        """
        Fit the model to the data by adjusting predictions to achieve fair segmentation.

        Parameters:
        - f: ndarray of shape (n_samples, ) - Thresholds.
        - h: ndarray of shape (n_samples, n_pixels) - Scoring functions.
        - y: ndarray of shape (n_samples, n_pixels) - True labels.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators.
        - alpha: float - The desired calibration level.
        - tol: float - Tolerance for stopping criteria.

        Returns:
        - None, but updates the model's internal state.
        """
        # Verify that h and y have the good shape:
        if h.ndim == 3 and y.ndim == 3:
            h = h.reshape(h.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
        # Begin fitting process
        t = 0
        f_t = f
        group_metrics = subgroup_metrics(groups, y, h, f_t, self.metric_name)
        global_errs = []
        violations = []
        previous_global_err = group_metrics["agg"]["ERR"]  # overall E[s(f,h,y)]
        worst_violation = group_metrics["max"][
            "VIOLATION"
        ]  # worst [ E[s(v,h,y) * 1_{g(x)=1}]
        global_errs.append(previous_global_err)
        violations.append(worst_violation)

        adjustments = {}

        while worst_violation >= alpha and t < max_iter:
            print(f"Iteration: {t}", end="\r")
            g_t = self._find_max_patch(f_t, h, y, groups)
            adjustments[t] = {"g_t": g_t}
            f_t = self._update_predictions(f_t, g_t, groups)
            group_metrics = subgroup_metrics(groups, y, h, f_t, self.metric_name)

            # Current global_err
            current_global_err = group_metrics["agg"]["ERR"]
            global_errs.append(current_global_err)

            # Update previous global_err for the next iteration
            previous_global_err = current_global_err
            worst_violation = group_metrics["max"]["VIOLATION"]
            violations.append(worst_violation)
            t += 1

        print(f"\nModel fitting complete after {t} round(s)")
        self.alpha = alpha
        self.global_errs = global_errs
        self.violations = violations
        self.adjustments = adjustments

    def predict(self, f, groups):
        """
        Predict the adjusted thresholds after fitting the model.

        Parameters:
        - f: ndarray of shape (n_samples, ) - Thresholds.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators.

        Returns:
        - f_t: ndarray of shape (n_samples, ) - Adjusted thresholds.
        """
        if self.adjustments is None:
            raise RuntimeError(
                "The model must be fitted using the `fit` method before calling `predict`."
            )
        f_t = f.copy()
        for t, data in tqdm(self.adjustments.items()):
            g_t = data["g_t"]
            f_t = self._update_predictions(f_t, g_t, groups)

        return f_t
