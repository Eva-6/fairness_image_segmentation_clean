import pickle

import matplotlib.pyplot as plt

from happymap.fair_segmentation_boost import FairSegmentationBoost
from happymap.subgroup_metrics_for_image_segmentation import subgroup_metrics


def plot_convergence_algorithm(
    fsb: FairSegmentationBoost,
    save_path="../output/convergence_plot_stomach.png",
    name_task="Stomach",
):
    plt.figure(figsize=(8, 5))
    plt.plot(fsb.global_errs, label="Global Error (IoU)", color="b")
    plt.plot(fsb.violations, label="Max Group Violation", color="r")
    plt.axhline(y=fsb.alpha, color="g", linestyle="--", label="Tolerance Level")
    plt.grid()
    plt.legend()
    plt.title(
        f"FairSegmentationBoost convergence on {name_task} (alpha={fsb.alpha}, step_size={fsb.step_size})"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Error / Violation")
    plt.xlim(0, len(fsb.violations) - 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_metrics_before_after_correction(
    groups_test, y_test, h_test, f_test, new_f_test, fsb, save_path=None
):
    metrics_before = subgroup_metrics(
        groups_test, y_test, h_test, f_test, fsb.metric_name
    )
    metrics_after = subgroup_metrics(
        groups_test, y_test, h_test, new_f_test, fsb.metric_name
    )

    print("==> Test Performance before and after the post-processing algorithm")
    print(f"Global Error before: {metrics_before['agg']['ERR']:.4f}")
    print(f"Global Error after : {metrics_after['agg']['ERR']:.4f}")
    print(f"Max Violation before: {metrics_before['max']['VIOLATION']:.4f}")
    print(f"Max Violation after : {metrics_after['max']['VIOLATION']:.4f}")

    # Optional: save all results in a file
    results_dict = {
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fsb_params": {
            "alpha": fsb.alpha,
            "step_size": fsb.step_size,
            "metric_name": fsb.metric_name,
        },
    }

    if save_path:
        with open(save_path, "wb") as f_out:
            pickle.dump(results_dict, f_out)
    return results_dict


# def plot_groupwise_errors_boxplot(f_before, f_after, h, y, g, group_names, metric_name="IoU", save_path=None):
#     """
#     Displays boxplots of errors by group (before/after correction).

#     Parameters:
#     - f_before: (n_samples,) – initial thresholds
#     - f_after: (n_samples,) – corrected thresholds (FairSegmentationBoost)
#     - h: (n_samples, n_pixels) – scores/logits
#     - y: (n_samples, n_pixels) – binary masks
#     - g: (n_samples, n_groups) – group indicators
#     - group_names : list of group names (same order as columns in g)
#     - metric_name: str – name of the metric to use
#     - save_path: str (optional) – path where to save the graph
#     """
#     error_fn = get_error_function(metric_name)

#     errors_before, errors_after = [], []
#     valid_group_names = []

#     for group_id in range(g.shape[1]):
#         group_mask = g[:, group_id].astype(bool)

#         if not np.any(group_mask):
#             continue  # skip empty group

#         h_group = h[group_mask]
#         y_group = y[group_mask]
#         f_b_group = f_before[group_mask]
#         f_a_group = f_after[group_mask]

#         err_b = [
#             error_fn(np.array([f_b_group[i]]), h_group[i:i+1], y_group[i:i+1])
#             for i in range(len(f_b_group))
#         ]
#         err_a = [
#             error_fn(np.array([f_a_group[i]]), h_group[i:i+1], y_group[i:i+1])
#             for i in range(len(f_a_group))
#         ]

#         errors_before.append(err_b)
#         errors_after.append(err_a)
#         valid_group_names.append(group_names[group_id])  # only keep if group has data

#     # Plot
#     plt.figure(figsize=(14, 6))
#     positions_before = np.arange(len(errors_before)) * 2.0
#     positions_after = positions_before + 0.8

#     b1 = plt.boxplot(errors_before, positions=positions_before, widths=0.6, patch_artist=True)
#     b2 = plt.boxplot(errors_after, positions=positions_after, widths=0.6, patch_artist=True)

#     for patch in b1['boxes']:
#         patch.set_facecolor('skyblue')
#     for patch in b2['boxes']:
#         patch.set_facecolor('orange')

#     plt.xticks(positions_before + 0.4, valid_group_names, rotation=45)
#     plt.ylabel(f"Error (1 - {metric_name})")
#     plt.title("Group-wise Error Before and After FairSegmentationBoost")
#     plt.legend([b1["boxes"][0], b2["boxes"][0]], ["Before", "After"], loc="upper right")
#     plt.grid(True)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path)
#         print(f"✅ Boxplot saved in : {save_path}")

#     plt.show()
