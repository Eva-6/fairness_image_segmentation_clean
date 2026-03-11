import os
import pickle
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from happymap.analysis_results_happymap import (
    evaluate_metrics_before_after_correction, plot_convergence_algorithm,
    plot_groupwise_errors_boxplot)
from happymap.estimate_density import (_estimate_density_upper_bound,
                                       _estimate_ordered_density_upper_bound)
from happymap.fair_segmentation_boost import FairSegmentationBoost
from happymap.metrics import get_error_function
from happymap.subgroup_metrics_for_image_segmentation_version_with_value import \
    subgroup_metrics
from preprocess.preprocess_results import build_data_for_organ

if __name__=="__main__":
    # Preprocess and build the data for stomach without saving (30mn)
    path_total_segmentator = "/export/gaon1/data/jteneggi/TotalSegmentator"
    path_results = f"/cis/home/ezribi1/my_documents/fair_segmentation/results"
    path_data_organ = f"/cis/home/ezribi1/my_documents/fair_segmentation/data_organs"
    
    print("Building data for stomach...")
    dict_stomach = build_data_for_organ(
        organ="stomach",
        path_results=path_results,
        path_total_segmentator=path_total_segmentator,
        group_columns=["age", "gender"],
        save_path=None,
        max_samples=100  # Limit to 128 samples for faster testing
    )
    f_all, h_all, y_all, g_all, sample_ids, group_names = dict_stomach["f"], dict_stomach["h"], dict_stomach["y"], dict_stomach["g"], dict_stomach["sample_ids"], dict_stomach["group_names"]
    
    # Take a small portion of the data for faster testing
    # n_samples_total = len(f_all)
    # n_samples_subset = min(128, n_samples_total)  # Use at most 200 samples for testing
    # indices_subset = np.random.choice(n_samples_total, n_samples_subset, replace=False)

    # f_all, h_all, y_all, g_all, sample_ids = f_all[indices_subset], h_all[indices_subset], y_all[indices_subset], g_all[indices_subset], sample_ids[indices_subset]
    # print(f"Using a subset of {n_samples_subset} samples for testing.")

    # Split into train, val, test (60%, 20%, 20%)
    print("\nSplitting data into train, val, test...")
    idx_train, idx_temp = train_test_split(np.arange(len(f_all)), test_size=0.4, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    print("Train size:", len(idx_train), "\tVal size:", len(idx_val), "\tTest size:", len(idx_test))

    f_train, h_train, y_train, groups_train, sample_ids_train = f_all[idx_train], h_all[idx_train], y_all[idx_train], g_all[idx_train], sample_ids[idx_train]
    f_val, h_val, y_val, groups_val, sample_ids_val = f_all[idx_val], h_all[idx_val], y_all[idx_val], g_all[idx_val], sample_ids[idx_val]
    f_test, h_test, y_test, groups_test, sample_ids_test = f_all[idx_test], h_all[idx_test], y_all[idx_test], g_all[idx_test], sample_ids[idx_test]

    # Initialize and fit the algorithm
    print("\nFitting FairSegmentationBoost...")
    fsb = FairSegmentationBoost(metric_name = "IoU", step_size=0.1)
    alpha = 0.3
    param_fsb = f"(alpha={alpha}, step_size={fsb.step_size},n={len(f_all)})"

    t_start_fit = time.time()
    fsb.fit(f_train, h_train, y_train, groups_train, alpha, tol=1e-3)
    t_end_fit = time.time()
    print(f"Time taken to fit the model: {t_end_fit - t_start_fit:.2f} seconds")

    print("\nAdjustments of the fitting phase:")
    print("Adjustments (step_size, worst_v, worst_g):")
    print(pd.DataFrame(fsb.adjustments))

    # Predict new thresholds
    print("\nPredicting new thresholds on validation set...")
    t_start_predict = time.time()
    f_val_calibrated = fsb.predict(f_val, groups_val)
    t_end_predict = time.time()
    print(f"Time taken to predict: {t_end_predict - t_start_predict:.2f} seconds")

    # Evaluate on the test set BEFORE and AFTER correction
    print("\nAnalysis of results")
    print("\nEvaluating metrics before and after correction on test set...")
    evaluate_metrics_before_after_correction(groups_test, y_test, h_test, f_test, fsb, save_path=f"./output/metrics_before_after_stomach_{param_fsb}.pkl")

    # Plot convergence of the algorithm
    print("\nPlotting convergence of the algorithm...")
    plot_convergence_algorithm(fsb, save_path=f"./output/convergence_plot_stomach_{param_fsb}.png", name_task="Stomach")

    # Generate and plot boxplots by groups
    print("\nGenerating and plotting boxplots by groups...")
    f_val_round = np.round(f_val * fsb.m) / fsb.m
    f_val_corrected = fsb.predict(f_val, groups_val)

    plot_groupwise_errors_boxplot(
        f_val_round,
        f_val_corrected,
        h_val,
        y_val,
        groups_val,
        group_names=group_names,#['age1', 'age2', 'age3', 'age4', 'age5', 'f', 'm'],
        metric_name="IoU",
        save_path=f"./output/boxplot_fairness_by_group_stomach_{param_fsb}.png"
    )