import math
import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw  # module to draw shapes
from sklearn.model_selection import train_test_split

from generate_data.generate_image import (GROUPS, GeomSegDataset,
                                          SyntheticSegModel, get_good_threshold)
from generate_data.visualize_fairness import (plot_and_save_metric_by_group,
                                              visualize_fairness_correction)
from happymap.analysis_results_happymap import (
    evaluate_metrics_before_after_correction, plot_convergence_algorithm)
from happymap.fair_segmentation_boost import FairSegmentationBoost
from happymap.metrics import compute_fpr, compute_tpr, get_error_function
from happymap.subgroup_metrics_for_image_segmentation import subgroup_metrics

def test_fairness_algorithm(f_all, h_all, y_all, g_all, sample_ids=None, save_dir=None):
    if sample_ids is None:
        sample_ids =np.array([f"img_{i}" for i in range(h_all.shape[0])])

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
    fsb = FairSegmentationBoost(metric_name = "IoU", step_size=0.01)
    alpha = 0.2
    param_fsb = f"(alpha={alpha}, step_size={fsb.step_size},n={len(f_all)})"

    t_start_fit = time.time()
    fsb.fit(f_train, h_train, y_train, groups_train, alpha, tol=1e-3)
    t_end_fit = time.time()
    print(f"Time taken to fit the model: {t_end_fit - t_start_fit:.2f} seconds")

    print("\nAdjustments of the fitting phase:")
    print("Adjustments (step_size, worst_v, worst_g):")
    print(fsb.adjustments)

    # Predict new thresholds
    print("\nPredicting new thresholds on validation set...")
    t_start_predict = time.time()
    new_f_val = fsb.predict(f_val, groups_val)
    t_end_predict = time.time()
    evaluate_metrics_before_after_correction(groups_val, y_val, h_val, f_val, new_f_val, fsb, save_path=f"./output/synthetic_model_metrics_before_after_{param_fsb}.pkl")
    print(f"Time taken to predict: {t_end_predict - t_start_predict:.2f} seconds")

    # Evaluate on the test set BEFORE and AFTER correction:should work
    print("\nAnalysis of results")
    print("\nEvaluating metrics before and after correction on test set...")
    new_f_test= fsb.predict(f_test, groups_test)
    evaluate_metrics_before_after_correction(groups_test, y_test, h_test, f_test, new_f_test, fsb, save_path=f"./output/synthetic_model_metrics_before_after_{param_fsb}.pkl")

    # Plot convergence of the algorithm: should work
    print("\nPlotting convergence of the algorithm...")
    plot_convergence_algorithm(fsb, save_path=f"./output/convergence_plot_synthetic_model_{param_fsb}.png", name_task="Synthetic Model")

    #Visualize what is going on Test Set
    visualize_fairness_correction(
        masks=y_test,
        logits=h_test,
        groups=groups_test,                      # (N,G)
        thresholds_before=f_test,  # (N,)
        thresholds_after=new_f_test,    # (N,)
        metric_name=fsb.metric_name,
        group_names=list(GROUPS.keys()),    # ou tes noms d’attributs
        title=f"{fsb.metric_name} Correction on Test Set",
        save_path=f"{save_dir}/fairness_output"
    )

    #Visualize what is going on Validation Set (Sanity Check)
    new_f_val = fsb.predict(f_val, groups_val)
    visualize_fairness_correction(
        masks=y_val,
        logits=h_val,
        groups=groups_val,                      # (N,G)
        thresholds_before=f_val,  # (N,)
        thresholds_after=new_f_val,    # (N,)
        metric_name=fsb.metric_name,
        group_names=list(GROUPS.keys()),    # ou tes noms d’attributs
        title=f"{fsb.metric_name} Correction on Valisation Set",
        save_path=f"{save_dir}/fairness_output"
    )


def main():
    n_samples = 52
    image_size = 64
    save_dir = "toy_dataset_report"  # set to None if you don't want saving
    p_pos=0.85
    p_neg=0.2
    sigma=0.1
    metric_name="FNR"

    # Dataset parameters
    dataset = GeomSegDataset(image_size=image_size, n_shapes=4)
    model = SyntheticSegModel(p_pos, p_neg, sigma)
    print(f"\nGenerate samples with group, and a model with p_pos={p_pos}, p_neg={p_neg}, sigma={sigma}")
    images, masks, groups = dataset.generate_samples_with_groups(n_samples, save_dir, [0.6, 0.4], [0.5, 0.3, 0.1, 0.1])
    masks = dataset.get_binary_mask(masks, None)
    logits = model.get_logits_for_binary_mask(masks)
    model.save_logits_as_images(logits, save_dir)

    print("All good thresholds:")
    good_thr  = get_good_threshold(model.p_neg, model.p_pos)
    print("Good threshold:",good_thr)
    thresholds=np.full(n_samples, fill_value=good_thr)
    # penalized_groups = ["age_3", "age_4"]
    # print(f"\nGenerate thresholds and penalize {penalized_groups}")
    # thresholds = model.generate_unfair_thresholds_on_groups_by_targeting_tpr(
    #     groups,
    #     penalized_groups=penalized_groups,   # worse performance for age_3
    #     tpr_target=0.4
    # )
    # bad_thr=input("Which bad threshold ?")
    # thresholds = model.choose_unfair_thresholds_on_groups(
    #     groups,
    #     penalized_groups=penalized_groups,   # worse performance for age_3
    #     good_thr=good_thr,
    #     bad_thr=bad_thr
    # )
    pred_mask = (logits >= thresholds[:, None, None]).astype(np.uint8)

    print("images:", images.shape)
    print("masks:", masks.shape)
    print("logits:", logits.shape)
    print("thresholds:", thresholds.shape)
    print("groups:", groups.shape)
    print("pred_mask", pred_mask.shape)

    print("\nWITH THE GOOD THRESHOLD FOR ALL IMAGES")
    print("good thr:", good_thr)
    for err in ['IoU', 'FNR', 'Dice']:
        err_fn = get_error_function(err)  # returns a function
        overall_error = err_fn(thresholds, logits, masks)
        print(f"{err} Error:\t", overall_error)
    print("TPR:\t\t", compute_tpr(logits, thresholds, masks))
    print("FPR:\t\t", compute_fpr(logits, thresholds, masks))

    model.save_predictions_as_images(model.logits, thresholds, save_dir=save_dir,prefix="good_pred")

    print("\nTHE BAD THRESHOLD TPR=0.4 FOR ALL IMAGE")
    thresholds = model.generate_unfair_thresholds_on_groups_by_targeting_tpr(
        groups,
        penalized_groups=['age_1', 'age_2', 'age_3', 'age_4', 'f', 'm'],   # worse performance for age_3
        tpr_target=0.4
    )
    print("bad thr:", np.unique(thresholds) )
    for err in ['IoU', 'FNR', 'Dice']:
        err_fn = get_error_function(err)  # returns a function
        overall_error = err_fn(thresholds, logits, masks)
        print(f"{err} Error:\t", overall_error)
    print("TPR:\t\t", compute_tpr(logits, thresholds, masks))
    print("FPR:\t\t", compute_fpr(logits, thresholds, masks))
    model.save_predictions_as_images(model.logits, thresholds, save_dir=save_dir, prefix="bad_pred_tp0,4")

    print("\nTHE BAD THRESHOLD FPR=0.4 FOR ALL IMAGE")
    thresholds = model.generate_unfair_thresholds_on_groups_by_targeting_fpr(
        groups,
        penalized_groups=['age_1', 'age_2', 'age_3', 'age_4', 'f', 'm'],   # worse performance for age_3
        fpr_target=0.4
    )
    model.save_predictions_as_images(model.logits, thresholds, save_dir=save_dir,prefix="fpr=0,4")
    print("bad thr:", np.unique(thresholds) )
    for err in ['IoU', 'FNR', 'Dice']:
        err_fn = get_error_function(err)  # returns a function
        overall_error = err_fn(thresholds, logits, masks)
        print(f"{err} Error:\t", overall_error)
    print("FPR:\t\t", compute_fpr(logits, thresholds, masks))
    print("FPR:\t\t", compute_fpr(logits, thresholds, masks))
    # Overall report
    # for metric_name in ['FNR', 'IoU', 'Dice', 'Precision']:
    #     err_fn = get_error_function(metric_name)  # returns a function
    #     overall_error = err_fn(thresholds, logits, masks)
    #     print(f"\t\t{metric_name} error:\t {overall_error}")
    #     print(plot_and_save_metric_by_group(
    #         masks=masks,
    #         logits=logits,
    #         thresholds=thresholds,
    #         groups=groups,
    #         metric_name=metric_name,
    #         save_path=f"{save_dir}/fairness_metrics_before_algorithm",
    #         group_names=GROUPS.keys(),
    #         object_name=None,      # None => tous objets (mask != 0) ; "circle" pour une classe précise
    #         title=f"{metric_name} error by group"
    #     ))
    #     break
    
    # # VISUALIZE
    # series = plot_and_save_metric_by_group(
    #     masks=masks,
    #     logits=logits,
    #     thresholds=thresholds,
    #     groups=groups,
    #     metric_name="FNR",
    #     save_path=f"{save_dir}/fairness_output",
    #     group_names=GROUPS.keys(),
    #     object_name=None,      # None => tous objets (mask != 0) ; "circle" pour une classe précise
    #     title="FNR error by group"
    # )
    # print(series)   # valeurs par groupe + 'all'

    # APPLY THE FAIRNESS CORRECTION ALGORITHM
    # test_fairness_algorithm(thresholds, logits, masks, groups, save_dir)

if __name__=="__main__":
    main()