import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from happymap.metrics import compute_group_errors_weighted, get_error_function

# Adapte si tu as un mapping de noms de classes
LABELS = {"background": 0, "circle": 1, "square": 2, "triangle": 3, "star": 4}


def compute_metric_by_group_with_error_fn(
    masks: np.ndarray,  # (n, H, W) int {0..K}
    logits: np.ndarray,  # (n, H, W) float in [0,1]
    thresholds: np.ndarray,  # (n,)
    groups: np.ndarray,  # (n, G) 0/1
    error_function,  # callable: (lambda_vec, h_list, y_list) -> float
    group_names=None,  # list[str] of length G (ex: ["age_1","age_2","age_3","age_4","m","f"])
    object_name=None,  # None => tous objets (mask!=0) ; sinon "circle"/"square"/... (ou id int)
) -> pd.Series:
    """
    Retourne une Series pandas: <group_name> -> valeur métrique, plus une entrée 'all'.

    On construit pour chaque sample i:
      - y_i : masque binaire flatten (H*W,)
      - h_i : logits flatten (H*W,)
    Puis on appelle error_function sur les sous-ensembles d'indices appartenant à chaque groupe (g=1).
    """
    n = masks.shape[0]
    if group_names is None:
        group_names = [f"g{i}" for i in range(groups.shape[1])]

    # 1) Creation of the binary mask if necessary
    if object_name is None:
        y_bin = (masks != 0).astype(np.uint8)
    else:
        cls_id = LABELS.get(object_name, object_name)  # supporte aussi un id int direct
        y_bin = (masks == int(cls_id)).astype(np.uint8)

    # 2) Flatten: for each sample, we have an array of size n_pixels
    y_list = [y_bin[i].ravel() for i in range(n)]
    h_list = [logits[i].ravel() for i in range(n)]
    lam = np.asarray(thresholds, dtype=float)

    # 3) All: when we look a the global error
    all_val = error_function(lam, h_list, y_list)

    # 4) For each group, we compute the error
    out = {}
    for g_idx, g_name in enumerate(group_names):
        idx = np.where(groups[:, g_idx] == 1)[0]
        size_group = idx.sum() / thresholds.shape[0]
        if idx.size == 0:
            out[g_name] = np.nan
            continue
        val = (
            error_function(lam[idx], [h_list[i] for i in idx], [y_list[i] for i in idx])
            * size_group
        )
        out[g_name] = val

    s = pd.Series(out)
    s["all"] = all_val  # pandas with a error value for each group and for all
    return s


def plot_metric_by_group(
    row,
    ax=None,
    bar_color="lightseagreen",
    bar_width=0.4,
    title="Multiaccuracy Before/After Correction",
):
    """
    Plot metric values per group using standardized labels ($g_{i}$).
    Assumes row.index already contains the labels ($g_{i}$, ..., 'all').
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Retirer "all" s’il est présent
    row = row.drop("all", errors="ignore")

    x_labels = list(row.index)
    x = np.arange(len(x_labels))
    heights = row.values

    ax.bar(x, heights, width=bar_width, color=bar_color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=18)
    ax.set_ylabel("Error", fontsize=18)
    ax.set_xlabel("Groups", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)

    ax.set_title(title, fontsize=18)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    return ax


def plot_and_save_metric_by_group(
    masks: np.ndarray,
    logits: np.ndarray,
    thresholds: np.ndarray,
    groups: np.ndarray,
    metric_name: str,
    save_path: str,
    group_names=None,
    object_name=None,
    title=None,
    bar_color="lightseagreen",
    bar_width=0.4,
):
    """
    Calcule la métrique par groupe (g=1) et la trace via plot_metric_by_group_g1_only,
    puis sauvegarde la figure à save_path. Retourne aussi la Series utilisée.

    metric_name must be in: {'FNR', 'FPR', 'IoU', 'Precision', 'Dice', 'PixelAccuracy'}
    """
    print(f"\nVisualize the fairness among the groups by plotting the error {metric_name}")
    error_function = get_error_function(metric_name)

    # 1) Compute the error by group and save in a pandas/dictionary
    s = compute_metric_by_group_with_error_fn(
        masks=masks,
        logits=logits,
        thresholds=thresholds,
        groups=groups,
        error_function=error_function,
        group_names=group_names,
        object_name=object_name,
    )

    # 2) Plot the differences of errors among groups
    ax = plot_metric_by_group(
        row=s,
        bar_color=bar_color,
        bar_width=bar_width,
        title=title or f"{metric_name} by group",
    )

    # 3) Save it
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title}.png"), bbox_inches="tight")

    plt.close(ax.figure)
    return s

# def visualize_fairness_correction(
#     masks,
#     logits,
#     groups,
#     thresholds_before,
#     thresholds_after,
#     metric_name,
#     group_names,
#     title="Fairness correction",
#     save_path=None,
#     bar_width=0.38,
#     bar_color_before="#1f77b4",
#     bar_color_after="#ff7f0e",
#     group_size_color="#2ca02c",
#     group_size_on_percent=True,
# ):
#     """
#     Affiche sur UNE ligne:
#       - Erreur AVANT (pondérée)
#       - Erreur APRES (pondérée)
#       - Taille de groupe (courbe sur axe secondaire)
#     Toutes les barres partagent la même échelle Y (puisqu'elles sont sur le même axe).
#     """
#     # Calculs
#     err_w_before, _, gsize = compute_group_errors_weighted(
#         masks, logits, groups, thresholds_before, metric_name, group_names
#     )
#     err_w_after, _, _ = compute_group_errors_weighted(
#         masks, logits, groups, thresholds_after, metric_name, group_names
#     )

#     # Remplace NaN (groupes vides) par 0 pour affichage
#     err_w_before = np.nan_to_num(err_w_before, nan=0.0)
#     err_w_after = np.nan_to_num(err_w_after, nan=0.0)

#     # Axe principal : barres (même Y)
#     fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(group_names) + 4), 5))
#     x = np.arange(len(group_names))
#     ax.bar(
#         x - bar_width / 2,
#         err_w_before,
#         width=bar_width,
#         color=bar_color_before,
#         label="Avant",
#     )
#     ax.bar(
#         x + bar_width / 2,
#         err_w_after,
#         width=bar_width,
#         color=bar_color_after,
#         label="Après",
#     )

#     # Y commun = max des deux séries + marge
#     ymax = float(np.nanmax([err_w_before.max(), err_w_after.max()])) if len(x) else 1.0
#     ax.set_ylim(0, max(1e-12, ymax * 1.10))

#     # Axe secondaire : tailles de groupe
#     ax2 = ax.twinx()
#     if group_size_on_percent:
#         ax2.plot(
#             x,
#             gsize * 100.0,
#             color=group_size_color,
#             marker="o",
#             linestyle="-",
#             label="Taille (%)",
#         )
#         ax2.set_ylabel("Taille du groupe (%)")
#         ax2.set_ylim(0, max(1.0, gsize.max() * 100.0 * 1.15))
#     else:
#         ax2.plot(
#             x, gsize, color=group_size_color, marker="o", linestyle="-", label="Taille"
#         )
#         ax2.set_ylabel("Taille du groupe")
#         ax2.set_ylim(0, max(1.0, gsize.max() * 1.15))

#     # Habillage
#     ax.set_xticks(x)
#     ax.set_xticklabels(group_names, rotation=30, ha="right")
#     ax.set_ylabel(f"{metric_name} × taille du groupe")
#     ax.set_title(title)
#     ax.grid(axis="y", linestyle="--", alpha=0.5)

#     # Légendes combinées
#     h1, l1 = ax.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     ax.legend(h1 + h2, l1 + l2, loc="upper right")

#     fig.tight_layout()

#     # Sauvegarde
#     if save_path:
#         # Si c'est un dossier, composer le nom
#         if save_path.lower().endswith(".png"):
#             out_path = save_path
#             folder = os.path.dirname(save_path) or "."
#         else:
#             folder = save_path
#             out_path = os.path.join(
#                 save_path, f"{metric_name}_weighted_error_before_after.png"
#             )
#         os.makedirs(folder, exist_ok=True)
#         plt.savefig(out_path, bbox_inches="tight")
#     plt.show()

def compute_group_sizes(groups: np.ndarray, group_names=None) -> pd.Series:
    """
    Retourne une Series avec la taille (proportion) de chaque groupe, et 'all' = 1.0.
    - groups: (n, G) binaire
    - group_names: liste optionnelle de longueur G
    """
    groups = np.asarray(groups)
    n, G = groups.shape

    if group_names is None:
        group_names = [f"g{i}" for i in range(G)]
    else:
        assert len(group_names) == G, "len(group_names) doit égaler G"

    sizes = groups.mean(axis=0)  # proportion d'images par groupe
    s = pd.Series({name: float(sizes[i]) for i, name in enumerate(group_names)})
    s["all"] = 1.0
    return s

def visualize_fairness_correction(
    masks: np.ndarray,
    logits: np.ndarray,
    thresholds_before: np.ndarray,
    thresholds_after:np.ndarray,
    groups: np.ndarray,
    metric_name: str,
    save_path: str,
    group_names=None,
    object_name=None,
    title=None,
    bar_color="lightseagreen",
    bar_width=0.4,
):
    fig, axes = plt.subplots(
        nrows=1, ncols=3
    )  # figsize to change after

    error_function = get_error_function(metric_name)

    # Plot Before Correcting for fairness
    dict_row_previous_thr = compute_metric_by_group_with_error_fn(
        masks=masks,
        logits=logits,
        thresholds=thresholds_before,
        groups=groups,
        error_function=error_function,
        group_names=group_names,
        object_name=object_name,
    )

    plot_metric_by_group(
        row=dict_row_previous_thr,
        title=f"Before: {metric_name} error by group",
        ax=axes[0],
    )

    # Plot After correcting for fairness
    dict_row_new_thr = compute_metric_by_group_with_error_fn(
        masks=masks,
        logits=logits,
        thresholds=thresholds_after,
        groups=groups,
        error_function=error_function,
        group_names=group_names,
        object_name=object_name,
    )
    plot_metric_by_group(
        row=dict_row_new_thr,
        title=f"After: {metric_name} error by group",
        ax=axes[1],
    )

    # Plot the group sizes on axes[2]
    dict_row_size = compute_group_sizes(groups, group_names)
    plot_metric_by_group(
        row=dict_row_size,
        title=f"Group sizes",
        ax=axes[1],
    )

    # Ajoute le titre avec espace
    fig.suptitle(title, fontsize=18, y=0.93)  # y plus bas que 1 = plus d’espace

    # Espace vertical entre subplots + espace avec le titre
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])  # laisse de la place pour le titre
    fig.subplots_adjust(hspace=0.4)  # espace entre les subplots

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path+f"/{title}.png", bbox_inches="tight")
    plt.show()
