import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_groups_from_csv(
    csv_path: str, group_columns: list[str], continuous_bins: dict = {"age": 5}
):
    """ "
        Load group indicators from a CSV file.

        Parameters:
        - csv_path: Path to the CSV file containing group information.
        - group_columns: List of column names in the CSV to be used as group indicators.
        - continuous_bins: optional dict indicating for each numeric column the number of quantiles (e.g. {"age": 5})
    arbitrarily
        Returns:
        - g: A numpy array of shape (n_samples, n_groups) with group indicators.
        - sample_ids: A list of sample identifiers corresponding to each row in g.
    """
    df = pd.read_csv(csv_path, sep=";")
    df = df.dropna(subset=group_columns)

    sample_ids = df["image_id"].tolist()

    group_features = []
    column_names = []  # to stock names

    for col in group_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            K = (
                continuous_bins[col]
                if continuous_bins and col in continuous_bins
                else 4
            )
            binned = pd.qcut(df[col], q=K, labels=[f"{col}_q{i}" for i in range(K)])
            encoded = pd.get_dummies(binned)
        else:
            encoded = pd.get_dummies(df[col], prefix=col)
        group_features.append(encoded)
        column_names.extend(encoded.columns.tolist())  # # save names

    g = pd.concat(group_features, axis=1).to_numpy()

    return g, sample_ids, column_names


def load_predictions_and_labels(
    path_results: str, path_total_segmentator: str, organ: str, max_samples: int = None
):
    """
    Load predicted segmentation scores and ground-truth masks for a specific organ.

    Parameters:
    - path_results: Path to the directory containing prediction results.
    - path_total_segmentator: Path to the TotalSegmentator dataset directory.
    - organ: The organ of interest (e.g., "liver", "spleen").

    Returns:
    - f: A numpy array of shape (n_samples,) with thresholds (initialized to 0.5).
    - h: A numpy array of shape (n_samples, n_pixels) with predicted scores/logits.
    - y: A numpy array of shape (n_samples, n_pixels) with ground-truth masks.
    - sample_ids: A list of sample identifiers corresponding to each row in h and y.
    """
    print("Loading predictions and labels (f,h,y, sample_ids) for organ:", organ)
    sample_ids = sorted([f for f in os.listdir(path_results) if f.startswith("s")])

    if max_samples is not None:
        n_samples_total = len(sample_ids)
        n_samples_subset = min(max_samples, n_samples_total)
        indices_subset = np.random.choice(
            n_samples_total, n_samples_subset, replace=False
        )
        sample_ids = np.array(sample_ids)[indices_subset]

    h_list, y_list, filtered_sample_ids = [], [], []

    for sid in tqdm(sample_ids):
        h_path = os.path.join(path_results, sid, "segmentations", f"{organ}.nii.gz")
        y_path = os.path.join(
            path_total_segmentator, sid, "segmentations", f"{organ}.nii.gz"
        )

        if not os.path.exists(h_path) or not os.path.exists(y_path):
            continue

        # Here: will have to choose best slice (smaller vectors)
        h_data = nib.load(h_path).get_fdata().flatten()
        y_data = nib.load(y_path).get_fdata().flatten()

        h_list.append(h_data)
        y_list.append(y_data)
        filtered_sample_ids.append(sid)

    h = np.array(h_list, dtype=object)
    y = np.array(y_list, dtype=object)
    f = np.full((h.shape[0],), 0.5)

    return f, h, y, filtered_sample_ids


def build_data_for_organ(
    organ,
    path_results,
    path_total_segmentator,
    group_columns,
    save_path=None,
    max_samples=None,
):
    """
    Build the dataset for a specific organ by loading predictions, labels, and group indicators.

    Parameters:
    - organ: The organ of interest (e.g., "liver", "spleen").
    - path_results: Path to the directory containing prediction results.
    - path_total_segmentator: Path to the TotalSegmentator dataset directory.
    - group_columns: List of column names in the CSV to be used as group indicators.
    - save_path: Optional path to save the dictionary with data.

    Returns:
    - A dictionary with keys:
        "f": (n_samples, )              – thresholds
        "h": (n_samples, n_pixels)      – prediction logits per pixel
        "y": (n_samples, n_pixels)      – ground-truth labels per pixel
        "g": (n_samples, n_groups)      – group indicators
        "sample_ids": list[str]        – list of sample IDs
    """
    f, h, y, sample_ids = load_predictions_and_labels(
        path_results, path_total_segmentator, organ, max_samples=max_samples
    )
    g, g_ids, group_names = load_groups_from_csv(
        path_total_segmentator + "/meta.csv", group_columns
    )

    id_to_index = {sid: i for i, sid in enumerate(sample_ids)}
    common_ids = [sid for sid in g_ids if sid in id_to_index]

    indices = [id_to_index[sid] for sid in common_ids]
    g_indices = [g_ids.index(sid) for sid in common_ids]

    dict_organ = {
        "f": f[indices],
        "h": h[indices],
        "y": y[indices],
        "g": g[g_indices],
        "sample_ids": np.array([sample_ids[i] for i in indices]),
        "group_names": group_names,
    }

    print(f"Data for organ '{organ}':")
    print(f"\tNumber of samples: {dict_organ['f'].shape[0]}")
    print(f"\tNumber of groups: {dict_organ['g'].shape[1]}")
    print("\tShape of f:", dict_organ["f"].shape)
    print("\tShape of h:", dict_organ["h"].shape)
    print("\tShape of y:", dict_organ["y"].shape)
    print("\tShape of g:", dict_organ["g"].shape)
    print("\tgroup names:", group_names)

    if save_path:
        print("Saving the data for stomach...")
        with open(save_path, "wb") as f_out:
            pickle.dump(dict_organ, f_out)
        print(f"Data for organ '{organ}' has been built and saved to {save_path}.")
    return dict_organ


if __name__ == "__main__":
    path_total_segmentator = "/export/gaon1/data/jteneggi/TotalSegmentator"
    path_results = f"/cis/home/ezribi1/my_documents/fair_segmentation/results"
    path_data_organ = f"/cis/home/ezribi1/my_documents/fair_segmentation/data_organs"

    # Example usage : load_groups_from_csv
    # g, sample_ids = load_groups_from_csv(
    #     csv_path="/export/gaon1/data/jteneggi/TotalSegmentator/meta.csv",
    #     group_columns=["age", "gender"],
    #     continuous_bins={"age": 5}  # on découpe 'age' en 5 groupes de taille égale
    # )

    # f, h, y, g, sample_ids = load_predictions_and_labels(path_results, path_total_segmentator, "stomach")

    # Example usage : build_data_for_organ
    data = build_data_for_organ(
        organ="stomach",
        path_results=path_results,
        path_total_segmentator=path_total_segmentator,
        group_columns=["age", "gender"],
        save_path=path_data_organ
        + "/data_stomach_0.pkl",  # Optional: specify a path to save the data
    )

    f, h, y, g, sample_ids = (
        data["f"],
        data["h"],
        data["y"],
        data["g"],
        data["sample_ids"],
    )
