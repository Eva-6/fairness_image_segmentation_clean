"""
Minimal synthetic setup to test fairness in image segmentation.

Classes:
- GeomSegDataset: generates grayscale images with random geometric shapes
  (circle, square, triangle, star) and their labeled segmentation masks
  (0=background, 1=circle, 2=square, 3=triangle, 4=star).

- SyntheticSegModel: simulates model logits in [0,1] for binary masks with
  controllable accuracy (p_pos, p_neg) and noise (sigma), and can introduce
  group-specific thresholds to simulate bias.

Use cases:
- Quickly create toy datasets.
- Generate imperfect, biased predictions for fairness evaluation.
"""

import math
import os
from statistics import NormalDist

import numpy as np
from PIL import Image, ImageDraw  # module to draw shapes

LABELS = {"background": 0, "circle": 1, "square": 2, "triangle": 3, "star": 4}
GROUPS = {"age_1": 0, "age_2": 1, "age_3": 2, "age_4": 3, "m": 4, "f": 5}


class GeomSegDataset:
    def __init__(self, image_size=32, n_shapes=2, size_range=None, seed=None):
        """
        Parameters
        ----------
        image_size : int
            Size of the square image (H=W=image_size).
        n_shapes : int
            Maximum number of shapes (randomly choose between 1 and n_shapes).
        size_range : tuple(int, int) | None
            Shape size range (radius or side length). If None, defaults to (img_size/8, img_size/2).
        seed : int | None
            Seed for reproducibility.
        """
        self.image_size = image_size
        self.n_shapes = n_shapes
        self.size_range = size_range
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_samples = None

    def make_sample(self, bg_intensity=255, shape_intensity=0, save_dir=None, idx=None):
        """
        Draw random geometric shapes on an image.

        Returns
        -------
        image : np.uint8 array (H,W) in [0,255]
        mask : np.uint8 array (H,W) with labels {0..4}
        """
        size_range = self.size_range
        if size_range is None:
            size_range = (max(3, self.image_size // 8), max(3, self.image_size // 3))

        H = W = self.image_size

        img = Image.fromarray(np.full((H, W), bg_intensity, dtype=np.uint8))
        draw_img = ImageDraw.Draw(img)
        mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
        draw_mk = ImageDraw.Draw(mask)

        # how_many = self.rng.randint(1, max(1, self.n_shapes))
        how_many = int(self.rng.integers(1, max(2, self.n_shapes + 1)))  # 1..n_shapes

        for _ in range(how_many):
            # cls = rng_py.choice(classes)
            cls = self.rng.choice(["circle", "square", "triangle", "star"])
            # size = rng_py.randint(size_range[0], size_range[1])
            size = int(self.rng.integers(size_range[0], size_range[1] + 1))
            # cx = random.randint(size + 2, W - size - 2)
            cx = int(self.rng.integers(size + 2, W - size - 1))
            # cy = random.randint(size + 2, H - size - 2)
            cy = int(self.rng.integers(size + 2, H - size - 1))

            if cls == "circle":
                r = size
                bbox = (cx - r, cy - r, cx + r, cy + r)
                draw_img.ellipse(bbox, fill=shape_intensity)
                draw_mk.ellipse(bbox, fill=LABELS["circle"])

            elif cls == "square":
                a2 = size
                bbox = (cx - a2, cy - a2, cx + a2, cy + a2)
                draw_img.rectangle(bbox, fill=shape_intensity)
                draw_mk.rectangle(bbox, fill=LABELS["square"])

            elif cls == "triangle":
                a = size
                h = int(a * math.sqrt(3) / 2)
                pts = [
                    (cx, cy - 2 * h // 3),
                    (cx - a // 2, cy + h // 3),
                    (cx + a // 2, cy + h // 3),
                ]
                draw_img.polygon(pts, fill=shape_intensity)
                draw_mk.polygon(pts, fill=LABELS["triangle"])

            else:  # star
                outer = size
                inner = max(3, int(size * 0.45))
                pts = []
                angle0 = -math.pi / 2
                for i in range(10):  # 5 branches => 10 points
                    r = outer if i % 2 == 0 else inner
                    a = angle0 + i * math.pi / 5
                    x = int(round(cx + r * math.cos(a)))
                    y = int(round(cy + r * math.sin(a)))
                    pts.append((x, y))
                draw_img.polygon(pts, fill=shape_intensity)
                draw_mk.polygon(pts, fill=LABELS["star"])

        img_np, mask_np = np.array(img), np.array(mask)

        # Save if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            i = idx if idx is not None else self.rng.integers(1e9)
            img_name = f"image_{i}.png"
            mask_name = f"mask_{i}.npy"
            # save the image as image_i
            Image.fromarray(img_np).save(os.path.join(save_dir, img_name))
            # save the masks
            np.save(os.path.join(save_dir, mask_name), mask_np)
            # print(f"Saved files in {save_dir}: {os.path.join(save_dir, img_name)} and {os.path.join(save_dir, mask_name)}")
        return img_np, mask_np

    def generate_multiple_samples(self, n_samples, save_dir=None):
        """
        Generate multiple samples and optionally save them.
        """
        images, masks = [], []
        for i in range(n_samples):
            img, mk = self.make_sample(save_dir=save_dir, idx=i)
            images.append(img)
            masks.append(mk)
        self.n_samples = n_samples
        return np.stack(images), np.stack(masks)

    def get_binary_mask(self, masks, object_name=None):
        if object_name is not None and object_name not in LABELS:
            raise ValueError(
                f"Invalid object '{object_name}'. Must be one of {list(LABELS.keys())}."
            )
        if object_name is None:
            print("Created one binary mask for all objects at once")
            return masks != 0
        print(f"Created one binary mask for {object_name}")
        return masks == LABELS[object_name]

    def generate_groups(
        self,
        n_samples=None,
        age_proportions=None,
        sex_proportions=None,
        save_dir=None,
        seed=None,
    ):
        """
        Generate a binary group membership matrix for age and sex.

        Each sample belongs to exactly:
        - one age group among age_1..age_4
        - one sex group among m, f

        Args
        ----
        n_samples : int
            Number of samples.
        age_proportions : list[float] | None
            Length 4, proportions for each age group (will be normalized if not summing to 1).
            Default = equal proportions.
        sex_proportions : list[float] | None
            Length 2, proportions for m and f (will be normalized).
            Default = equal proportions.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        group_matrix : np.ndarray
            Shape (n_samples, 6), binary matrix with 1 if sample belongs to the group.
            Column order: age_1, age_2, age_3, age_4, m, f.
        """
        if n_samples is None and self.n_samples is None:
            raise ValueError(
                "n_samples must be provided before generate_groups() is called."
            )
        if n_samples is None:
            n_samples = self.n_samples
        if self.n_samples:
            n_samples = self.n_samples

        rng = np.random.default_rng(seed)

        # Default proportions if not provided
        if age_proportions is None:
            age_proportions = [0.25, 0.25, 0.25, 0.25]
        if sex_proportions is None:
            sex_proportions = [0.5, 0.5]

        # Normalize proportions
        age_proportions = np.array(age_proportions, dtype=float)
        age_proportions /= age_proportions.sum()
        sex_proportions = np.array(sex_proportions, dtype=float)
        sex_proportions /= sex_proportions.sum()

        # Allocate matrix
        groups = np.zeros((n_samples, len(GROUPS)), dtype=np.uint8)

        # Sample ages
        age_choices = rng.choice(4, size=n_samples, p=age_proportions)
        for i in range(4):
            groups[age_choices == i, i] = 1

        # Sample sexes
        sex_choices = rng.choice(2, size=n_samples, p=sex_proportions)
        groups[sex_choices == 0, GROUPS["m"]] = 1
        groups[sex_choices == 1, GROUPS["f"]] = 1

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "groups.npy")
            np.save(file_path, groups)
        return groups

    def generate_samples_with_groups(
        self,
        n_samples,
        save_dir=None,
        sex_proportions=None,
        age_proportions=None,
        seed=None,
    ):
        """
        generate numpy images, masks, groups
        where
        images: (n_samples, size_image, size_image)
        masks:  (n_samples, size_image, size_image)
        groups: (n_samples, 6) - because groups = 4 ages categories and 2 genders
        """
        # 1. Génération des images et masques
        images, masks = self.generate_multiple_samples(n_samples, save_dir=save_dir)

        # 2. Génération des groupes
        groups = self.generate_groups(
            n_samples=n_samples,
            age_proportions=age_proportions,
            sex_proportions=sex_proportions,
            save_dir=save_dir,
            seed=seed,
        )
        self.images = images
        self.masks = masks
        self.groups = groups

        return images, masks, groups


class SyntheticSegModel:
    def __init__(self, p_pos=0.85, p_neg=0.30, sigma=0.2, seed=None):
        """
        Parameters
        ----------
        p_pos : float
            Mean target probability for positive pixels.
        p_neg : float
            Mean target probability for negative pixels.
        sigma : float
            Standard deviation of added Gaussian noise.
        seed : int | None
            Seed for reproducibility.
        """
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def get_logits_for_binary_mask(self, binary_mask):
        """
        Generate logits in [0,1] for a binary mask.
        """
        eps = self.rng.normal(0, self.sigma, size=binary_mask.shape)
        logits = (
            self.p_neg
            + (self.p_pos - self.p_neg) * binary_mask.astype(np.float32)
            + eps
        )
        logits = 1 / (1 + np.exp(-logits))
        print("Computed logits:", logits.shape)
        self.logits = logits.astype(np.float32)
        return logits

    def choose_unfair_thresholds_on_groups(
        self, groups, penalized_groups, bad_thr=0.9, good_thr=None
    ):
        """
        Generate thresholds biased against certain groups.
        Good threshold everywhere but on the penalized group, bad threshold close to 1
        """
        if good_thr is None:
            good_thr = get_good_threshold(self.p_neg, self.p_pos)
        thresholds = np.full(groups.shape[0], good_thr)
        for pg in penalized_groups:
            thresholds[groups[:, GROUPS[pg]] == 1] = bad_thr
        self.thresholds = np.asarray(thresholds, dtype=float)
        return self.thresholds

    def generate_unfair_thresholds_on_groups_by_targeting_fpr(
        self, groups, penalized_groups, fpr_target
    ):
        """
        Make recall worse: choose a high cutoff so FPR is small → TPR drops.
        """
        good_thr = get_good_threshold(self.p_neg, self.p_pos)
        bad_thr = threshold_for_target_fpr(
            self.p_neg, self.sigma, fpr_target=fpr_target
        ) 
        thresholds = np.full(groups.shape[0], good_thr, dtype=float)
        for pg in penalized_groups:
            thresholds[groups[:, GROUPS[pg]] == 1] = bad_thr
        self.thresholds = np.asarray(thresholds, dtype=float)
        return self.thresholds

    def generate_unfair_thresholds_on_groups_by_targeting_tpr(
        self, groups, penalized_groups, tpr_target
    ):
        """
        Generate thresholds biased against certain groups.
        Good threshold everywhere but on the penalized group, bad threshold

        Make specificity worse: choose a low cutoff so TPR is big → FPR increases
        """
        good_thr = get_good_threshold(self.p_neg, self.p_pos)
        bad_thr = threshold_for_target_tpr(
            self.p_pos, self.sigma, tpr_target=tpr_target
        )

        thresholds = np.full(groups.shape[0], good_thr)
        for pg in penalized_groups:
            thresholds[groups[:, GROUPS[pg]] == 1] = bad_thr
        self.thresholds = np.asarray(thresholds, dtype=float)
        return self.thresholds

    def describe_theoretic_rates(self):
        print(
            rates_at_threshold_vec(self.p_neg, self.p_pos, self.sigma, self.thresholds)
        )

    def save_logits_as_images(self, logits_array, save_dir, prefix="logits"):
        """
        logits_array: np.ndarray of shape (n_samples, H, W)
        save_dir: where to save the PNGs
        """
        os.makedirs(save_dir, exist_ok=True)

        for i in range(logits_array.shape[0]):
            # On s'assure que les valeurs sont dans [0,1]
            logits_norm = np.clip(logits_array[i], 0.0, 1.0)
            # Convertir en intensité 0-255
            img_uint8 = ((1-logits_norm) * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8)
            img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

        print(f"Saved {logits_array.shape[0]} logits images in {save_dir}")

    def save_predictions_as_images(self, logits_array, thresholds, save_dir, prefix="pred"):
        """
        logits_array: np.ndarray of shape (n_samples, H, W)
        save_dir: where to save the PNGs
        """
        os.makedirs(save_dir, exist_ok=True)

        for i in range(logits_array.shape[0]):
            # On s'assure que les valeurs sont dans [0,1]
            logits_norm = np.clip(logits_array[i], 0.0, 1.0)
            pred = (logits_norm >= thresholds[i])
            # Convertir en intensité 0-255
            img_uint8 = ((1-pred) * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8)
            img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

        print(f"Saved {logits_array.shape[0]} logits images in {save_dir}")


# THRESHOLDS
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def get_good_threshold(p_neg, p_pos):
    """Balanced (EER / Bayes) threshold in probability space."""
    t = 0.5 * (p_neg + p_pos)
    return sigmoid(t)


def threshold_for_target_tpr(p_pos, sigma, tpr_target):
    """Return f so that TPR ≈ tpr_target for rule (sigmoid(X) > f).

    TPR = TP / (TP +FN) = how much we get the true 1
    TPR low -> FNR high
    Big threshold = hard to be labeled 1 = Low TPR = We don't recognize all the true 1 =  Big FPR
    """
    z = NormalDist().inv_cdf(1 - tpr_target)  # Φ^{-1}(1 - TPR)
    t = p_pos + sigma * z
    return sigmoid(t)


def threshold_for_target_fpr(p_neg, sigma, fpr_target):
    """Return f so that FPR ≈ fpr_target for rule (sigmoid(X) > f).

    FPR = FP/(FP + TN) = how much we make errors on the true 0

    FPR high -> Precision, IoU, Dice low
    Big threshold = hard to be labeled 1 = Small FPR = We recognize all the true 0 =  Small TPR
    """
    z = NormalDist().inv_cdf(1 - fpr_target)  # Φ^{-1}(1 - FPR)
    t = p_neg + sigma * z
    return sigmoid(t)


def rates_at_threshold(p_neg, p_pos, sigma, f):
    """
    theoretic rates for a single threshold real f
    Compute FPR, TPR given threshold f in probability space."""
    # Map back to X-space cutoff
    t = math.log(f / (1 - f))
    Phi = NormalDist().cdf
    FNR = Phi((t - p_pos) / sigma)
    FPR = 1 - Phi((t - p_neg) / sigma)
    return {"FPR": FPR, "TPR": 1 - FNR, "FNR": FNR, "threshold_prob": f}


def rates_at_threshold_vec(p_neg, p_pos, sigma, f_vec):
    "theoretic rates for a vector of thresholds"
    f_vec = np.asarray(f_vec, dtype=float)
    t = np.log(f_vec / (1.0 - f_vec))
    Phi = NormalDist().cdf
    FNR = np.vectorize(Phi)((t - p_pos) / sigma)
    FPR = 1 - np.vectorize(Phi)((t - p_neg) / sigma)
    return {"FPR": FPR, "TPR": 1 - FNR, "FNR": FNR, "threshold_prob": f_vec}


def main_v1():
    # ==== 1. PARAMETERS ====
    n_samples = 10
    image_size = 128
    save_dir = "toy_dataset"  # set to None if you don't want saving

    # Dataset parameters
    dataset = GeomSegDataset(image_size=image_size, n_shapes=3)

    # # Test 1 sample
    # dataset.make_sample(save_dir=save_dir, idx=105)
    # mask = np.load(save_dir+"/mask_105.npy")
    # print("TEST ONE SAMPLE")

    # Test many samples
    # print("TEST GENERATE SAMPLES")
    # images, masks = dataset.generate_multiple_samples(n_samples, save_dir=save_dir)
    # print(f"Generated dataset: images {images.shape}, masks {masks.shape}")

    # # Test generate groups for the samples
    # groups = dataset.generate_groups(
    #     n_samples,
    #     age_proportions=[0.5, 0.3, 0.1, 0.1],
    #     sex_proportions=[0.6, 0.4],
    #     save_dir=save_dir,
    #     seed=42
    # )
    # groups=np.load(save_dir+"/groups.npy")
    # print(f"Groups shape: {groups.shape} (age_1, age_2, age_3, age_4, m, f)")
    # print(groups)

    print("Test all together: generate samples with groups")
    images, masks, groups = dataset.generate_samples_with_groups(
        n_samples, save_dir, [0.6, 0.4], [0.5, 0.3, 0.1, 0.1]
    )
    print("images:", images.shape)
    print("masks:", masks.shape)
    print("groups:", groups.shape)

    # Model parameters (p_pos=prob for positive pixels, p_neg=prob for negative pixels)
    print("TEST GENERATE SYNTHETIC SEGMENTATION MODEL")
    model = SyntheticSegModel(p_pos=0.85, p_neg=0.3, sigma=0.2)

    print("Test creation of logits")
    binary_mask = dataset.get_binary_mask(masks, None)
    logits = model.get_logits_for_binary_mask(binary_mask)
    model.save_logits_as_images(logits, save_dir)

    # SIMULATE MODEL LOGITS + BIAS
    thresholds = model.choose_unfair_thresholds_on_groups(
        groups,
        penalized_groups=["age_3"],  # worse performance for age_3
        bad_threshold=0.9,
        good_threshold=0.4,
        mid_threshold=0.6,
    )
    print(thresholds)

    pred_mask = (logits >= thresholds).astype(np.uint8)

    # VISUALIZE
    # group_names = GROUPS.keys()  # adapte à ton ordre de colonnes
    # # Exemple: métrique = ton error_function configurée pour "multiaccuracy error"
    # series = plot_and_save_metric_by_group_g1(
    #     masks=masks,
    #     logits=logits,
    #     thresholds=thresholds,
    #     groups=groups,
    #     metric_name="FNR",
    #     save_path="toy_dataset/fairness_output/Results Before Correction FNR.png",
    #     group_names=group_names,
    #     object_name=None,      # None => tous objets (mask != 0) ; "circle" pour une classe précise
    #     title="FNR error by group (g=1)"
    # )
    # print(series)   # valeurs par groupe + 'all'


def predict_masks_from_logits(logits, thresholds):
    """
    logits: (N,H,W) in [0,1]
    thresholds: (N,) in [0,1]
    returns: (N,H,W) uint8 {0,1}
    """
    thr = np.asarray(thresholds, dtype=float)[:, None, None]
    return (logits >= thr).astype(np.uint8)


if __name__ == "__main__":
    ds = GeomSegDataset(image_size=32, n_shapes=2, seed=0)
    imgs, masks, groups = ds.generate_samples_with_groups(
        n_samples=10, save_dir="toy_dataset", sex_proportions=[0.6, 0.4]
    )
    bin_masks = ds.get_binary_mask(masks)  # (N,H,W) for “any shape”

    model = SyntheticSegModel(p_pos=0.85, p_neg=0.30, sigma=0.2, seed=0)
    logits = model.get_logits_for_binary_mask(bin_masks)  # (N,H,W)
    thresholds = model.generate_unfair_thresholds_on_groups_by_targeting_fpr(
        groups, "m", fpr_target=0.4
    )

    print("Imgs:", imgs.shape)
    print("Binary masks:", bin_masks.shape)
    print("logits:", logits.shape)
    print("thresholds:", thresholds.shape)

    # # Fair baseline (same threshold for all)
    # good_thr = get_good_threshold(model.p_neg, model.p_pos)
    # thr_vec  = np.full(logits.shape[0], good_thr, dtype=float)
    # rates    = rates_at_threshold_vec(model.p_neg, model.p_pos, model.sigma, thr_vec)
    # print("Theoretic (fair) rates:", {k: v[0] if isinstance(v, np.ndarray) else v for k,v in rates.items()})

    # # Penalize recall for group "m" (lower TPR)
    # print("Penalization of age_1")
    # print(groups[:,GROUPS["age_1"]])
    # thr_bad_recall = model.generate_unfair_thresholds_on_groups_by_targeting_fpr(groups, ["age_1"], fpr_target=0.05)
    # print("Bad-recall thresholds (first 8):", thr_bad_recall)

    # # Penalize specificity for group "f" (higher FPR)
    # thr_bad_spec = model.generate_unfair_thresholds_on_groups_by_targeting_tpr(groups, ["f"], tpr_target=0.95)
    # print("Bad-spec thresholds (first 8):", thr_bad_spec)

    # # Turn thresholds into predictions
    # preds = predict_masks_from_logits(logits, thr_bad_recall)  # (N,H,W)
    # print("Preds shape:", preds.shape)
