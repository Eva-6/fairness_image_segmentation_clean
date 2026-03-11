# Fairness for Medical Image Segmentation

This repository contains code for studying fairness in medical image segmentation, with a focus on post-processing methods applied to segmentation models.

The project investigates how to adjust model predictions in order to reduce disparities across demographic groups, using ideas inspired by multicalibration and fairness-aware boosting methods (such as HappyMap).

The experiments are conducted on outputs produced by the SuPreM segmentation framework on the TotalSegmentator dataset, and the fairness algorithm is applied to the resulting logits and ground-truth labels.

The main objective of this repository is to provide a clean and modular implementation of fairness post-processing for segmentation models, allowing experiments both on synthetic data and real medical imaging outputs.

## Project Overview

The workflow of the project can be summarized as follows:

1. Run segmentation models (SuPreM) to obtain logits and predictions for medical images.

2. Preprocess the raw outputs into structured tensors/vectors suitable for fairness analysis.

3. Apply a fairness-aware post-processing algorithm (based on HappyMap ideas).

4. Evaluate fairness and performance metrics across demographic subgroups.

The repository contains both:

- experiments on synthetic data (to validate the algorithm),

- experiments on real segmentation outputs from medical imaging models.

## Architecture of the Project

fair_segmentation

├── data_organs                  # Storage of processed data (serialized outputs)

│   └── data_stomach_0.pkl

├── output                       # Folder used to store experiment outputs

├── happymap                     # Implementation of the fairness algorithm

│   ├── analysis_results_happymap.py

│   ├── fair_segmentation.py

│   ├── __init__.py

│   ├── metrics.py

│   └── subgroup_metrics_for_image_segmentation.py

├── preprocess                   # Converts SuPreM raw outputs into vectors (F, G, H, Y)

│   ├── __init__.py

│   └── preprocess_results.py

├── repo                         # External research repositories used in the project

│   └── SuPreM

├── results                      # Raw segmentation results from SuPreM on TotalSegmentator

├── semantic_uq                  # External module for semantic uncertainty quantification

├── k-rcps                       # External repository related to risk control / prediction sets

├── venv                         # Local Python environment (not required for running the code)

├── requirements.txt             # Python dependencies

├── segment_logits.sh            # Script used to generate segmentation logits

├── main.py                      # Runs the fairness algorithm on real segmentation outputs

└── main_fairness_on_synthetic_data.py
                                 # Runs the fairness algorithm on synthetic datasets

## Description of the Main Components
### happymap

Contains the core implementation of the fairness algorithm used in the project.
This includes the logic for:

- computing subgroup metrics,

- adjusting predictions,

- evaluating fairness-aware performance.

### preprocess

Transforms raw segmentation outputs into structured data representations used by the fairness algorithm.
In particular, it constructs vectors corresponding to:

- model scores $h$, a numpy array of shape (n_samples, n_pixels)

- thresholds $f$, a numpy array of shape (n_samples,)

- ground-truth labels $y$, a numpy array of shape (n_samples, n_pixels)

- group attributes $g$, a numpy array of shape (n_samples,)

These vectors are then used as inputs for the fairness procedure.

### repo / SuPreM

This folder contains the SuPreM segmentation framework, an external research project used to produce segmentation predictions on medical images.
It is not part of the fairness algorithm itself but provides the segmentation outputs used in the experiments.

### semantic_uq

External repository related to semantic uncertainty quantification in segmentation models.

### k-rcps

External repository implementing methods related to risk-controlled prediction sets.

## Running the Project
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Generate segmentation outputs

Segmentation logits can be generated using: ```segment_logits.sh```

This produces the files stored in the results/ directory.

3. Preprocess the results

The preprocessing step converts raw outputs into vectors used by the fairness algorithm.

4. Run the fairness algorithm

To run the algorithm on real segmentation outputs: 
```bash
python main.py
```

To run experiments on synthetic data: 
```bash 
python main_fairness_on_synthetic_data.py
```

## Notes

Some folders in the repository (SuPreM, semantic_uq, k-rcps) correspond to external research projects integrated into the pipeline.
They are included here to reproduce the experimental environment but are not part of the fairness method implemented in this repository.

