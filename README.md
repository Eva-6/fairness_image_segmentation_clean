# Fairness for Medical Image Segmentation

This repository contains code for studying fairness in medical image segmentation, with a focus on post-processing methods applied to segmentation models.

The project investigates how to adjust model predictions in order to reduce disparities across demographic groups, using ideas inspired by multicalibration and fairness-aware boosting methods (such as HappyMap).

The experiments are conducted on outputs produced by the SuPreM segmentation framework on the TotalSegmentator dataset, and the fairness algorithm is applied to the resulting logits and ground-truth labels.

The main objective of this repository is to provide a clean and modular implementation of fairness post-processing for segmentation models, allowing experiments both on synthetic data and real medical imaging outputs.

## Project Overview

The workflow of the project can be summarized as follows:

1. Run segmentation models (SuPreM) to obtain logits and predictions for medical images.

2. Preprocess the raw outputs into structured tensors/vectors suitable for fairness analysis.

3. Apply a fairness-aware post-processing algorithm (based on HappyMap ideas [1]).

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

The module includes the main implementation of the fairness algorithm, utilities for computing segmentation and fairness metrics, analysis tools to study model behavior across subgroups.

Key files:

```fair_segmentation.py``` – main implementation of the fairness adjustment procedure for segmentation outputs.

```metrics.py``` – computation of standard segmentation performance metrics.

```subgroup_metrics_for_image_segmentation.py``` – evaluation of metrics separately for demographic subgroups.

```analysis_results_happymap.py``` – utilities for analyzing and visualizing fairness results.

### preprocess

Transforms raw segmentation outputs into structured data representations used by the fairness algorithm.
In particular, it constructs vectors corresponding to:

- model scores $h$, a numpy array of shape (n_samples, n_pixels)

- thresholds $f$, a numpy array of shape (n_samples,)

- ground-truth labels $y$, a numpy array of shape (n_samples, n_pixels)

- group attributes $g$, a numpy array of shape (n_samples,n_groups)

Key file:

```preprocess_results.py``` – transforms raw segmentation outputs into the processed vectors and tensors used by the fairness pipeline.

### generate_data
This module contains utilities to **generate synthetic segmentation data and visualize fairness properties**.  
Synthetic experiments allow us to study the behavior of the fairness algorithm in controlled settings where the data-generating process is known.

The synthetic pipeline is mainly used to:
- simulate segmentation prediction scores and labels,
- construct fairness scenarios with controllable bias,
- visualize fairness metrics and subgroup disparities.

Main files:

- **`generate_image.py`**: Generates synthetic segmentation images and associated prediction scores.  

- **`synthetic_thresholds.py`**: Defines synthetic thresholding rules used to convert prediction scores into segmentation decisions.  
  This script allows the simulation of different decision policies across groups, making it possible to introduce controlled fairness violations and study how the fairness algorithm corrects them.

- **`visualize_fairness.py`**: Provides visualization utilities for analyzing fairness behavior on the synthetic datasets.  

### data_organs
This directory stores serialized processed datasets used by the fairness algorithm. These files typically contain the outputs of the preprocessing step and allow experiments to be rerun without regenerating all intermediate data from scratch. Example:data_stomach_0.pkl

### results
This directory contains the raw segmentation outputs produced by the SuPreM models on the TotalSegmentator dataset.
These outputs are later processed and used as inputs for the fairness analysis.

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

## References

### Research Papers

- [1] **Zhun Deng, Cynthia Dwork, and Linjun Zhang.**  
  *HappyMap: A Generalized Multi-calibration Method.*  
  arXiv preprint, 2023.  
  https://arxiv.org/abs/2303.04379

- [2] **Lujing Zhang, Aaron Roth, and Linjun Zhang.**  
  *Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks.*  
  arXiv preprint, 2024.  
  https://arxiv.org/abs/2405.02225

### External Repositories

- **SuPreM** – Medical image segmentation framework  
  https://github.com/MrGiovanni/SuPreM

- **k-rcps** – Risk-controlled prediction sets  
  https://github.com/Sulam-Group/k-rcps

- **semantic_uq** – Semantic uncertainty quantification tools  
  https://github.com/Sulam-Group/semantic_uq
