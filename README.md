# BPHP MODY Probability Detector

## Overview
This repository contains the Biomarker Pathway-Preserving Differential Privacy (BPHP) modular code package designed for MODY (Maturity-Onset Diabetes of the Young) probability detection.

## Modular Structure
- `01_BPHP_Model.py`: Core model implementation.
- `02_Statistical_Significance.py`: 10-seed validation and statistical tests.
- `03_Visualizations.py`: Publication-quality graphs and plots.
- `04_Ablation_Studies.py`: Testing component contributions.
- `05_ROC_PR_Curves.py`: Evaluation metrics (ROC-AUC & PR curves).
- `06_Complete_Pipeline.py`: Automated end-to-end execution.

## Usage
Specifically designed for use in Google Colab. 
1. Upload the NHANES dataset.
2. Run the desired module or the complete pipeline.

## Results
BPHP consistently demonstrates improved MODY recall while preserving privacy and biological pathway correlations.
- Standard DP: ~90.2%
- BPHP (Ours): ~94.2%
- Statistical Significance: p = 0.002
