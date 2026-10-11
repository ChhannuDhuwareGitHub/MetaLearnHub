# Scientific Methodology: MetaLearnHub

This document outlines the exact scientific methodology used to evaluate few-shot adaptation in the MetaLearnHub prototype.

## 1. The Dataset
The project utilizes a dataset of 5 cattle breeds. 
*   **Total Images**: ~1,208 files.
*   **Underlying Samples**: ~604 pairs.
*   **Leakage Check**: Each original image (`base_name.jpg`) and its augmented counterpart (`base_name_c.jpg`) are grouped by a shared `sample_group_id`. When splitting data, groups are assigned atomically.

## 2. Leave-One-Class-Out (LOCO) Evaluation
Because there are only 5 breeds, we do not permanently hardcode one unseen class. We employ a configurable Leave-One-Class-Out (LOCO) protocol:
*   **Meta-Training Classes**: 4 breeds.
*   **Held-Out Class**: 1 breed.
*   **Strict Exclusion**: Absolutely zero images (neither base nor `_c`) from the held-out breed are seen during the meta-training phase.

## 3. Meta-Training (Prototypical Networks)
*   **Algorithm**: Prototypical Networks (Snell et al., 2017).
*   **Backbone**: ResNet-18 (ImageNet pretrained, stripped classification head, outputting a 512-dim embedding).
*   **Episodic Training**: The model is trained via N-way K-shot episodes (e.g., 3-way, 5-shot) sampled purely from the 4 meta-training classes.
*   **Disjoint Sets**: Within an episode, support sets and query sets are strictly disjoint.

## 4. Few-Shot Evaluation
Evaluation measures how well the network constructs a representation for the *completely unseen* held-out breed.
*   **Shots Tested**: 1-shot, 3-shot, 5-shot, 10-shot.
*   **Procedure**:
    1.  Sample $K$ support images of the unseen breed.
    2.  Compute the class prototype via the mean embedding of those $K$ images.
    3.  Sample $Q$ distinct query images of the unseen breed, plus query images from known breeds (for N-way comparison).
    4.  Classify queries using Euclidean/Cosine distance to prototypes.
    5.  Calculate Accuracy, Precision, Recall, and Macro F1.
*   **Statistical Rigor**: This process is repeated for $E$ independent episodes (e.g., 50 episodes), reporting the **Mean ± Standard Deviation** (and 95% Confidence Intervals).

## 5. Baseline Comparison
A conventional supervised ResNet-18 is implemented to demonstrate the fundamental limitation of standard classifiers. It is trained on the 4 meta-training classes and achieves high accuracy there. However, it cannot adapt to the held-out class without full network retraining, proving the necessity of the meta-learning paradigm for dynamic adaptation.
