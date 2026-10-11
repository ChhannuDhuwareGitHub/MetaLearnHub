# MetaLearnHub


MetaLearnHub is a few-shot image classification system based on meta-learning.

## 1. Project Objective

The research objective is to **determine whether a meta-learning model can learn transferable visual knowledge from several cattle-breed classes and then adapt to a previously unseen class using only a few labeled examples.**

The cattle domain is used as a controlled experimental domain for demonstrating the general few-shot learning concept. The goal is *not* universal cattle-breed recognition, but demonstrating **transferable few-shot learning**.

## 2. Few-Shot Learning vs Conventional Classification

*   **Conventional Classification**: A model is trained on a fixed set of classes. To add a new class, the entire model must be retrained.
*   **Meta-Learning**: The model is trained over many diverse *tasks* (episodes) to learn *how to learn*.
*   **Meta-Training**: Learning a generic feature representation (embedding) across a set of known classes.
*   **Unseen Class**: A class strictly held out from meta-training, used to evaluate adaptation.
*   **Support Set**: A tiny set of labeled images (e.g., 5 images) used to build a representation (prototype) for a new class.
*   **Query Set**: Unseen images of that class to classify against the constructed prototype.

## 3. Data Leakage Prevention

The dataset contains base images (e.g., `Ayrshirecattle100.jpg`) and transformed counterparts (`Ayrshirecattle100_c.jpg`).
**Strict Guarantee:** A base image and its `_c` counterpart are grouped together (`sample_group_id`). They are guaranteed to land in the *same* partition (Train, Val, or Test). They are never split across train/test boundaries, preventing catastrophic data leakage.

## 4. How to Run

### Setup
```bash
# Install backend requirements
pip install -r requirements.txt

# Install frontend requirements
cd frontend
npm install
```

### Run
To start both the API and the React Dashboard:
```bash
run.bat
```
*   Frontend: `http://localhost:5173`
*   Backend API: `http://localhost:8000`

### 1. Dataset Inspection
Go to the **Dataset** tab in the UI. It will scan the dataset and report base/_c pairs and identify corrupted files.

### 2. Train ProtoNet (Leave-One-Out)
Go to the **Training** tab. Select a held-out breed (e.g., Red Dane). The model will meta-train on the remaining 4 breeds using Prototypical Networks (ResNet-18).

### 3. Scientific Evaluation
Go to the **Evaluation** tab. Select your trained model and click "Run Evaluation". It will perform multiple K-shot episodes (1, 3, 5, 10 shots) exclusively on the unseen held-out class.

### 4. User-Created Class (Prediction)
Go to the **Few-Shot Predict** tab. 
1. Create a dynamic new class name.
2. Upload a few support images.
3. Build the prototype.
4. Upload a query image to classify it against your new temporary classes!

MetaLearnHub is an AI-powered research platform that uses few-shot learning and meta-learning to identify unseen animal species from just a few labeled images, featuring explainable AI, similarity search, and performance analytics for efficient and reliable classification.

