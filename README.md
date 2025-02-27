# Multi-Class and Multi-Label Classification Using SVMs & K-Means Clustering

## Overview

This project explores **multi-class and multi-label classification** using **Support Vector Machines (SVMs)** and **unsupervised clustering (K-Means)** on the **Anuran Calls (MFCCs) dataset**. The dataset contains amphibian call recordings categorized into **Family, Genus, and Species**, making it a multi-label problem.

## Key Objectives

- **Multi-Class and Multi-Label Classification** using SVMs:
  - Implement **binary relevance** by training one SVM per label.
  - Optimize **Gaussian Kernel SVMs** using **10-fold cross-validation**.
  - Compare results using **ℓ1-penalized SVMs**.
  - Apply **SMOTE** or similar techniques to address class imbalance.

- **Unsupervised Clustering (K-Means) for Multi-Label Data**:
  - Cluster data and evaluate label consistency.
  - Compute **Hamming distances** to assess clustering accuracy.

## Dataset: Anuran Calls (MFCCs)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)
- **Description**: Contains **sound recordings** of amphibians, represented as **Mel-frequency cepstral coefficients (MFCCs)**.
- **Labels**:
  - **Family (3 classes)**
  - **Genus (10 classes)**
  - **Species (10+ classes)**
- **Preprocessing**:
  - 70% of the data used for training.
  - Standardization of features for SVM training.

## Methodology

### **1️⃣ Multi-Label Classification with Support Vector Machines (SVMs)**
#### **A. Training Independent Classifiers (Binary Relevance)**
- Train a **separate SVM for each label** (Family, Genus, Species).
- Use **Gaussian Kernel** with **10-fold cross-validation** for hyperparameter tuning.
- Evaluate using **Precision, Recall, F1-score, and AUC**.

#### **B. ℓ1-Penalized SVMs**
- Apply **ℓ1 regularization** to reduce model complexity.
- Compare performance with standard SVMs.

#### **C. Handling Class Imbalance**
- Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to balance classes.
- Analyze the impact on classification performance.

#### **D. Exploring Alternative Multi-Label Methods**
- Implement **Classifier Chain Method** to model label dependencies.
- Evaluate how this approach compares to Binary Relevance.

---

### **2️⃣ K-Means Clustering for Multi-Label Data**
#### **A. Monte Carlo Simulation**
- Run **50 iterations** of K-Means to assess stability.
- Choose **optimal k** using **CH Index, Gap Statistics, or Silhouette Score**.

#### **B. Assigning Labels to Clusters**
- For each cluster, determine the **majority label triplet** (Family, Genus, Species).
- Evaluate clustering results using **Hamming distance, Hamming loss, and Hamming score**.

#### **C. Performance Evaluation**
- Compare the true labels with cluster-assigned labels.
- Compute **confusion matrices** to assess clustering performance.

## Results & Insights

- **SVMs with Gaussian Kernels** performed best for multi-label classification.
- **ℓ1-penalized SVMs** improved interpretability by reducing unnecessary features.
- **SMOTE significantly improved recall** for minority labels.
- **K-Means clustering** struggled with overlapping feature distributions, leading to **higher Hamming loss**.

## Technologies Used

- **Python** (NumPy, Pandas, Matplotlib, Scikit-Learn, Imbalanced-Learn)
- **Machine Learning**:
  - **Supervised**: SVMs (Gaussian Kernel, ℓ1-Penalized)
  - **Unsupervised**: K-Means Clustering
- **Evaluation Metrics**:
  - Multi-label metrics: **Hamming Loss, Hamming Score, ROC-AUC**
  - Classification metrics: **Precision, Recall, F1-Score, Confusion Matrix**
  - Clustering validation: **Silhouette Score, CH Index, Gap Statistics**

## Future Improvements

- **Investigate deep learning approaches** for multi-label classification (e.g., CNNs).
- **Try other clustering techniques** like DBSCAN or Hierarchical Clustering.
- **Implement active learning** to reduce labeled data requirements.

## Author

This project was developed as part of **DSCI 552** at the **University of Southern California (USC)**. It showcases expertise in **multi-label classification, support vector machines, clustering, and class imbalance handling**.

