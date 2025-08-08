
---

# Brain Tumor Detection using DenseNet121

## 📌 Overview

This project implements a **DenseNet121-based transfer learning model** to classify brain MRI images into four categories:

* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor**

The system leverages the power of **deep learning** and **MRI imaging** to achieve **98.75% accuracy** on the test set, providing a reliable, automated method for brain tumor detection.

---

## 🚀 Features

* **DenseNet121 architecture** with pretrained ImageNet weights
* Advanced **data preprocessing & augmentation**
* Handles **imbalanced datasets** via Random OverSampling
* Implements **early stopping** and **best model checkpointing**
* High **classification accuracy** with detailed performance metrics
* Applicable in **hospital diagnostics** and **health surveys**

---

## 📂 Dataset

The dataset was sourced from:

* **Kaggle**
* **BRATS 2019**
* Other publicly available medical archives

The dataset contains MRI scans labeled into four classes:

```
- Glioma
- Meningioma
- Pituitary
- No Tumor
```

---

## 🛠 Methodology

1. **Data Preprocessing**
   - **Dataset Organization:** MRI scan images are grouped into four labeled categories — *Glioma*, *Meningioma*, *Pituitary*, and *No Tumor*.  
   - **Label Encoding & One-Hot Encoding:** Text labels are converted into numerical format for model compatibility, followed by one-hot encoding for multi-class classification.  
   - **Dataset Balancing:** Applied `RandomOverSampler` from `imblearn` to handle class imbalance, ensuring each category has equal representation.  
   - **Image Resizing:** All images resized to `224x224x3` pixels to match DenseNet121 input size.  
   - **Pixel Normalization:** Pixel values scaled to `[0, 1]` range for faster convergence.  
   - **Data Augmentation:**  
     - Random rotations (±15°)  
     - Horizontal and vertical flips  
     - Gaussian noise addition  
     - Zoom-in/zoom-out  
     - Brightness and contrast shifts  

2. **Model Training**
   - **Architecture:**  
     - Base model: DenseNet121 with pretrained weights from ImageNet.  
     - Added a custom classification head: Global Average Pooling → Dense (ReLU) → Dropout → Dense (Softmax).  
   - **Loss Function:** Sparse Categorical Cross-Entropy — chosen for multi-class classification with integer labels.  
   - **Optimizer:** Adam optimizer with learning rate scheduling for stable training.  
   - **Regularization:** Dropout layers and L2 weight decay to reduce overfitting.  
   - **Training Strategy:**  
     - Early Stopping: Halts training if validation loss doesn’t improve for 10 epochs.  
     - Model Checkpointing: Saves best-performing model based on validation accuracy.  

3. **Evaluation Metrics**
   - **Accuracy:** Overall classification correctness.  
   - **Precision & Recall:** To evaluate prediction reliability and coverage for each tumor type.  
   - **F1-Score:** Harmonic mean of precision and recall for balanced performance measurement.  
   - **Confusion Matrix:** Visual inspection of per-class performance.  
   - **Specificity & Sensitivity:** Critical for medical diagnosis to minimize false negatives.  
   - **Matthews Correlation Coefficient (MCC):** Balanced metric for multi-class evaluation.  
---

## 📊 Results

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 98.57% |
| Test Accuracy  | 98.75% |
| Loss (Test)    | 0.0182 |

The model demonstrates excellent **generalization** and can be integrated into clinical workflows.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/brain-tumor-detection-densenet121.git
cd brain-tumor-detection-densenet121
pip install -r requirements.txt
```

---

## ▶️ Usage

```python
python train.py
```

For inference:

```python
python predict.py --image path_to_image.jpg
```

---

## 📜 Citation

If you use this work, please cite:

```
Avinaba Chakraborty, "Brain Tumor Detection using DenseNet121", 2025.
```

---


