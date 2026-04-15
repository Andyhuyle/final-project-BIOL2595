# 🧬 Multimodal Prostate Cancer Severity Prediction  
**AIH 2025 (Spring 2026) – Final Project**

---

## 📌 Project Overview  
This project develops a **multimodal machine learning framework** to improve prostate cancer (PCa) severity assessment by integrating:

- 🧫 Histopathology biopsy images (PANDA Dataset)
- 🏥 Electronic Health Record (EHR) data (MIMIC IV Dataset) 

In real-world clinical settings, these datasets are often **unpaired** (i.e., they do not correspond to the same patients). To address this, we implement a **weakly supervised learning approach** that aligns modalities using shared disease severity labels (e.g., Gleason score), rather than direct patient matching.

The goal is to learn clinically meaningful representations that enable:
- Severity prediction from EHR data  
- Cross-modal retrieval (EHR → similar biopsy images)  
- Exploration of relationships between clinical and imaging features  

---

## 🎯 Objectives  
- Develop a **multimodal representation learning model** for prostate cancer severity  
- Compare against a **late-fusion baseline model**  
- Evaluate clinical validity, fairness, and interpretability  
- Simulate realistic healthcare data constraints (unpaired datasets)  

---

## 🧠 Methods  

### 1. Data Sources  
**Histopathology Images**
- PANDA prostate biopsy dataset  
- Labels: Gleason score / ISUP grade  

**EHR Data**
- MIMIC-IV database  

Features include:
- Age, race  
- PSA lab frequency  
- Length of stay (LOS)  
- Number of procedures  
- Medication counts  

---

### 2. Data Preprocessing  
- Image normalization and augmentation  
- EHR feature engineering (aggregation per patient visit)  
- Creation of **severity labels** (low / medium / high or Gleason-based)  
- Statistical sampling to reduce bias when combining unpaired datasets  

---

### 3. Model Architectures  

#### 🔹 Primary Model: Weakly Supervised Multimodal Learning  
- CNN → image embeddings  
- MLP → EHR embeddings  
- Shared latent space  
- Contrastive loss:
  - Similar severity → closer embeddings  
  - Different severity → farther apart  

#### 🔹 Baseline Model: Late Fusion  
- Separate CNN (images) and MLP (EHR)  
- Combine predictions via averaging or weighted voting  
- Provides interpretable benchmark  

---

### 4. Training Strategy  
- Train models on severity classification task  
- Use train/test split  
- Optimize using:
  - Cross-entropy loss  
  - Contrastive loss (for multimodal model)  

---

## 📊 Evaluation  

### Performance Metrics  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- (Optional) AUROC for multiclass classification  

### Clinical Usefulness  
- Alignment with known severity indicators:
  - Length of stay  
  - Number of procedures  
  - Lab frequency (e.g., PSA tests)  
- Cross-modal retrieval quality  
- Case-based validation  

### Fairness Analysis  
- Evaluate performance across:
  - Race  
  - Age groups  
- Identify and mitigate bias in predictions  

---

## 📁 Repository Structure  
- current directories:
  - multimodal_model: contains train.py (main model)
  - scripts: contains exploratory data analysis and cohort building code
