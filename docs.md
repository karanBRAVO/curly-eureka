# **ML Challenge 2025: Smart Product Pricing Solution**

**Team Name:** AI Wizard

**Team Members:** Karan Yadav, Nikhil Gupta, Krishna Jain, Amrit Singhal

**Submission Date:** 13 October 2025

---

## **1. Executive Summary**

Our approach integrates **multimodal learning** by combining product **text, image, and numerical metadata** into a unified deep learning pipeline. The solution leverages **transformer-based encoders (SentenceTransformer + ViT)** to learn rich semantic and visual embeddings, fused with engineered numerical features for robust price prediction.

---

## **2. Methodology Overview**

### **2.1 Problem Analysis**

We analyzed product catalog data to understand correlations between textual descriptions, product images, and price. Exploratory analysis revealed **high price variance** and **long-tailed distributions**, prompting the use of **log-transformed prices** for regression stability.

**Key Observations:**

- Texts often contained structured "value–unit" pairs.
- Units varied across hundreds of noisy representations.
- Price distributions were highly skewed.

### **2.2 Solution Strategy**

**Approach Type:** Multimodal Hybrid Model
**Core Innovation:** Fusion of **semantic text embeddings**, **visual embeddings**, and **standardized quantitative metrics** (volume, weight, count) through a dense neural network for price regression.

---

## **3. Model Architecture**

### **3.1 Architecture Overview**

```
SentenceTransformer (text) → 384-d embedding
ViT-base (image) → 768-d embedding
Numeric features (3)
        │
        ▼
Concatenate → Dense (1024→512→256) → ReLU → Dropout → Linear → Price
```

### **3.2 Model Components**

**Text Processing Pipeline:**

- **Preprocessing:** Lowercasing, whitespace normalization, regex-based extraction of “value” and “unit.”
- **Model type:** `sentence-transformers/all-MiniLM-L6-v2`
- **Key parameters:** 384-dimensional embeddings.

**Image Processing Pipeline:**

- **Preprocessing:** Resize to 224×224, RGB conversion.
- **Model type:** `google/vit-base-patch16-224`
- **Key parameters:** Hidden size 768, pre-trained on ImageNet.

**Numerical Features:**

- Derived standardized metrics: `volume_ml`, `weight_g`, `count`.

---

## **4. Model Performance**

### **4.1 Validation Results**

- **SMAPE Score:** ~0.18 (validation set, log-scale)

---

## **5. Conclusion**

The **AI Vertex** multimodal model effectively integrates visual, textual, and numeric signals for price estimation. Key gains came from **unit normalization**, **log-price regression**, and **transformer-based fusion**. Future improvements may include **attention-based fusion layers** and **contrastive pretraining** for enhanced multimodal alignment.
