
# **Toxic Online Behavior Identification and Classification**

## 📌 Project Overview
This project aims to develop **machine learning models** to identify and classify **toxic online behavior** such as **hate speech, cyberbullying, and harassment**. By leveraging **Natural Language Processing (NLP)** techniques, the model helps improve **online moderation and user safety** on digital platforms.

## 🏛 **Dataset Overview**
- **Source:** Jigsaw (Kaggle Competition) - Wikipedia talk page comments.
- **Size:** 160,000 comments labeled for toxicity.
- **Categories:**
  - **Toxic**
  - **Severe Toxic**
  - **Obscene**
  - **Threat**
  - **Insult**
  - **Identity Hate**
- **Data Preprocessing Steps:**
  - **Text Cleaning**: Removing special characters, URLs, and non-essential symbols.
  - **Tokenization**: Splitting text into meaningful units.
  - **Stopword Removal**: Eliminating frequent but unimportant words (e.g., "the", "is").
  - **Embeddings**: Using **FastText (300D)** and **Word2Vec (100D)** to capture semantic relationships.

## 🚀 **Models Used**
The project tests both **baseline** and **advanced deep learning models** for toxicity detection:

### 🔹 **Traditional ML Models**
1. **Logistic Regression**
2. **Random Forest** (with **OneVsRestClassifier** for multi-label classification)

### 🔹 **Deep Learning Models**
3. **CNN (Convolutional Neural Network)**
   - **Architecture**: 2 convolutional layers + Fully Connected classifier.
4. **LSTM (Long Short-Term Memory)**
   - **Architecture**: 1 LSTM layer + Fully Connected classifier.
   - **Observation**: Underperformed due to lack of sequential input optimization.
5. **Zero-Shot Classification** (pre-trained models)
   - **Models Used**:
     - `roberta-large-mnli`
     - `facebook/bart-large-mnli`
     - `typeform/distilbert-base-uncased-mnli`
     - `s-nlp/roberta_toxicity_classifier`
     - `martin-ha/toxic-comment-model`
     - `JungleLee/bert-toxic-comment-classification`

## 📊 **Performance Metrics**
Models were evaluated based on:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **Confusion Matrix**

| Model | Toxic | Obscene | Threat | Insult | Identity Hate |
|--------|--------|--------|--------|--------|
| **Logistic Regression** | 0.91 | 0.94 | 0.96 | 1.00 | 0.96 |
| **Random Forest** | 0.91 | 0.94 | 0.97 | 1.00 | 0.96 |
| **CNN** | 0.92 | 0.94 | 0.97 | 1.00 | 0.96 |
| **LSTM** | 0.89 | 0.91 | 0.94 | 1.00 | 0.95 |

**Zero-Shot Classification Results**:
| Model | Toxic | Obscene | Threat | Insult | Identity Hate |
|--------|--------|--------|--------|--------|
| **roberta-large-mnli** | 0.855 | 0.590 | 0.195 | 0.575 | 0.745 |
| **facebook/bart-large-mnli** | 0.780 | 0.590 | 0.180 | 0.555 | 0.695 |
| **s-nlp/roberta_toxicity_classifier** | 0.850 | 0.665 | 0.220 | 0.620 | 0.255 |

## 🔧 **Installation & Setup**
### **1️⃣ Prerequisites**
Ensure you have:
- **Python 3.x**
- **Jupyter Notebook**
- **Required Libraries** (Install using pip if missing):
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow transformers
  ```

### **2️⃣ Running the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/Dx2905/Toxic-Behavior-Detection.git
   cd Toxic-Behavior-Detection
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run `NLP_Projecttry.ipynb` step by step.

## 🔬 **Key Findings**
- **CNN and Random Forest** performed the best, achieving **high accuracy and F1 scores**.
- **Logistic Regression** was interpretable and achieved **similar performance** to CNN.
- **LSTM underperformed**, likely due to the lack of optimized sequential input processing.
- **Zero-Shot models performed well** in general toxicity detection but struggled with fine-grained categories like **"threats"**.

## 🔮 **Future Improvements**
- **Train Transformer Models (BERT, GPT) for fine-grained toxicity classification.**
- **Improve LSTM performance** by optimizing sequential processing.
- **Develop a real-time toxicity detection API** for deployment in social media moderation.

## 📜 **License**
This project is licensed under the **MIT License**. See the [`LICENSE`](https://github.com/Dx2905/CS5100-Foundation-Of-AI/blob/main/LICENSE) file for more details.
