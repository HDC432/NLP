# NLP assignments

# PS1-Reviews.ipynb
# Sentiment Classification with Movie Reviews

## Overview
This project explores how to classify movie reviews as **positive** or **negative**.  
The assignment builds two feature extraction functions:
- **createBasicFeatures**: uses single words (unigrams) to represent each review.  
- **createFancyFeatures**: adds extra features such as bigrams (two-word phrases) and TF-IDF weighting to capture more context.  

Logistic Regression models with **L1** and **L2** regularization are trained and compared.

---

## Practical Uses
Sentiment classification is useful in many real-world applications:
- **Product reviews**: find out if customers are happy or unhappy.  
- **Social media monitoring**: measure public opinion on events, brands, or products.  
- **Customer service**: detect negative feedback quickly.  
- **Market research**: analyze trends in user sentiment to support business decisions.  

---

## Technology Used
- **Python** (Google Colab environment)  
- **scikit-learn**:  
  - `CountVectorizer` for bag-of-words features  
  - `TfidfVectorizer` for TF-IDF weighting  
  - `LogisticRegression` for classification  
- **Numpy / Scipy**: to handle sparse matrices  
- **Cross-validation**: to test model accuracy  
- **Regularization (L1 and L2)**: compare sparse vs smooth models  

---

## Results
- Basic features gave about **83% accuracy**.  
- Fancy features with bigrams and TF-IDF improved accuracy slightly, up to **85% with L2 regularization**.  
- L1 highlighted a few strong words, while L2 used more features and gave smoother results.  

---

## Takeaway
This assignment shows the fundamentals of **text classification**:  
1. Turning raw text into numerical features.  
2. Training a classifier.  
3. Evaluating results with cross-validation.  

The same approach can be scaled up to more advanced models like **word embeddings** or **transformers** for even better performance.

Absolutely 👍 Here’s a clean, professional **README.md** draft for your BERT probing project — it briefly introduces what the project does, its purpose, and its practical applications:

---
# Assignment 3
# 🧠 Probing BERT Representations: Named Entity Recognition and Capitalization

## 📋 Overview

This project investigates what linguistic information is encoded in different layers of the **BERT** language model.
Using **probing experiments**, we test whether BERT’s hidden representations — learned during unsupervised pretraining — contain knowledge about **named entities** and **capitalization**, even though BERT was never explicitly trained for these tasks.

Specifically, we:

1. Extract token embeddings from all **13 layers** of `bert-base-cased` (1 embedding layer + 12 Transformer layers).
2. Train a simple **logistic regression classifier** on each layer to predict:

   * **Named Entity Recognition (NER)** labels (Person, Location, Organization, Misc, and Outside).
   * **Capitalization** (whether a word starts with an uppercase letter).
3. Compare the accuracy across layers using **10-fold cross-validation**.

---

## 🧩 Methodology

1. **Dataset:** CoNLL-2003 English NER dataset (via Hugging Face `datasets` library).
2. **Model:** Pretrained `bert-base-cased` encoder from the Hugging Face `transformers` library.
3. **Tokenization:** Subword tokenization using the BERT tokenizer (`AutoTokenizer`).
4. **Probing:**

   * Extract hidden states for every token at all 13 layers.
   * Train linear classifiers (logistic regression) per layer to predict the target labels.
   * Evaluate performance using 10-fold cross-validation.
5. **Baselines:**

   * Always predict “O” for NER.
   * Always predict “non-capitalized” for capitalization.

---

## 📈 Key Findings

* **NER Probing:**
  Accuracy peaks in the **middle layers (layers 7–9)**, indicating that BERT’s mid-level representations capture the richest semantic and contextual information for entity recognition.

* **Capitalization Probing:**
  Accuracy is highest in the **early layers (layers 1–3)**, which retain surface-level word form and orthographic features.
  Deeper layers abstract away from surface cues toward semantic understanding.

These results support the hypothesis that:

> Lower layers capture **form and syntax**,
> Middle layers encode **semantics and entity structure**,
> Higher layers specialize for **pretraining objectives** like masked language modeling.

---

## 💡 Practical Applications

Understanding how BERT layers represent linguistic features helps in:

* **Model interpretability:** Knowing which layers store which types of information allows researchers to better understand transformer-based models.
* **Efficient fine-tuning:** Selecting the most informative layers can improve performance and reduce computation when adapting BERT for downstream tasks.
* **Feature extraction:** Using specific layers for syntactic or semantic features can enhance applications like information extraction, question answering, or text classification.


---

## 📚 Reference

* Devlin et al., **“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”** (NAACL 2019)
* CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition



