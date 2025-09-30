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

