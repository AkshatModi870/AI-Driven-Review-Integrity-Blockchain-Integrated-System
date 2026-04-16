# AI-Driven-Review-Integrity-Blockchain-Integrated-System – A Machine Learning Project

## What is this project?

When you shop on Amazon, you see thousands of reviews. Some are helpful (“This cable works perfectly with my laptop”). Others are useless (“Good product” with no details). Amazon already shows a “Helpful” count for each review – but that only comes after many people have voted.

This project builds a **machine learning model** that predicts whether a review will be helpful **before** anyone votes on it. The model looks at:

- The review text (what words are used)
- The star rating (1 to 5 stars)
- The product category (microphone, cable, pop filter, etc.)
- The sentiment (positive or negative tone)
- Emotions detected in the text (joy, anger, sadness, etc. using BERT)

Then it outputs a **helpfulness prediction**: “This review will be helpful” or “This review will not be helpful”.

We also add two extra features for demonstration:

1. **A prediction chain** (like a simple blockchain) – to prove that predictions have not been tampered with.
2. **A reviewer rewards system** – if the model says a review is helpful, the reviewer earns points that can be converted into discount codes.

## Why is this important?

- **For Amazon**: They can promote helpful reviews higher in the list, improving customer experience.
- **For sellers**: They can quickly identify which reviews point to real product problems and fix them.
- **For reviewers**: They can earn rewards for writing truly helpful reviews.

Without a model, Amazon has to wait for human votes. With a model, helpfulness can be estimated instantly.

## What data do we use?

We use an Amazon product review dataset (you provide the file `dataset.csv`). The dataset contains:

- `reviewerID`, `reviewerName` – removed (not useful)
- `asin` – Amazon Standard Identification Number (unique product ID)
- `reviewText` – the actual review
- `overall` – star rating (1–5)
- `helpful` – a pair like [2,3] meaning 2 people found it helpful out of 3
- `reviewTime`, `unixReviewTime` – removed

We calculate **helpful_ratio = helpful[0] / helpful[1]** (if helpful[1] > 0). Our target (label) is: is helpful_ratio > 0.5? That means “more than half of voters found this review helpful”.

## How did we build the model?

### Step 1 – Data cleaning and product classification

- Removed unnecessary columns (`reviewerID`, `reviewerName`, `reviewTime`, `unixReviewTime`).
- Filled missing `overall` values with the mean.
- Each product (`asin`) needed a category. We used a two‑step method:
  - **Keyword matching** – if review text contains words like “mic”, “cable”, “pop filter”, we assign the product to that category.
  - **Machine learning** – for products that did not match any keyword, we trained a Naive Bayes model on the keyword‑matched products and predicted their category.
- Result: each product gets a category: microphone, cable, pop_filter, pedal, adapter, or uncategorized.

### Step 2 – Feature extraction

For each review, we created many features:

| Feature type | How we got it |
|--------------|----------------|
| **TF‑IDF** | Converted review text into 200 most important words (bag‑of‑words with term frequency‑inverse document frequency) |
| **Sentiment score** | Used VADER sentiment analyzer to get a compound score (-1 to +1) |
| **Star rating** | The `overall` column (1–5) |
| **Product category** | One‑hot encoding of the category (e.g., `cat_microphone=1` if product is a microphone) |
| **BERT emotions** | Used a pre‑trained DistilBERT model to extract probabilities for 6 emotions: sadness, joy, love, anger, fear, surprise |

All features were combined into a single numeric matrix `X_all` with thousands of columns (the TF‑IDF features plus the others).

### Step 3 – Training the model

We used **XGBoost** (a powerful gradient boosting algorithm) with GPU acceleration. The target variable `y` was 1 if helpful_ratio > 0.5, else 0.

We split the data into 80% training, 20% testing. The model was trained for 100 rounds using logloss as the evaluation metric.

### Step 4 – Evaluation

On the test set, the model achieved:

 - Accuracy: ~0.78
 - Precision: ~0.76
 - Recall: ~0.72
 - F1-score: ~0.74
 - AUC: ~0.84

(Exact numbers depend on the dataset and random split.)

The ROC curve (in the code) shows that the model is much better than random guessing.

### Step 5 – Prediction chain (blockchain demonstration)

To show that predictions are verifiable and cannot be changed later, we implemented a simple **chain of blocks**:

- Each block contains: prediction result, a short excerpt of the review, a hash of the features, the hash of the previous block, a timestamp.
- If anyone changes a past prediction, the hash will change, breaking the chain.
- You can verify the chain using `verify_chain()`.

This is a small‑scale demonstration of how blockchain can be used to audit machine learning predictions.

### Step 6 – Reviewer rewards system

We also built a **reward system**:

- If a review is both **actually helpful** (helpful_ratio ≥ 0.7) and **predicted as helpful**, the reviewer earns 10 reward points.
- Points can be converted into discount codes (simulated).
- We visualise rewards by product and the distribution of helpfulness ratios.

This shows how a business could incentivise high‑quality reviews.

## How to run this project on your own laptop (Google Colab recommended) but you can use code editors like vscode also?

Initially the code for this project is written for **Google Colab**. It uses a GPU (free in Colab) to run XGBoost and BERT faster.

### Step 1 – Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### Step 2 – Upload your dataset

You need a file named `output.csv` (Amazon review dataset). Upload it to the Colab environment using the file upload icon on the left sidebar.

Alternatively, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

and copy the file to /content/.
