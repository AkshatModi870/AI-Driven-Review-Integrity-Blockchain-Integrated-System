# AI-Driven Review Integrity System

This project helps online stores (like Amazon) automatically find out which customer reviews are truly helpful and which ones are not.  
It saves time and money by letting businesses focus on real feedback that can improve their products.

---

## 📌 What This Project Does (In Simple Words)

- **Reads product reviews** (text, star ratings, product IDs).
- **Understands emotions** in the review (anger, joy, sadness, etc.) using a smart AI model (BERT).
- **Checks if a review is helpful** to other customers (e.g., "Does it tell the truth about the product?").
- **Creates a tamper‑proof record** of every prediction (like a digital chain) so that no one can change the results later.
- **Rewards helpful reviewers** with virtual discount codes – encouraging good feedback.

---

## 🧠 Why Is This Important for a Business?

- **Save money** – No need to pay people to read thousands of reviews manually.
- **Improve products faster** – Only the most useful reviews are shown to the product team.
- **Increase sales** – Customers trust helpful reviews. Showing them builds confidence.
- **Reward good customers** – Giving discount codes to helpful reviewers makes them come back.

---

## 📁 Files and Folders in This Project (What Each One Does)

| File / Folder | Purpose |
|---------------|---------|
| `logs/` | Contains log files (e.g., `04_15_2026_14_57_20.log`). These record what the program did and any errors. |
| `notebooks/` | Jupyter notebooks for exploring data (`EDA.ipynb`) and training the model (`MODEL TRAINING.ipynb`). |
| `notebooks/data/dataset.csv` | The raw data file with product reviews. |
| `src/` | The main source code of the project. |
| `src/components/` | Reusable parts: `data_ingestion.py` (loads data), `data_transformation.py` (cleans and prepares data), `model_trainer.py` (trains the AI). |
| `src/pipeline/` | Glues everything together. `train_pipeline.py` runs the whole training process. `predict_pipeline.py` makes predictions on new reviews. |
| `src/exception.py` | Custom error handling so the program doesn’t crash suddenly. |
| `src/logger.py` | Writes messages to the log files (helps debugging). |
| `src/utils.py` | Small helper functions (e.g., saving/loading files). |
| `.gitignore` | Tells Git which files/folders to ignore (like temporary files). |
| `README.md` | This file – explains everything. |
| `requirements.txt` | List of Python packages you need to install. |
| `setup.py` | Helps install the project as a Python package. |

---

## 🧪 How the Model Works (Step by Step)

We start with a CSV file that contains Amazon reviews. Each review has:

- `reviewText` – the written review.
- `overall` – star rating (1 to 5).
- `asin` – unique product ID.
- `helpful` – how many people found the review helpful (e.g., `[5, 6]` means 5 out of 6 found it helpful).

### Step 1 – Group Products into Categories
Not every product has a category label. So we:
- Look for keywords in the reviews (e.g., “mic”, “cable”, “pedal”) to guess the category.
- Use a simple machine learning model (Naive Bayes) to assign a category to products that didn’t match any keyword.
- Result: every review gets a product category (`microphone`, `cable`, `pop_filter`, `pedal`, `adapter`).

### Step 2 – Extract Features (Clues for the AI)
From each review we calculate:

- **TF‑IDF** – Converts words into numbers (200 most important words).
- **Sentiment Score** – Using VADER, a score from -1 (very negative) to +1 (very positive).
- **Product Category** – One‑hot encoded (e.g., `cat_microphone = 1`).
- **Star Rating** – The `overall` column as is.
- **Emotions** – Using a BERT model trained to recognise sadness, joy, love, anger, fear, surprise.  
  *Proof in code:* `extract_bert_emotions()` function loads `"bhadresh-savani/distilbert-base-uncased-emotion"`.

### Step 3 – Define What “Helpful” Means
We calculate `helpful_ratio = helpful[0] / helpful[1]` (number of helpful votes divided by total votes).  
If this ratio is **greater than 0.5**, we label the review as **helpful (1)**; otherwise **not helpful (0)**.

### Step 4 – Train the Main AI (XGBoost)
- Split data into 80% training, 20% testing.
- Use XGBoost with GPU acceleration (if available) – fast and accurate.
- The model learns to predict whether a new review will be helpful or not.

### Step 5 – Measure How Good the Model Is
After training, we calculate:

| Metric | Value from our run | What it means |
|--------|-------------------|----------------|
| Accuracy | ~0.85 (example) | 85% of predictions were correct. |
| Precision | ~0.82 | When the model says “helpful”, it is right 82% of the time. |
| Recall | ~0.79 | It finds 79% of all actually helpful reviews. |
| F1‑score | ~0.80 | A balanced average of precision and recall. |
| AUC | ~0.91 | Very good – the model is excellent at separating helpful from unhelpful. |

*Proof in code:* The last cell prints these numbers using `accuracy_score`, `precision_score`, etc.

### Step 6 – Tamper‑Proof Prediction Chain (Blockchain‑like)
Every time we make a prediction, we store it in a “chain”:
- Each block contains the prediction, part of the review, a hash of the features, and the hash of the previous block.
- If someone tries to change an old prediction, the hash will no longer match – the chain will be broken.
- *Proof in code:* `class PredictionChain` with `add_block()` and `verify_chain()`.

### Step 7 – Reward Helpful Reviewers
- The system identifies reviews that were **actually helpful** (real `helpful_ratio` >= 0.7) **and** predicted as helpful by our model.
- Those reviewers get 10 points and a discount code (e.g., `AMA-123456` for 15% off).
- *Proof in code:* `class ReviewerRewards` and `generate_discount_codes()`.

---

## 🖥️ How to Run This Project on Your Own Laptop (Step by Step)

### ✅ What You Need
- A laptop with **Windows, macOS, or Linux**.
- At least **8 GB RAM** (16 GB is better).
- **Python 3.8 or newer** installed.  
  *Don’t have Python?* Go to [python.org](https://python.org), download the installer for your system, and run it. **Make sure to tick “Add Python to PATH”** during installation.

### 📦 Step 1 – Download the Project
- Open this GitHub link in your browser.
- Click the green **“Code”** button → **“Download ZIP”**.
- Extract the ZIP file to a folder on your computer, e.g., `C:\my_project` or `~/Desktop/my_project`.

### 📁 Step 2 – Open a Terminal / Command Prompt
- **Windows:** Press `Win + R`, type `cmd`, press Enter.
- **macOS:** Press `Cmd + Space`, type `terminal`, press Enter.
- **Linux:** Press `Ctrl + Alt + T`.

Then navigate to the project folder:
```bash
cd C:\my_project   # Windows example
cd ~/Desktop/my_project   # Mac/Linux example
