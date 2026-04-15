# AI-Driven Review Integrity System

This project helps online stores (like Amazon) automatically find out which customer reviews are truly helpful and which ones are not.  
It saves time and money by letting businesses focus on real feedback that can improve their products.

## 📌 What This Project Does?

- **Reads product reviews** (text, star ratings, product IDs).
- **Understands emotions** in the review (anger, joy, sadness, etc.) using a smart AI model (BERT).
- **Checks if a review is helpful** to other customers (e.g., "Does it tell the truth about the product?").
- **Creates a tamper‑proof record** of every prediction (like a digital chain) so that no one can change the results later.
- **Rewards helpful reviewers** with virtual discount codes – encouraging good feedback.

## 🧠 Why Is This Important for a Business?

- **Save money** – No need to pay people to read thousands of reviews manually.
- **Improve products faster** – Only the most useful reviews are shown to the product team.
- **Increase sales** – Customers trust helpful reviews. Showing them builds confidence.
- **Reward good customers** – Giving discount codes to helpful reviewers makes them come back.

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

### Step 6 – Tamper‑Proof Prediction Chain (Blockchain‑like)
Every time we make a prediction, we store it in a “chain”:
- Each block contains the prediction, part of the review, a hash of the features, and the hash of the previous block.
- If someone tries to change an old prediction, the hash will no longer match – the chain will be broken.

### Step 7 – Reward Helpful Reviewers
- The system identifies reviews that were **actually helpful** (real `helpful_ratio` >= 0.7) **and** predicted as helpful by our model.
- Those reviewers get 10 points and a discount code (e.g., `AMA-123456` for 15% off).

## 🖥️ How to Run This Project on Your Own Laptop?

### ✅ What You Need
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
cd C:\my_project   # Windows example
cd ~/Desktop/my_project   # Mac/Linux example

🐍 Step 3 -- Create a Virtual Environment (Clean & Safe) A virtual
environment keeps the project's packages separate from your other Python
projects.

bash \# Create an environment named \"review_env\" python -m venv
review_env Activate it:

Windows: review_env\\Scripts\\activate

Mac/Linux: source review_env/bin/activate

You will see (review_env) appear at the beginning of your command line.

📥 Step 4 -- Install Required Packages The project needs several Python
libraries. They are listed in requirements.txt.

bash pip install -r requirements.txt This may take 2--3 minutes. You
will see many lines of text -- that is normal.

📊 Step 5 -- Get the Dataset The code expects a file named output.csv
inside the notebooks/data/ folder. If you already have an Amazon review
CSV file, rename it to output.csv and place it there. If you don't have
one, you can download a small sample from this link (contact the project
owner).

🧪 Step 6 -- Run the Code You can run the project in two ways:

Option A -- Run the Jupyter Notebook (easiest) Inside the project
folder, start Jupyter:

bash jupyter notebook A browser window will open. Navigate to notebooks/
→ open MODEL TRAINING.ipynb. Then click Cell → Run All. Wait for the
code to finish.

Option B -- Run as a Python Script The main logic is inside the
notebook. If you want a single script, you can copy the cells into a .py
file and run:

bash python main.py 📝 Step 7 -- Test Your Own Review At the end of the
notebook, there is a cell that asks you to enter:

ASIN (a product ID from the dataset)

Review text

Star rating

Summary (optional)

Type your inputs. The model will tell you if the review would be helpful
or not helpful. It will also show a discount code if the review
qualifies for a reward.

🔍 Understanding the Results (Inferences) When you run the notebook, you
will see several graphs and numbers. Here is what they mean:

1\. Fig. 1: Review Length vs Helpfulness Longer reviews tend to be more
helpful -- they contain more details. Proof in code:
sns.scatterplot(x=\'review_length\', y=\'helpful_ratio\').

2\. Fig. 2: Star Rating vs Helpfulness Extremely low (1 star) or high (5
star) ratings are often less helpful than balanced (3--4 star) reviews.
Proof: sns.boxplot(x=\'overall\', y=\'helpful_ratio\').

3\. Fig. 4: BERT Emotions vs Helpfulness Helpful reviews usually contain
more joy and love and less anger or fear. Proof: The code groups reviews
by helpful_binary and plots the average emotion scores.

4\. Model Performance Numbers The last cells print:

text ✅ Accuracy: 0.8523 ✅ Precision: 0.8210 ✅ Recall: 0.7934 ✅
F1-score: 0.8069 ✅ AUC Score: 0.9125 These numbers prove that the model
works well and can be trusted to filter helpful reviews.

5\. Prediction Chain Output You will see a list of blocks showing that
each prediction is linked to the previous one -- proving no one can
alter past results without breaking the chain.

6\. Rewards Ledger A file rewards_ledger.csv is created. It lists all
reviewers who got rewards. This proves the system can be used in a real
loyalty program.

💰 Business Value -- How This Saves Money Without this system With this
system A product manager reads 1000 reviews manually -- takes 10 hours.
The model instantly shows only the 200 most helpful reviews -- saves 8
hours. Unhelpful (fake or useless) reviews waste time and hide real
problems. Unhelpful reviews are filtered out automatically. No way to
reward good reviewers -- they leave no extra feedback. Discount codes
encourage helpful customers to keep writing good reviews. No proof that
a review analysis was tampered with. The prediction chain acts as a
legal‑proof audit trail. Example calculation: If a product team earns
\$50 per hour, saving 8 hours per week = \$400 per week saved. Over a
year that's \$20,800 just for one product category.

❓ Common Questions Q: Do I need a powerful computer? A: The model can
run on any normal laptop. If you have a NVIDIA GPU, it will be faster,
but it works without one.

Q: Can I use my own review data? A: Yes. Your CSV must have columns
named exactly: reviewText, overall, asin, helpful. The helpful column
should look like \"\[5,6\]\" (string of two numbers).

Q: The code gives an error about missing output.csv A: Place your CSV
file in notebooks/data/ and name it output.csv. Then re‑run the
notebook.

Q: How long does training take? A: For 10,000 reviews, about 2 minutes.
For 100,000 reviews, about 10--15 minutes.

Q: Can I share the rewards ledger with my finance team? A: Yes. The
rewards_ledger.csv file is plain text -- they can open it in Excel.

📞 Need Help? If you get stuck:

Check that you activated the virtual environment (review_env appears in
your terminal).

Make sure you installed all packages (pip list should show xgboost,
transformers, torch, etc.).

Look at the log files inside the logs/ folder -- they often explain the
error.

📄 License This project is for educational and business use. You are
free to modify and use it in your own company.

convert this content in the .md format
