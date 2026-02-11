# Admissions Classifier

Binary classifier (logistic regression) that predicts Admit/Reject for college applications using the [Kaggle student admission dataset](https://www.kaggle.com/datasets/amanace/student-admission-dataset). Features: GPA, SAT score, extracurricular activities.

## Project Requirements

**Business scenario**: University/College Admissions would like an admissions 
pipeline from beginning to end.

- **Beginning**: A student comes to the university/college for an admission enquiry  
- **End product**: A script which invokes a model that 
decides if student is accepted 
or rejected 

---

## Option A: Colab notebook (no local setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adzuci/admissions-classifier/blob/main/admissions_classifier.ipynb)

Open the notebook in Google Colab. It fetches the dataset from GitHub and trains neural net, logistic regression, and random forest models. Run all cells to train, evaluate, and predict. Works on Colab, Windows, and Mac.

## Option B: Script (local)

### Setup

```bash
pip install -r requirements.txt
```

### 1. Get training data from Kaggle

Create a [Kaggle account](https://www.kaggle.com) and set up API credentials:

1. Go to Kaggle → Account → Create New API Token (downloads `kaggle.json`)
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)
3. Ensure the file is not publicly readable: `chmod 600 ~/.kaggle/kaggle.json`

Download the dataset:

```bash
kaggle datasets download -d amanace/student-admission-dataset
unzip student-admission-dataset.zip
```

Produces `student_admission_dataset.csv` (columns: `GPA`, `SAT_Score`, `Extracurricular_Activities`, `Admission_Status`).

### 2. Train

```bash
python admissions.py --train student_admission_dataset.csv
```

Or with default path: `python admissions.py --train`

Saves `model.joblib` and `scaler.joblib`.

### 3. Run (review applications)

```bash
python admissions.py
```

Prompts for GPA, SAT score, and extracurricular activities; prints ACCEPTED or REJECTED; asks if you want to review another application.
