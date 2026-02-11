# Admissions Classifier

Binary classifier (logistic regression) that predicts Admit/Reject for **undergraduate** college applications using the [Kaggle student admission dataset](https://www.kaggle.com/datasets/amanace/student-admission-dataset). Features: GPA, SAT score, extracurricular activities.

## Project Requirements (Instructor)

**Business scenario**: University/College Admissions — a pipeline from beginning to end.

- **Beginning**: A student comes to the university/college for an admission enquiry  
- **End product**: Admitted

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get training data from Kaggle

Create a [Kaggle account](https://www.kaggle.com) and set up API credentials:

1. Go to Kaggle → Account → Create New API Token (downloads `kaggle.json`)
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)
3. Ensure the file is not publicly readable: `chmod 600 ~/.kaggle/kaggle.json`

Download the dataset:

```bash
kaggle datasets download -d amanace/student-admission-dataset
unzip student-admission-dataset.zip
```

This produces `student_admission_dataset.csv` with columns: `GPA`, `SAT_Score`, `Extracurricular_Activities`, `Admission_Status` (Accepted / Rejected / Waitlisted).

## 2. Train

```bash
python admissions.py --train student_admission_dataset.csv
```

Or with default path (uses `student_admission_dataset.csv` if in current directory):

```bash
python admissions.py --train
```

Produces `model.joblib` and `scaler.joblib`.

## 3. Run (review applications)

```bash
python admissions.py
```

Prompts for GPA, SAT score, and extracurricular activities; prints ACCEPTED or REJECTED; asks if you want to review another application.

---

**Notebook**: `admissions_classifier.ipynb` — interactive exploration with neural net, logistic regression, and random forest. Runs in Colab or locally (TensorFlow required).
