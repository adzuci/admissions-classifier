# Admissions Classifier

Binary classifier that predicts Admit/Reject for graduate school applications using the [Kaggle student admission dataset](https://www.kaggle.com/datasets/amanace/student-admission-dataset).

## Project Requirements (Instructor)

**Business scenario**: University/College Admissions — a pipeline from beginning to end.

- **Beginning**: A student comes to the university/college for an admission enquiry  
- **End product**: Admitted

## Setup

Uses the `ai-fundamentals` pyenv (Python 3.10.14). Install deps if needed:

```bash
pip install -r requirements.txt
```

## Train

With the dataset present:

```bash
python train.py
```

Produces `model.keras` and `scaler.joblib`. Or run `admissions_classifier.ipynb`.

**If TensorFlow hangs on Mac** (stuck at Epoch 1): use the NumPy fallback:

```bash
python train_numpy.py
```

Produces `model_numpy.joblib` and `scaler.joblib`. When running `admissions_review.py`, enter `model_numpy.joblib` when prompted for model path.

## Data

Download the dataset from Kaggle (requires account and API credentials in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d amanace/student-admission-dataset
unzip student-admission-dataset.zip
# Output: student_admission_dataset.csv
```

Columns: `GPA`, `SAT_Score`, `Extracurricular_Activities`, `Admission_Status` (Accepted/Rejected/Waitlisted).
