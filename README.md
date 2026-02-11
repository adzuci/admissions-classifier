# Admissions Classifier

Binary classifier that predicts Admit/Reject for graduate school applications using the [Kaggle student admission dataset](https://www.kaggle.com/datasets/amanace/student-admission-dataset).

## Setup

Requires Python 3.11 (TensorFlow does not support 3.14).

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the dataset from Kaggle (requires account and API credentials in `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d amanace/student-admission-dataset
unzip student-admission-dataset.zip
```
