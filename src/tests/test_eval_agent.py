import ctypes
import pathlib

_orig_cdll = ctypes.CDLL


def patched_cdll(path, *args, **kwargs):
    if isinstance(path, pathlib.Path):
        path = str(path)  # Convert Path object to string
    return _orig_cdll(path, *args, **kwargs)


ctypes.CDLL = patched_cdll

import os

os.environ["TABREPO_DISABLE_CPP_AUC"] = "1"

from agents.eval_agent import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from utils import load_arff_to_dataframe

table = load_arff_to_dataframe("data/houses.arff")
evaluate(table)

X = table.drop(columns=["binaryClass"])
y = table["binaryClass"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=400, random_state=0)
rf.fit(X_tr, y_tr)
auc_holdout = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
print("Hold‑out AUC:", auc_holdout)
