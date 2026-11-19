"""
第三轮实验：固定 baseline 特征（word + char + 30 维风格），比较
  - LinearSVC 的 C；
  - LogisticRegression / ComplementNB；
  - 三者概率平均的小集成。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline import StyleFeatures  # noqa: E402

SEED = 42
FOLDS = 5


def make_fold(train, tr_idx, va_idx):
    tr_text = train["text"].iloc[tr_idx].tolist()
    va_text = train["text"].iloc[va_idx].tolist()
    word = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
        sublinear_tf=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w'-]+\b",
    )
    Xw = word.fit_transform(tr_text)
    Xwv = word.transform(va_text)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000,
        sublinear_tf=True, lowercase=False,
    )
    Xc = char.fit_transform(tr_text)
    Xcv = char.transform(va_text)
    st = StyleFeatures()
    a = st.fit_transform(tr_text)
    b = st.transform(va_text)
    sc = StandardScaler().fit(a)
    Xtr = hstack([Xw, Xc, csr_matrix(sc.transform(a)) * 0.5], format="csr")
    Xva = hstack([Xwv, Xcv, csr_matrix(sc.transform(b)) * 0.5], format="csr")
    # ComplementNB 要求非负特征，单独用 word+char
    Xnb = hstack([Xw, Xc], format="csr")
    Xnbv = hstack([Xwv, Xcv], format="csr")
    return Xtr, Xva, Xnb, Xnbv


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    classes = sorted(train["model"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[v] for v in train["model"]])
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    svc_cs = [0.25, 0.5, 1.0, 2.0, 4.0]
    oof_pred = {f"svc_{c}": np.full(len(train), -1) for c in svc_cs}
    oof_pred["lr"] = np.full(len(train), -1)
    oof_pred["nb"] = np.full(len(train), -1)
    proba = {k: np.zeros((len(train), len(classes))) for k in oof_pred}

    for k, (tr_idx, va_idx) in enumerate(sgkf.split(train["text"], y, groups=train["prompt_id"]), 1):
        t0 = time.time()
        Xtr, Xva, Xnb, Xnbv = make_fold(train, tr_idx, va_idx)
        for c in svc_cs:
            clf = LinearSVC(C=c, class_weight="balanced", max_iter=5000, random_state=SEED)
            clf.fit(Xtr, y[tr_idx])
            d = clf.decision_function(Xva)
            oof_pred[f"svc_{c}"][va_idx] = d.argmax(1)
            proba[f"svc_{c}"][va_idx] = softmax(d * 2.0, axis=1)

        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        lr.fit(Xtr, y[tr_idx])
        oof_pred["lr"][va_idx] = lr.predict(Xva)
        proba["lr"][va_idx] = lr.predict_proba(Xva)

        nb = ComplementNB(alpha=0.3)
        nb.fit(Xnb, y[tr_idx])
        oof_pred["nb"][va_idx] = nb.predict(Xnbv)
        proba["nb"][va_idx] = nb.predict_proba(Xnbv)
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)

    print("\n--- 单模型 OOF ---")
    for name, pred in oof_pred.items():
        print(f"{name:10s} {balanced_accuracy_score(y, pred):.4f}")

    print("\n--- 概率平均集成 ---")
    combos = {
        "svc1+lr": (["svc_1.0", "lr"], [1, 1]),
        "svc1+nb": (["svc_1.0", "nb"], [1, 1]),
        "svc1+lr+nb": (["svc_1.0", "lr", "nb"], [1, 1, 1]),
        "svc.5+lr+nb": (["svc_0.5", "lr", "nb"], [1, 1, 1]),
        "svc1+lr+nb(2:1:1)": (["svc_1.0", "lr", "nb"], [2, 1, 1]),
    }
    for name, (keys, weights) in combos.items():
        p = sum(w * proba[k] for k, w in zip(keys, weights)) / sum(weights)
        print(f"{name:18s} {balanced_accuracy_score(y, p.argmax(1)):.4f}")


if __name__ == "__main__":
    main()
