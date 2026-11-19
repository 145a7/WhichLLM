"""
特征/分类器消融实验（不作为最终提交脚本，用于挑选 baseline 配置）。

所有实验都用 StratifiedGroupKFold 按 prompt_id 分组，保证可比。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline import StyleFeatures, ROOT

SEED = 42
FOLDS = 5


def vectorize(train_texts, test_texts, cfg):
    """按配置生成特征矩阵。vocab 只在训练折上拟合。"""
    mats_tr, mats_te = [], []
    if cfg.get("word"):
        w = TfidfVectorizer(
            lowercase=True,
            ngram_range=cfg.get("word_ngram", (1, 2)),
            min_df=cfg.get("word_min_df", 2),
            max_features=cfg.get("word_max_features", 200_000),
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w[\w'-]+\b",
            stop_words=cfg.get("word_stop_words"),
        )
        mats_tr.append(w.fit_transform(train_texts))
        mats_te.append(w.transform(test_texts))
    if cfg.get("char"):
        c = TfidfVectorizer(
            analyzer=cfg.get("char_analyzer", "char_wb"),
            ngram_range=cfg.get("char_ngram", (2, 5)),
            min_df=cfg.get("char_min_df", 3),
            max_features=cfg.get("char_max_features", 300_000),
            sublinear_tf=True,
            lowercase=False,
        )
        mats_tr.append(c.fit_transform(train_texts))
        mats_te.append(c.transform(test_texts))
    if cfg.get("style"):
        s = StyleFeatures()
        st, sv = s.fit_transform(train_texts), s.transform(test_texts)
        scaler = StandardScaler().fit(st)
        mats_tr.append(csr_matrix(scaler.transform(st)))
        mats_te.append(csr_matrix(scaler.transform(sv)))
    if len(mats_tr) == 1:
        return mats_tr[0], mats_te[0]
    return hstack(mats_tr, format="csr"), hstack(mats_te, format="csr")


def eval_cfg(train, cfg, clf_factory):
    y = train["model"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.empty(len(train), dtype=object)
    fold_scores = []
    for tr_idx, va_idx in sgkf.split(train["text"], y, groups=train["prompt_id"]):
        t0 = time.time()
        Xtr, Xva = vectorize(
            train["text"].iloc[tr_idx].tolist(), train["text"].iloc[va_idx].tolist(), cfg
        )
        clf = clf_factory()
        clf.fit(Xtr, y[tr_idx])
        pred = clf.predict(Xva)
        oof[va_idx] = pred
        fold_scores.append(balanced_accuracy_score(y[va_idx], pred))
        print(f"    fold {len(fold_scores)}: {fold_scores[-1]:.4f} ({time.time()-t0:.0f}s)")
    score = balanced_accuracy_score(y, oof)
    print(f"  -> OOF {score:.4f} | folds {np.mean(fold_scores):.4f}±{np.std(fold_scores):.4f}")
    return score, oof


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")

    configs = {
        "baseline(word+char+style)": dict(word=True, char=True, style=True),
        "word+char": dict(word=True, char=True, style=False),
        "word only": dict(word=True, style=False),
        "char only": dict(char=True, style=False),
        "style only": dict(style=True, word=False, char=False),
        "word1-3+char2-5+style": dict(word=True, char=True, style=True, word_ngram=(1, 3)),
        "word_stop_only+style": dict(
            word=True, word_stop_words="english", word_ngram=(1, 3), char=False, style=True
        ),
        "char(not wb)+style": dict(
            word=False, char=True, char_analyzer="char", style=True
        ),
    }

    print("\n### 特征消融（LinearSVC C=1）")
    for name, cfg in configs.items():
        print(f"\n[{name}]")
        eval_cfg(train, cfg, lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED))


if __name__ == "__main__":
    main()
