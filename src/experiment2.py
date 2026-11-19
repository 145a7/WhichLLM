"""
第二轮实验：扩展风格特征 + 功能词块 + 分类器超参。

做法：每个 fold 只拟合一次 word/char/stop 的 TF-IDF，缓存矩阵，
然后对不同组合快速训练 LinearSVC，避免重复向量化。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline import StyleFeatures  # noqa: E402
from style_features import RichStyleFeatures  # noqa: E402

SEED = 42
FOLDS = 5


def dense_style(extractor, train_texts, valid_texts):
    s = extractor()
    a = s.fit_transform(train_texts)
    b = s.transform(valid_texts)
    scaler = StandardScaler().fit(a)
    return csr_matrix(scaler.transform(a)), csr_matrix(scaler.transform(b))


def build_fold_cache(train, tr_idx, va_idx, use_stop=False):
    tr_text = train["text"].iloc[tr_idx].tolist()
    va_text = train["text"].iloc[va_idx].tolist()
    cache = {}

    word = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=200_000,
        sublinear_tf=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w'-]+\b",
    )
    cache["word"] = (word.fit_transform(tr_text), word.transform(va_text))

    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000,
        sublinear_tf=True, lowercase=False,
    )
    cache["char"] = (char.fit_transform(tr_text), char.transform(va_text))

    cache["style_old"] = dense_style(StyleFeatures, tr_text, va_text)
    cache["style_rich"] = dense_style(RichStyleFeatures, tr_text, va_text)

    if use_stop:
        stop = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 3), min_df=2, sublinear_tf=True,
            stop_words="english", token_pattern=r"(?u)\b\w[\w'-]+\b",
        )
        cache["stop"] = (stop.fit_transform(tr_text), stop.transform(va_text))
    return cache


def stack(cache, spec):
    """spec: [(name, weight), ...]，name 为 'const' 表示常数块（占位）。"""
    tr, va = [], []
    for name, weight in spec:
        a, b = cache[name]
        tr.append(a * weight)
        va.append(b * weight)
    return hstack(tr, format="csr"), hstack(va, format="csr")


def evaluate(train, specs, clf_c=1.0, use_stop=False):
    y = train["model"].to_numpy()
    classes = sorted(set(y))
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof = {name: np.empty(len(train), dtype=int) for name in specs}
    times = {name: 0.0 for name in specs}
    for k, (tr_idx, va_idx) in enumerate(sgkf.split(train["text"], yi, groups=train["prompt_id"]), 1):
        cache = build_fold_cache(train, tr_idx, va_idx, use_stop=use_stop)
        for name, spec in specs.items():
            t0 = time.time()
            Xtr, Xva = stack(cache, spec)
            clf = LinearSVC(C=clf_c, class_weight="balanced", max_iter=5000, random_state=SEED)
            clf.fit(Xtr, yi[tr_idx])
            oof[name][va_idx] = clf.predict(Xva)
            times[name] += time.time() - t0
        print(f"  fold {k} done", flush=True)

    results = {}
    for name in specs:
        s = balanced_accuracy_score(yi, oof[name])
        results[name] = s
        print(f"{name:42s} {s:.4f}  (clf {times[name]:.0f}s)", flush=True)
    return results


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    specs = {
        "word+char+oldstyle(.5)": [("word", 1.0), ("char", 1.0), ("style_old", 0.5)],
        "word+char+richstyle(.5)": [("word", 1.0), ("char", 1.0), ("style_rich", 0.5)],
        "word+char+richstyle(1.0)": [("word", 1.0), ("char", 1.0), ("style_rich", 1.0)],
        "richstyle only": [("style_rich", 1.0)],
        "word+richstyle(0.5)": [("word", 1.0), ("style_rich", 0.5)],
        "word+char+richstyle(.5)+stop(.3)": [
            ("word", 1.0), ("char", 1.0), ("style_rich", 0.5), ("stop", 0.3)
        ],
    }
    print("=== 5 折，LinearSVC C=1 ===", flush=True)
    evaluate(train, specs, clf_c=1.0, use_stop=True)


if __name__ == "__main__":
    main()
