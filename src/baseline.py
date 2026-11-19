"""
Name That LLM - 可运行基线方案
================================

任务：给定一段 40~250 词的英文段落，判断它由 18 个大模型中的哪一个生成。
数据：train.csv 2,699 条（150 个 prompt × 18 个模型），test.csv 的 prompt 与训练集完全不同。
评测：Balanced Accuracy（18 类召回率的宏平均），因此每个类同样重要，必须处理类别平衡。

思路（基线）：
1. 文本特征三路拼接：
   - 词级 TF-IDF（1~2 元语法）：捕捉用词偏好、常用搭配；
   - 字符级 TF-IDF（char_wb 2~5 元语法）：捕捉标点、大小写、缩写、排版等"指纹"；
   - 人工风格特征（句长、标点、引号类型、词汇丰富度、markdown 残留等）。
2. 分类器用 LinearSVC + class_weight='balanced'：高维稀疏文本上线性模型又快又稳。
3. 验证用 StratifiedGroupKFold，按 prompt_id 分组：同一 prompt 的 18 个回答不会同时出现在
   训练和验证两侧，模拟"测试集 prompt 完全没见过"的真实场景，避免高估分数。
4. 全量训练后预测 test.csv，按 sample_submission.parquet 的 id 顺序写 submission。

运行：
    python src/baseline.py
    python src/baseline.py --folds 5 --no-cv     # 跳过交叉验证，直接训练并出提交

依赖：numpy pandas scipy scikit-learn pyarrow joblib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5

# 数据/输出目录：优先 Kaggle 路径，其次本地 data/
ROOT = Path(__file__).resolve().parents[1]

# 标点/排版计数用的正则。保持原样，不做"清洗"，因为这些正是模型指纹。
STYLE_PATTERNS = {
    "comma": re.compile(r","),
    "semicolon": re.compile(r";"),
    "colon": re.compile(r":"),
    "emdash": re.compile(r"—"),
    "endash": re.compile(r"–"),
    "hyphen": re.compile(r"-"),
    "paren": re.compile(r"[()]"),
    "quote_double": re.compile(r"[\"“”]"),
    "apostrophe_curly": re.compile(r"’"),
    "apostrophe_straight": re.compile(r"'"),
    "exclaim": re.compile(r"!"),
    "question": re.compile(r"\?"),
    "ellipsis": re.compile(r"…|\.\.\."),
    "bullet": re.compile(r"(?m)^\s*[-*•]\s+"),
    "markdown_bold": re.compile(r"\*\*|__"),
    "markdown_heading": re.compile(r"(?m)^\s*#+\s"),
    "newline": re.compile(r"\n"),
    "digit": re.compile(r"\d"),
    "uppercase": re.compile(r"[A-Z]"),
}


def _sentences(text: str) -> list[str]:
    """极简分句：按 . ! ? 切分，过滤空串。"""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """人工风格特征。

    sklearn 的 Transformer 需要 fit/transform 接口，这里 fit 什么都不做，
    返回稠密矩阵，由 FeatureUnion 与 TF-IDF 稀疏矩阵拼接。
    """

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X, y=None):  # noqa: N803
        # 返回稠密矩阵：风格特征维度低（约 30 维），且 StandardScaler 需要中心化
        return np.asarray([self._one(t) for t in X], dtype=np.float32)

    @staticmethod
    def _one(text: str) -> list[float]:
        text = str(text)
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        n_words = max(len(words), 1)
        sents = _sentences(text)
        sent_lens = [len(re.findall(r"\S+", s)) for s in sents] or [0]
        uniq = set(w.lower() for w in words)
        low = [w.lower() for w in words]
        stop_ratio = sum(w in ENGLISH_STOP_WORDS for w in low) / n_words

        feats: list[float] = [
            np.log1p(n_words),                       # 文本长度（对数）
            float(len(text)) / n_words,              # 平均词长
            float(np.mean(sent_lens)),               # 平均句长
            float(np.std(sent_lens)),                # 句长波动
            float(len(sents)),                       # 句子数
            len(uniq) / n_words,                     # 类符/形符比
            sum(1 for w in uniq if low.count(w) == 1) / n_words,  # 只出现一次的词比例
            stop_ratio,                              # 停用词比例
        ]
        # 句长分布：短句/中句/长句占比，反映节奏
        bins = [(0, 12), (13, 20), (21, 30), (31, 45), (46, 10**9)]
        for lo, hi in bins:
            feats.append(sum(lo <= s <= hi for s in sent_lens) / len(sent_lens))

        # 标点与排版：每 100 词出现次数
        for pat in STYLE_PATTERNS.values():
            feats.append(len(pat.findall(text)) * 100.0 / n_words)
        return feats


def build_pipeline() -> Pipeline:
    """特征拼接 + 线性 SVM。"""
    features = FeatureUnion(
        [
            # 词级 TF-IDF：保留常见词（stop_words=None），因为功能词正是模型偏好的信号
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w[\w'-]+\b",
                ),
            ),
            # 字符级 TF-IDF：对大小写、标点、空格习惯敏感
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=300_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            # 人工风格特征：先抽取，再标准化，使其与 TF-IDF 的量纲可比
            ("style", Pipeline([("style", StyleFeatures()), ("scale", StandardScaler())])),
        ],
        # 风格特征维度低，权重减半，避免压过词/字符特征
        transformer_weights={"word": 1.0, "char": 1.0, "style": 0.5},
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",  # 均衡处理，配合 balanced accuracy
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def load_data(data_dir: Path):
    """读取训练/测试/样例提交，并做最基本的完整性检查。"""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_path = data_dir / "sample_submission.parquet"
    sample = pd.read_parquet(sample_path) if sample_path.exists() else None

    for col in ["id", "prompt_id", "text", "model"]:
        assert col in train.columns, f"train.csv 缺少列 {col}"
    for col in ["id", "text"]:
        assert col in test.columns, f"test.csv 缺少列 {col}"
    assert train["text"].notna().all() and test["text"].notna().all()
    assert set(test["id"]).issubset(set(sample["id"])) if sample is not None else True

    print(f"train: {train.shape}, test: {test.shape}, "
          f"模型数: {train['model'].nunique()}, prompt 数: {train['prompt_id'].nunique()}")
    return train, test, sample


def run_group_cv(train: pd.DataFrame, n_splits: int) -> tuple[float, np.ndarray, list[str]]:
    """按 prompt 分组的交叉验证，返回 OOF 预测与分数。"""
    classes = sorted(train["model"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(class_to_idx).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(train), -1, dtype=int)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train["text"], y, groups=train["prompt_id"]), start=1
    ):
        t0 = time.time()
        pipe = build_pipeline()
        pipe.fit(train["text"].iloc[tr_idx], y[tr_idx])
        pred = pipe.predict(train["text"].iloc[va_idx])
        oof[va_idx] = pred
        score = balanced_accuracy_score(y[va_idx], pred)
        fold_scores.append(score)
        print(f"  fold {fold}: {score:.4f}  "
              f"(train={len(tr_idx)}, valid={len(va_idx)}, {time.time()-t0:.1f}s)")

    overall = balanced_accuracy_score(y, oof)
    print(f"  fold 平均: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  OOF 汇总 : {overall:.4f}")
    return overall, oof, classes


def report_details(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], out_dir: Path):
    """输出每类召回率、混淆矩阵，便于写报告与分析弱类。"""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    ranking = sorted(zip(classes, recall), key=lambda x: x[1])
    print("  各类召回率（从低到高）：")
    for name, r in ranking:
        print(f"    {name:24s} {r:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame({"model": classes, "recall": recall}).sort_values("recall").to_csv(
        out_dir / "per_class_recall.csv", index=False
    )
    return cm


def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    """全量训练并预测测试集。"""
    classes = sorted(train["model"].unique())
    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(train["text"], train["model"])
    print(f"全量训练完成，用时 {time.time()-t0:.1f}s")
    pred = pipe.predict(test["text"])
    return pipe, pred, classes


def write_submission(test: pd.DataFrame, pred: np.ndarray, sample: pd.DataFrame | None, out_dir: Path):
    """按样例提交的 id 顺序写 parquet（Kaggle Code Competition 要求 parquet）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if sample is not None:
        sub = sample[["id"]].copy()
        mapping = dict(zip(test["id"], pred))
        missing = [i for i in sub["id"] if i not in mapping]
        assert not missing, f"有 {len(missing)} 个 id 未预测"
        sub["model"] = sub["id"].map(mapping)
    else:
        sub = pd.DataFrame({"id": test["id"], "model": pred})

    sub.to_parquet(out_dir / "submission.parquet", index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)
    # Kaggle Code Competition 只收集 /kaggle/working/submission.parquet，
    # 因此在 Kaggle 环境额外写一份到标准位置。
    if Path("/kaggle/working").exists():
        sub.to_parquet("/kaggle/working/submission.parquet", index=False)
        print("已额外写出 /kaggle/working/submission.parquet")
    print(f"已写出 {out_dir/'submission.parquet'}（{len(sub)} 行）")
    print(sub.head().to_string(index=False))
    return sub


def resolve_data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for cand in [Path("/kaggle/input/name-that-llm"), ROOT / "data", Path.cwd() / "data"]:
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError("找不到 train.csv，请用 --data-dir 指定数据目录")


def main():
    parser = argparse.ArgumentParser(description="Name That LLM baseline")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--no-cv", action="store_true", help="跳过交叉验证")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_dir = Path(args.output_dir)
    print(f"数据目录: {data_dir}")

    train, test, sample = load_data(data_dir)
    classes = sorted(train["model"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(class_to_idx).to_numpy()

    if not args.no_cv:
        print(f"\n=== {args.folds} 折按 prompt 分组交叉验证 ===")
        score, oof, _ = run_group_cv(train, args.folds)
        report_details(y, oof, classes, out_dir)
        # 记录 CV 结果，方便在报告里引用
        with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
            json.dump({"balanced_accuracy": score, "folds": args.folds,
                       "seed": SEED, "classes": classes}, f, ensure_ascii=False, indent=2)

    print("\n=== 全量训练 + 预测 ===")
    pipe, pred, classes_fit = fit_predict(train, test)
    assert classes_fit == classes, "训练类别与预期不一致"
    # 打印预测分布：可检查是否出现某个模型被大量预测的异常
    uniq, counts = np.unique(pred, return_counts=True)
    print("预测分布:", dict(zip(uniq, counts)))
    write_submission(test, pred, sample, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
