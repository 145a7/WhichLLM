"""
扩展版手工风格特征。

用途：LLM 归属任务中，主题词在测试 prompt 上会失效，但标点、句长、功能词、
话语标记等写法习惯可以跨主题迁移。这里把可解释的风格信号尽量抽全。
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 常用功能词 / 话语标记：按"每 100 词频率"统计
FUNCTION_WORDS = [
    # 冠词、连词、介词
    "the", "a", "an", "and", "but", "or", "nor", "if", "because", "although",
    "though", "while", "whereas", "unless", "since", "so", "yet", "for", "as",
    "than", "that", "which", "who", "whom", "whose", "whether", "what", "when",
    "where", "how", "why", "in", "on", "at", "by", "with", "without", "from",
    "to", "of", "into", "onto", "over", "under", "between", "among", "through",
    "during", "before", "after", "above", "below", "about", "against", "around",
    # 代词、限定词
    "this", "these", "those", "it", "its", "they", "their", "them", "we", "our",
    "us", "you", "your", "i", "my", "me", "he", "she", "his", "her", "one",
    "some", "any", "all", "each", "every", "both", "few", "many", "much",
    "more", "most", "other", "another", "such", "same", "own",
    # 助动词、情态动词
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "can", "could", "may", "might", "must", "should",
    "would", "will", "shall", "ought",
    # 否定、程度、话语标记
    "not", "no", "never", "very", "too", "also", "just", "only", "even",
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "consequently", "nevertheless", "nonetheless", "instead", "rather",
    "perhaps", "maybe", "likely", "arguably", "notably", "importantly",
    "ultimately", "overall", "indeed", "still", "already", "often", "sometimes",
    "usually", "generally", "typically", "particularly", "especially",
]

HEDGES = ["perhaps", "maybe", "likely", "arguably", "might", "could", "seems",
          "appears", "possibly", "probably", "somewhat", "relatively"]
BOOSTERS = ["very", "extremely", "highly", "significantly", "clearly", "certainly",
            "absolutely", "crucial", "essential", "remarkable", "profound", "deeply"]
CONNECTIVES = ["however", "moreover", "furthermore", "therefore", "thus", "hence",
               "consequently", "nevertheless", "nonetheless", "meanwhile",
               "additionally", "similarly", "conversely", "instead", "ultimately",
               "overall", "in conclusion", "in short", "for example", "for instance"]
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "she", "it", "they", "him", "her", "them", "his", "hers",
                "its", "their", "theirs"}
MODALS = {"can", "could", "may", "might", "must", "should", "would", "will",
          "shall", "ought"}

# 正则表：统计标点与排版习惯
PUNCT = {
    "comma": r",",
    "semicolon": r";",
    "colon": r":",
    "emdash": r"—",
    "endash": r"–",
    "hyphen": r"(?<!-)-(?!-)",
    "paren": r"[()]",
    "quote_double_curly": r"[“”]",
    "quote_double_straight": r'"',
    "apostrophe_curly": r"’",
    "apostrophe_straight": r"'",
    "exclaim": r"!",
    "question": r"\?",
    "ellipsis": r"…|\.\.\.",
    "slash": r"/",
    "ampersand": r"&",
    "percent": r"%",
    "at": r"@",
    "dollar": r"\$",
    "asterisk": r"\*",
    "underscore": r"_",
    "plus": r"\+",
    "equals": r"=",
    "hash": r"#",
    "pipe": r"\|",
}

STARTERS = ["the", "this", "in", "it", "however", "but", "and", "while", "when",
            "if", "ultimately", "moreover", "a", "an", "as", "by", "for", "to",
            "one", "these", "such", "although", "because", "imagine", "consider"]

_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _sents(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _syllables(word: str) -> int:
    """粗糙音节数：元音组个数，去掉词尾不发音 e。"""
    w = word.lower()
    n = len(_VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def extract(text: str) -> list[float]:
    text = str(text).replace("\r\n", "\n")
    words = _WORD.findall(text)
    low = [w.lower() for w in words]
    n_words = max(len(words), 1)
    sents = _sents(text)
    sent_lens = [len(_WORD.findall(s)) for s in sents] or [0]
    counts = Counter(low)
    uniq = set(low)

    f: list[float] = []

    # --- 长度与节奏 ---
    f += [
        np.log1p(n_words),
        np.log1p(len(text)),
        float(np.mean(sent_lens)),
        float(np.std(sent_lens)),
        float(np.mean(sent_lens)) / (np.std(sent_lens) + 1e-6),  # 句长变异系数
        float(max(sent_lens)),
        float(min(sent_lens)),
        float(len(sents)),
    ]

    # --- 词汇丰富度与词形 ---
    word_lens = [len(w) for w in words] or [0]
    syllables = sum(_syllables(w) for w in words)
    f += [
        len(uniq) / n_words,                                   # 类符/形符比
        sum(1 for v in counts.values() if v == 1) / n_words,    # hapax 比例
        float(np.mean(word_lens)),
        float(np.std(word_lens)),
        sum(1 for w in words if len(w) >= 8) / n_words,        # 长词比例
        sum(1 for w in words if len(w) <= 3) / n_words,        # 短词比例
        syllables / n_words,                                    # 平均音节
        206.835 - 1.015 * (n_words / max(len(sents), 1)) - 84.6 * (syllables / n_words),
        sum(1 for w in words if w.isupper() and len(w) > 1) / n_words,  # 全大写词
        sum(1 for w in words if w[0].isupper()) / n_words,              # 首字母大写
    ]

    # --- 句长分布 ---
    for lo, hi in [(0, 8), (9, 15), (16, 22), (23, 30), (31, 45), (46, 10**9)]:
        f.append(sum(lo <= s <= hi for s in sent_lens) / len(sent_lens))

    # --- 句首/句尾习惯 ---
    starts = [s.split()[0].lower().strip("\"'“”") for s in sents if s.split()]
    for sw in STARTERS:
        f.append(sum(x == sw for x in starts) / max(len(starts), 1))
    f += [
        sum(s.rstrip().endswith(".") for s in sents) / max(len(sents), 1),
        sum(s.rstrip().endswith(("!", "?")) for s in sents) / max(len(sents), 1),
        sum(s.lstrip().startswith(('"', "“", "'")) for s in sents) / max(len(sents), 1),
    ]

    # --- 功能词、话语标记 ---
    for w in FUNCTION_WORDS:
        f.append(counts[w] * 100.0 / n_words)
    f += [
        sum(counts[w] for w in HEDGES) * 100.0 / n_words,
        sum(counts[w] for w in BOOSTERS) * 100.0 / n_words,
        sum(counts[w] for w in CONNECTIVES) * 100.0 / n_words,
        sum(counts[w] for w in FIRST_PERSON) * 100.0 / n_words,
        sum(counts[w] for w in SECOND_PERSON) * 100.0 / n_words,
        sum(counts[w] for w in THIRD_PERSON) * 100.0 / n_words,
        sum(counts[w] for w in MODALS) * 100.0 / n_words,
    ]

    # --- 标点/排版（每 100 词）---
    for pat in PUNCT.values():
        f.append(len(re.findall(pat, text)) * 100.0 / n_words)
    f += [
        len(re.findall(r"\s—\s", text)) * 100.0 / n_words,          # 带空格破折号
        len(re.findall(r"\w—\w", text)) * 100.0 / n_words,          # 不带空格破折号
        len(re.findall(r"\s[,.!?;:]", text)) * 100.0 / n_words,     # 标点前空格（异常排版）
        len(re.findall(r"  +", text)) * 100.0 / n_words,            # 连续空格
        len(_NON_ASCII.findall(text)) * 100.0 / max(len(text), 1),  # 非 ASCII 比例
        len(re.findall(r"(?m)^\s*[-*•]\s+", text)),                 # 项目符号数
        len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)),               # 有序列表数
        len(re.findall(r"\*\*|__", text)),                          # Markdown 加粗
        len(re.findall(r"(?m)^\s*#+\s", text)),                     # Markdown 标题
        len(re.findall(r"\n", text)),                               # 换行数
        len(re.findall(r"\n\s*\n", text)),                          # 段落数
    ]

    # --- 数词/符号 ---
    f += [
        sum(ch.isdigit() for ch in text) * 100.0 / max(len(text), 1),
        len(re.findall(r"\b\d[\d,.]*\b", text)) * 100.0 / n_words,
    ]
    return [float(x) for x in f]


class RichStyleFeatures(BaseEstimator, TransformerMixin):
    """sklearn 兼容包装：输出稠密矩阵，方便与 TF-IDF 拼接。"""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X, y=None):  # noqa: N803
        return np.asarray([extract(t) for t in X], dtype=np.float32)
