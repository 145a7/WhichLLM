# Name That LLM —— 可运行基线方案

Kaggle 竞赛主页：<https://www.kaggle.com/competitions/name-that-llm/overview>

## 1. 赛题理解

- **任务**：给定一段 40~250 词的英文段落，判断它由 18 个前沿大模型中的哪一个生成，属于 18 类文本归属（authorship attribution）问题。
- **数据**：`train.csv` 共 2,699 条，来自 150 个 prompt × 18 个模型（多数组合 18 条，`claude-fable-5` 少 1 条）。
- **关键难点**：测试集的 prompt 与训练集**完全不同**，模型不能靠记忆"某个 prompt 的固定回答"，只能学习跨主题的写作风格指纹。
- **评测指标**：Balanced Accuracy（18 类召回率的宏平均，`sklearn.metrics.balanced_accuracy_score`）。每类权重相同，因此不能只优化总体准确率。
- **提交**：Code Competition，隐藏测试集约 4,000 条；本地 `test.csv` 只有 3 条样例，用于保证代码可跑通。提交文件为 `submission.parquet`，列 `id, model`。
- **18 个模型（6 个实验室）**：
  - OpenAI：`gpt-5.5`、`gpt-5.6-luna`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-6-astra`
  - Google：`gemini-3.1-pro-preview`、`gemini-3.6-flash`、`gemini-3.7-flash`、`gemini-3.8-flash`
  - Anthropic：`claude-opus-4-6`、`claude-opus-4-8`、`claude-opus-5`、`claude-sonnet-5`、`claude-fable-5`
  - 阿里：`qwen3.8-max`
  - 智谱：`glm-5.3`、`glm-5.3-flash`
  - 月之暗面：`kimi-k3`

## 2. 方法

### 2.1 特征

三路特征拼接（`FeatureUnion`）：

| 特征 | 配置 | 作用 |
|---|---|---|
| 词级 TF-IDF | 1~2 元语法，`min_df=2`，`sublinear_tf` | 用词偏好、常见搭配、话语标记 |
| 字符级 TF-IDF | `char_wb` 2~5 元语法，`min_df=3` | 标点、大小写、缩写、空格与排版习惯 |
| 人工风格特征 | 约 30 维，标准化 | 句长均值/方差、句长分布、类符形符比、停用词比例、逗号/分号/破折号/引号/省略号频率、Markdown 残留等 |

风格特征权重设为 0.5，避免低维稠密特征压过高维稀疏特征。

### 2.2 分类器

`LinearSVC(C=1.0, class_weight="balanced")`：

- 文本 TF-IDF 维度高、样本量小（2,699），线性 SVM 训练快、泛化稳；
- `class_weight="balanced"` 与 Balanced Accuracy 指标一致，防止多数类主导；
- 不需要概率输出，直接取 `decision_function` 的最大值作为预测。

### 2.3 验证策略

使用 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`，**以 `prompt_id` 为分组**：

- 同一 prompt 的 18 个回答只会出现在训练侧或验证侧之一；
- 验证集 prompt 对模型完全不可见，与真实测试场景一致；
- 若用普通 KFold，同一 prompt 的其他模型回答会泄漏主题信息，分数会被严重高估（同一 prompt 的 18 段文字共享大量内容词），不能作为模型选择依据。

### 2.4 流程

```
读取数据 -> 分组交叉验证评估 -> 输出各类召回率/混淆矩阵
        -> 全量训练 -> 预测 test.csv -> 按 sample_submission 顺序写 parquet
```

## 3. 运行

环境：`D:\miniconda3\envs\Mechine-Learing-Lessons`（Python 3.12，scikit-learn 1.9.1，pandas 3.1，pyarrow 26）。

```powershell
# 完整流程（5 折 CV + 全量训练 + 生成提交）
D:\miniconda3\envs\Mechine-Learing-Lessons\python.exe src\baseline.py

# 跳过 CV，直接训练并出提交
D:\miniconda3\envs\Mechine-Learing-Lessons\python.exe src\baseline.py --no-cv

# 指定数据目录 / 输出目录
D:\miniconda3\envs\Mechine-Learing-Lessons\python.exe src\baseline.py --data-dir data --output-dir outputs
```

Kaggle Notebook 中会自动识别 `/kaggle/input/name-that-llm`，无需改代码。

### 在 Kaggle Notebook 中提交

1. 新建 Notebook，挂载竞赛数据（Input → Add Data → Competitions → Name That LLM）。
2. 把 `src/baseline.py` 的内容粘贴到一个代码单元格（或把文件上传到 `/kaggle/working`）。
3. 运行后，脚本除写 `outputs/submission.parquet` 外，还会额外写 `/kaggle/working/submission.parquet`（Code Competition 的收集位置）。
4. 点击 Submit，选择该 Notebook 版本；隐藏测试集约 4,000 条，prompt 与训练集不同。

## 4. 当前结果

5 折按 prompt 分组交叉验证：

| 指标 | 数值 |
|---|---|
| OOF Balanced Accuracy | **0.3586** |
| 各折分数 | 0.3445 / 0.3519 / 0.3870 / 0.3278 / 0.3815 |
| 折均值 ± 标准差 | 0.3585 ± 0.0225 |

各类召回率差异较大：`gemini-3.1-pro-preview` 0.767、`gpt-6-astra` 0.633、`claude-opus-5` 0.620；
而 `gpt-5.6-sol`、`claude-fable-5` 只有 0.087，`glm-5.3-flash` 0.167。
低分模型主要是同实验室的"兄弟模型"（如 GPT-5.6 的 luna/sol/terra），风格高度相似，是后续优化的重点。

### 特征消融（同一 5 折分组验证）

| 特征组合 | OOF Balanced Accuracy |
|---|---|
| 词 TF-IDF + 字符 TF-IDF + 风格（本方案） | **0.3586** |
| 词 TF-IDF + 字符 TF-IDF | 0.3030 |
| 字符 TF-IDF only | 0.2934 |
| 词 TF-IDF only | 0.2697 |
| 风格特征 only | 0.2700 |
| 扩展版 200+ 维风格特征（过拟合，未采用） | 0.3119 |

结论：三类特征互补；风格特征单独用只有 0.27，但去掉后整体掉 0.056，是跨 prompt 泛化的关键。
盲目堆更多风格特征反而过拟合。

### 分类器对比

| 分类器 | OOF Balanced Accuracy |
|---|---|
| LinearSVC C=1（本方案） | 0.3586 |
| LinearSVC C=4 | 0.3623 |
| LogisticRegression | 0.3237 |
| ComplementNB | 0.2371 |
| LinearSVC + ComplementNB 概率平均 | 0.3648 |

调参带来的提升在 ±0.006 以内，接近折间波动，因此基线保留最简单、稳定的 `C=1`。

输出文件：

- `outputs/submission.parquet` / `outputs/submission.csv`：提交文件；
- `outputs/cv_summary.json`：CV 汇总；
- `outputs/per_class_recall.csv`：各类召回率；
- `outputs/confusion_matrix.csv`：18×18 混淆矩阵。

## 5. 后续改进方向

1. **同实验室层级分类**：先分 6 个实验室，再在实验室内分模型，缓解 luna/sol/terra 这类近亲混淆。
2. **模型集成**：LinearSVC + ComplementNB 概率平均（实测 0.3648），或加入不同特征子空间的 SVC。
3. **prompt 不变特征筛选**：只保留在多个 prompt 上都稳定指向同一模型的 n-gram，去掉主题词。
4. **语义表征**：用 DeBERTa / sentence-transformers 提取句向量，再接线性分类头（需要 GPU 或下载预训练权重）。
5. **测试时按 prompt 聚类归一化**：隐藏测试集中同一 prompt 可能有多条回答，可先聚类再减去组内均值，削弱主题信号。

## 6. 文件结构

```
WhichLLM/
├── data/                      # 解压后的 train.csv / test.csv / sample_submission.parquet
├── src/
│   ├── baseline.py            # 主方案：特征 + LinearSVC + 分组 CV + 提交
│   ├── style_features.py      # 扩展版风格特征（实验用，最终未采用）
│   ├── experiment.py          # 特征消融
│   ├── experiment2.py         # 扩展风格特征 / 功能词块实验
│   └── experiment3.py         # 分类器调参与集成实验
├── outputs/                   # 运行产物
├── requirements.txt
└── README.md
```

> 注：本仓库的 `test.csv` 只有 3 条公开样例，因此本地生成的 `submission.parquet` 只有 3 行；提交到 Kaggle 后，同一份代码会在隐藏测试集（约 4,000 条）上重新运行。
