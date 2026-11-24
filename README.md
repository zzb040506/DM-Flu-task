# Flu Shot Learning：H1N1 与季节性流感疫苗接种预测

这是一个参加 [DrivenData Flu Shot Learning](https://www.drivendata.org/competitions/66/flu-shot-learning/) 竞赛的完整方案。任务是根据受访者的人口学特征、健康情况、行为习惯和疫苗态度，预测他/她是否会接种 **H1N1 疫苗** 和 **季节性流感疫苗** 两种疫苗。

评价指标是两个目标 [AUROC](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF#%E6%9B%B2%E7%BA%BF%E4%B8%8B%E9%9D%A2%E7%A7%AF) 的平均值（越接近 1 越好）。

本仓库不仅包含最终代码，还包含训练好的模型、提交文件和完整的实验记录，**下载后可以直接验证结果，也可以从零复现整套流程**。

| 本地验证（5876 条保留样本） | 平均 AUROC |
|---|---:|
| 竞赛参考 notebook（LightGBM）复现 | 0.86841 |
| LightGBM + XGBoost + CatBoost 融合 | 0.86960 |
| 再加四状态联合模型（本仓库首选方案） | **0.87004** |

> 说明：以上是本地保留集上的分数，不等于官方排行榜分数。官方分数必须由实际提交得到，本仓库不声称已达到任何名次。

---

## 目录

- [快速开始](#快速开始)
- [数据从哪里来](#数据从哪里来)
- [你会在仓库里看到什么](#你会在仓库里看到什么)
- [方法思路](#方法思路)
- [结果与可信度](#结果与可信度)
- [完整复现](#完整复现)
- [常见问题](#常见问题)
- [设计取舍](#设计取舍)
- [声明](#声明)

---

## 快速开始

### 第 1 步：获取代码和环境

```bash
git clone https://github.com/zzb040506/DM-Flu-task.git
cd DM-Flu-task

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
```

开发环境为 Python 3.14（Anaconda）。Python 3.10 及以上一般都可以；各依赖的精确版本见 `requirements.lock.txt`。

### 第 2 步：下载数据，放进 `data/`

数据不在仓库里（见下一节说明），需要自己从官网下载：

**下载地址：<https://www.drivendata.org/competitions/66/flu-shot-learning/data/>**

注册并登录后，下载以下 4 个 CSV，全部放进本仓库的 `data/` 目录，文件名保持不变：

```
data/
├── training_set_features.csv    # 训练集特征（26708 行）
├── training_set_labels.csv      # 训练集标签（两个疫苗是否接种）
├── test_set_features.csv        # 测试集特征（26708 行）
└── submission_format.csv        # 提交格式模板
```

`data/` 目录已经在仓库中预留好（只有一个 `.gitkeep` 占位文件），放进去即可。

### 第 3 步：验证环境是否正常（不训练，几秒钟）

```bash
python verify.py
python predict.py
```

- `verify.py`：检查 6 份已生成的提交文件格式、行数、ID 顺序、概率范围，并输出每个文件的 SHA-256 和配对 bootstrap 结果。
- `predict.py`：从 `artifacts/` 里已保存的模型重新推理，断言与导出的 `submission_fullbag.csv` / `submission_joint_fullbag.csv` 完全一致（误差 < 1e-9），证明提交文件确实由这些模型产生。

### 第 4 步：拿到提交文件

仓库里已经有 6 份生成好的提交文件，**首选是 `submission_joint_robust.csv`**，可以直接上传到官网：

| 文件 | 含义 |
|---|---|
| `submission_joint_robust.csv` | **首选**：三模型 + 联合模型，5 折平均与全量多种子平均各占 50% |
| `submission_joint_foldbag.csv` | 联合方案的 5 折 bagging 版本 |
| `submission_joint_fullbag.csv` | 联合方案的全量数据多种子版本 |
| `submission_robust.csv` / `submission_foldbag.csv` / `submission_fullbag.csv` | 不含联合模型的对应版本 |

### 第 5 步（可选）：从头训练

```bash
python experiment.py --lgb-trials 14 --xgb-trials 8 --cat-trials 6 --folds 5
python baseline.py
python joint.py
python verify.py
python predict.py
python report.py
```

注意：`artifacts/` 里已经缓存了调参结果和折模型预测。直接重跑会**读取缓存**而不是重新训练。如果要真正从零复现，请先删除 `artifacts/`（保留目录），或者把代码复制到一个新目录再跑。

---

## 数据从哪里来

- 来源：[DrivenData Flu Shot Learning 数据页](https://www.drivendata.org/competitions/66/flu-shot-learning/data/)
- 需要免费注册账号后下载。
- 仓库不附带原始数据，原因有两点：竞赛数据的分发受其使用条款约束；同时避免仓库体积过大。
- 下载后 4 个文件的 SHA-256 应与 `artifacts/provenance.json` 中记录的一致，可用来确认数据版本。

---

## 你会在仓库里看到什么

```
.
├── README.md                     # 本文件
├── REPORT.md                     # 实验结果报告（由 report.py 生成）
├── requirements.lock.txt         # 精确依赖版本
├── data/                         # 放比赛数据（需自行下载，见上）
├── artifacts/                    # 模型、缓存、划分、冻结配方
│   ├── frozen_recipe.json        # 保留集评估前冻结的模型与融合配方
│   ├── joint_frozen_recipe.json  # 联合模型的选择结果与权重
│   ├── provenance.json           # 数据哈希、环境版本、随机种子
│   ├── split.npz                 # 开发集 / 保留集划分
│   ├── predictions.npz           # 保留集、OOF、测试集预测数组
│   ├── model_*.joblib            # 21 个全量训练模型
│   ├── tune_*.npz / final_*.npz  # 调参与折模型预测缓存
│   └── search.sqlite             # Optuna 调参数据库
├── experiment.py                 # 主流程：特征、调参、验证、训练、出提交
├── baseline.py                   # 参考 notebook 参数复现，作为对照基线
├── joint.py                      # 四状态联合目标模型
├── verify.py                     # 提交文件校验 + 配对 bootstrap
├── predict.py                    # 用已保存模型重新推理并校验一致性
├── report.py                     # 根据 JSON 结果生成 REPORT.md
├── metrics.json                  # 三模型方案的成绩
├── joint_metrics.json            # 联合方案的成绩与权重
├── baseline_metrics.json         # 基线成绩
├── verification.json             # 6 份提交文件的哈希、范围和 bootstrap 区间
├── inference_verification.json   # 重新推理与导出文件的最大误差
├── run.log / joint_*.log         # 逐轮、逐折训练日志
└── submission_*.csv              # 6 份可直接提交的预测文件
```

`artifacts/` 里的 `.joblib` 模型是 Python pickle 格式，**只应加载自己生成或来自可信来源的模型文件**。本仓库中的模型均由这些脚本训练得到。

---

## 方法思路

### 1. 先解决"怎么相信本地分数"

表格竞赛最容易犯的错误是反复用同一份验证集调参，导致本地分数虚高。本方案把训练数据分成两部分：

- **开发集（78%，20832 条）**：所有调参、融合权重选择都在这里做，内部再用 3 折交叉验证。
- **保留集（22%，5876 条）**：只用来做最后一次检查。在它被使用之前，所有配置都先写入 `artifacts/frozen_recipe.json` 冻结。

分层依据是 `2 * h1n1_vaccine + seasonal_vaccine`（四种标签组合），保证两边的类别比例一致。

因此报告里的保留集分数是"配方冻结后只评估一次"的结果，而不是反复挑出来的最好数字。

### 2. 特征工程

原始特征约 35 列，混合了数值、有序类别（年龄、教育、收入）和无序类别（种族、性别、地区、职业），缺失很多。

代码提供一套可开关的组合特征（`features(..., engineering=True)`）：

- 全行缺失数量，以及行为、态度、医嘱、就业四组各自的缺失数量；
- 各组（行为、态度、医嘱）的求和，作为"整体积极性"；
- 疫苗态度净意愿：`有效性 + 风险 - 副作用担忧`，以及有效性与风险的交互；
- 两种疾病的态度差异、家庭成员总数、知识 × 关注度；
- 年龄、教育、收入的显式有序编码，避免树模型把有序变量当无序切分。

是否为每个模型启用这套特征由 Optuna 自己决定，而不是硬编码。最终 LightGBM 的两个目标都启用了，CatBoost 只在 H1N1 上启用，XGBoost 都没启用。

类别编码按模型区分：LightGBM 用原生类别（`pandas.Categorical`），XGBoost 用有序整数编码，CatBoost 用字符串类别加缺失类别。

### 3. 三种梯度提升模型分别调参

对 LightGBM、XGBoost、CatBoost 和两个目标，共 6 组配置，各自用 [Optuna](https://optuna.org/)（TPE）在开发集 3 折上最大化 AUROC：

- 搜索预算：LightGBM 14 次、XGBoost 8 次、CatBoost 6 次，每个目标 → 共 56 组参数、每组 3 折；
- 每折最多 2200 轮，90 轮早停，记录最佳轮数；
- 取 3 折最佳轮数的中位数作为最终轮数，避免被单折极值带偏；
- 调参结果存在 `artifacts/search.sqlite`，中断后可以续跑。

### 4. 融合

在开发集的 out-of-fold 预测上，从一组**预先定义的小网格**里选融合方式：

- 权重候选：等权、0.5/0.25/0.25 及其排列、0.4/0.2/0.4 等共 6 种；
- 融合空间：概率平均，或对数几率（logit）平均。

两个目标最终都选中"等权 + 概率平均"。限制在小网格里，是为了避免选出"某个单模型 + 极端权重"这种在测试集上容易失效的解。

### 5. 两种训练路径打包

配方冻结后，每个配置训练两条路径：

| 路径 | 做法 |
|---|---|
| 折模型 | 全量训练集上 5 折，每折用 `1.12 × 冻结轮数` 训练（不再早停），预测后取 5 折平均 |
| 全量模型 | 用全部训练数据、3 个随机种子，轮数再乘 1.08 |

- 6 个配置 × 5 折 = 30 个折模型；
- 6 个配置 × 3 种子 = 18 个全量模型（用 joblib 保存）；
- 两者各占 50% 就是 `submission_robust.csv`。这个 1:1 比例是**看保留集之前**就定好的规则。

### 6. 联合四状态模型

两个标签并不独立（大多数人两个都打或都不打），所以额外训练一个四分类模型：

```
state = 2 * h1n1 + seasonal  ∈ {0, 1, 2, 3}
```

再从类别概率边际化得到两个目标的概率：

```
P(h1n1=1)     = P(state=2) + P(state=3)
P(seasonal=1) = P(state=1) + P(state=3)
```

**真实标签从不作为输入特征**，只作为训练目标，因此不存在目标泄漏。6 组候选在开发集上筛选，选中 `num_leaves=25`、不启用组合特征；再在开发集上从 `{0, 0.15, 0.3, 0.5}` 里为每个目标选联合模型权重，两个目标都选中 0.5。最后同样做 5 折 + 3 种子打包，得到 `submission_joint_*.csv`。

---

## 结果与可信度

同一批 5876 条保留样本，参数和融合权重均先在开发集选择：

| 模型 | 保留集平均 AUROC |
|---|---:|
| 参考 notebook LightGBM 参数复现 | 0.86840886 |
| LightGBM + XGBoost + CatBoost | 0.86960108 |
| 三模型 + 四分类联合模型 | **0.87003861** |

- 相对基线：三模型 **+0.001192**，加联合模型 **+0.001630**。
- 配对 bootstrap（对保留行重采样 1000 次）差值 95% 区间：`[0.000599, 0.002720]`。区间不跨 0，但它只反映固定模型下的行抽样波动，**不包含训练波动和官方测试集分布差异**。
- 开发集：三模型 0.867864 → 加联合 0.868903。
- 单目标保留集 AUC（三模型）：H1N1 0.873763，季节性流感 0.865439。
- 全数据描述性 OOF：0.870489。这个数字与调参用的开发集有重叠，不是无偏估计，仅作参考。

**请理性看待本地分数**：0.0016 的提升在 5876 条样本上并不算大，官方成绩可能低于本地值。竞赛排行榜也出现过多名选手四舍五入后同为 0.8647 的情况，所以本地 0.87 不能直接等同于某个名次。

---

## 完整复现

### 运行顺序

```bash
python experiment.py --lgb-trials 14 --xgb-trials 8 --cat-trials 6 --folds 5
python baseline.py
python joint.py
python verify.py
python predict.py
python report.py
```

| 脚本 | 作用 |
|---|---|
| `experiment.py` | 读取数据、划分开发/保留集、三模型调参、冻结融合配方、训练折模型与全量模型、生成 `submission_*.csv` 和 `metrics.json` |
| `baseline.py` | 用参考 notebook 的超参数在同一保留集上训练，作为对照 |
| `joint.py` | 筛选并训练四状态联合模型，融合后生成 `submission_joint_*.csv` |
| `verify.py` | 校验所有提交文件并计算配对 bootstrap |
| `predict.py` | 从保存的模型重新推理，确认导出文件没有错位 |
| `report.py` | 汇总各 JSON，生成 `REPORT.md` |

### 复现相关约定

- 主随机种子：`20261115`；全量模型另用 `42`、`2026`。
- 开发/保留划分保存在 `artifacts/split.npz`，重新运行不会改变划分。
- 环境与数据哈希记录在 `artifacts/provenance.json`。
- 缓存文件很多，脚本会优先读缓存。要确保"真的重训"，请清空 `artifacts/` 或换新目录。
- 默认每个模型使用 4 个线程（`n_jobs=4` / `thread_count=4`），可通过环境变量 `OMP_NUM_THREADS` 调整。

### 计算量

- 调参：56 组参数 × 3 折（二分类）+ 6 组 × 3 折（联合模型）；
- 最终：30 个折模型 + 18 个全量模型 + 5 个联合折模型 + 3 个联合全量模型 = 56 次训练。

建议在多核机器上运行，内存建议 8GB 以上。只想验证结果的话，用第 3 步的两条命令即可，无需重新训练。

---

## 常见问题

**Q：运行时报 `FileNotFoundError: ... training_set_features.csv`？**
说明数据没放对。四个 CSV 必须直接在 `data/` 下，文件名不能改。脚本也兼容旧布局（数据放在仓库上一级的 `data/`），但推荐使用仓库内的 `data/`。

**Q：我改了参数或折数，为什么结果没变？**
因为脚本读了 `artifacts/` 里的缓存。请删除对应缓存，或使用全新的实验目录。**改变搜索空间、折数以外的训练设置时，最好新建目录**，避免新旧缓存混用。

**Q：本地分数和官网分数为什么对不上？**
本地保留集只有 5876 条，抽样波动不小；官方测试集是另一批人，分布也可能略有差异。本地分数用于比较方案，不用于预测名次。

**Q：可以直接用仓库里的模型吗？**
可以，`predict.py` 就是从这些模型重新推理的。但 `.joblib` 是 pickle 格式，只加载可信来源的文件。

**Q：为什么没有把数据一起上传？**
受竞赛数据使用条款限制，且仓库体积会明显变大。请按上面的官网链接自行下载。

**Q：跑得慢 / 内存不够？**
把 `OMP_NUM_THREADS` 设小、减少 `n_jobs`，或者减少 `--*-trials` 的搜索次数。也可以只跑 `joint.py`（它会复用 `experiment.py` 的冻结配方）。

---

## 设计取舍

这套流程刻意避开了一些常见的、会抬高本地分数的做法：

1. **早停不重训**：在训练折内部再切验证集做早停，然后用早停模型直接预测外层验证集——这会低估训练集规模、高估泛化。本方案按选定轮数在完整折上重训。
2. **跨目标特征泄漏**：用另一套划分产生的 out-of-fold 预测当特征，生成这些特征的模型可能见过当前验证行的标签。本方案不使用这类特征。
3. **目标编码不一致**：训练行用自身标签统计、验证行用训练集统计，两套编码分布不同会造成偏差。本方案不使用目标编码。
4. **在 OOF 上再拟合融合权重**：对一次生成的 OOF 再划折拟合 stacking 权重，不等价于严格嵌套交叉验证。本方案改为"先冻结、后评估"的保留集检查。
5. **行采样参数失效**：LightGBM 只设 `subsample` 而不设 `subsample_freq`，行采样实际不会生效。本方案显式设置了 `subsample_freq=1`。

这些取舍也是本地分数没有虚高的原因之一。

---

## 声明

- 训练与预测只使用竞赛提供的训练数据，**未使用外部数据，未使用测试集标签**。
- `baseline.py` 中的超参数来自公开的参考 notebook（Smith-MMA），仅用于同口径对照。
- 本仓库未记录任何官方提交成绩；README 与 REPORT 中的分数均为本地验证结果。
- 欢迎提 issue 或 PR 讨论改进方向（例如更充分的多种子 bagging、两阶段条件模型等）。