# Flu Shot Learning：冲前 20 名的完整实验方案

本目录是 **Flu Shot Learning（预测 H1N1 与季节性流感疫苗接种）** 数据挖掘大作业中效果最好的一轮实验。

目标：官方 public AUROC **> 0.8647**，并进入排行榜 **前 20 名**。

> 重要前提：本地验证分数不能替代官方成绩。本目录尚未记录本轮文件的官方提交结果，因此不作"已达标"的结论。2026-11-15 核对公开榜时，第 17–24 名均显示 0.8647，所以"显示 0.8647"本身并不足以保证前 20。

---

## 1. 任务与数据

- 训练集 26708 行，测试集 26708 行，两个二分类目标：`h1n1_vaccine`、`seasonal_vaccine`。
- 评价指标为两个目标 AUROC 的平均值。
- 特征约 35 列，混合了数值、有序类别（年龄、教育、收入）和无序类别（种族、性别、地区、职业等），且存在大量缺失。
- 数据位于上一级 `data/` 目录：四份官方 CSV。
- `respondent_id` 只用于对齐，**不进特征**。
- 全程**不使用外部数据，不读取/不使用测试标签**。数据指纹（SHA-256）记录在 `artifacts/provenance.json`。

原始特征大致可分为四组，后续特征工程正是围绕这个结构展开：

| 组别 | 示例 | 主要问题 |
|---|---|---|
| 认知与态度 | `h1n1_concern`、`h1n1_knowledge`、`opinion_*` | 有序但非线性的心理量表 |
| 行为 | `behavioral_*`（洗手、戴口罩、避免集会…） | 高度相关，缺失集中 |
| 医嘱与风险 | `doctor_recc_*`、`chronic_med_condition`、`health_worker` | 强预测信号，缺失本身有含义 |
| 人口学 | `age_group`、`education`、`income_poverty`、`race`… | 有序/无序混杂，缺失率高 |

---

## 2. 核心思路（先看这一节）

这类表格竞赛中，单模型调参的边际收益很快耗尽，真正决定名次的是三件事：**验证协议是否可信、融合是否稳、有没有利用标签之间的结构**。因此本方案的设计原则是：

1. **先把"选择"和"评估"彻底分开。**
   78% 数据做开发集（调参 + 选融合权重），22% 做保留集。所有超参数、迭代轮数、融合权重都在**不接触保留集标签**的前提下冻结，冻结之后才评估一次保留集。这样可以避免"反复看验证集调参"造成的乐观偏差。
2. **多模型、多随机性做方差压缩。**
   LightGBM / XGBoost / CatBoost 三种不同归纳偏置的 GBDT 分别调参，再等权融合；最终预测同时用"5 折折模型平均"和"全量数据多种子平均"两条路径，并按 1:1 打包。
3. **利用两个标签的结构。**
   两个目标相关（都接种、都不接种是多数），因此额外训练一个**四分类联合模型** `(h1n1, seasonal) ∈ {00,01,10,11}`，再对联合概率做边际化得到两个目标概率，与独立二分类模型融合。
4. **只做小幅度的、可解释的增益。**
   融合权重来自预先定义的小网格，而不是无约束优化；每一步增益都用配对 bootstrap 给出不确定性区间。

---

## 3. 特征工程

`experiment.py` 的 `features()` 支持三套编码（`kind`）和可开关的特征工程（`engineering`），训练时把"是否开启"也交给 Optuna 搜索，让数据决定它是否有用。

### 3.1 通用处理

- 数值特征按模型需要填补（CatBoost 用 `-999`，其余交给模型自身处理）。
- 类别编码按模型区分：
  - `lgb`：转成以训练集取值为类别的 `pandas.Categorical`，交给 LightGBM 原生类别处理；
  - `xgb`：按训练集取值排序后映射为整数（有序编码）；
  - `cat`：转字符串并保留缺失为单独类别，交给 CatBoost 的 `cat_features`。

### 3.2 组合特征（`engineering=True` 时启用）

| 特征 | 含义 / 动机 |
|---|---|
| `missing_count` | 单行缺失总数，缺失模式本身携带信息 |
| `behavioral_missing` / `opinion_missing` / `doctor_recc_missing` / `employment_missing` | 分组缺失数量 |
| `behavioral_sum` / `opinion_sum` / `doctor_recc_sum` | 组内行为/态度强度总和 |
| `{h1n1,seas}_opinion_net` | 有效性 + 风险 − 副作用担忧，构造"净意愿" |
| `{h1n1,seas}_eff_risk` | 有效性与风险认知的交互 |
| `risk_difference` / `effective_difference` | 两个疾病之间态度差异 |
| `household_total` | 家庭成人数 + 儿童数 |
| `knowledge_concern` | 知识 × 关注度交互 |
| `age_group_ordered` / `education_ordered` / `income_poverty_ordered` | 显式有序编码，避免树模型把有序变量当无序切分 |

特征工程不是默认开启，而是作为搜索维度交给数据决定。最终 6 组配置里：LightGBM 的两个目标都选中，CatBoost 只在 h1n1 上选中，XGBoost 两个目标都未选中——说明它的收益依赖于模型对类别的处理方式，这正是不能硬编码的原因。

---

## 4. 验证协议

这是整个方案里最需要讲清楚的部分。

```
全部训练数据 26708
├── 开发集 DEV 78%  (20832)  ── 调参 / 选融合权重
│     └── 内部 3 折 StratifiedKFold（按 2*h1n1 + seasonal 联合分层）
└── 保留集 HOLDOUT 22% (5876) ── 配方冻结后只评估一次
```

- 分层键统一用 `2*h1n1 + seasonal`（四种类别组合），保证两个目标的比例在开发集、保留集、各折内部都被保持。
- 随机种子固定为 `20261115`，划分结果落在 `artifacts/split.npz`。
- **冻结顺序**：调参 → 在开发集 OOF 上选融合权重 → 写 `artifacts/frozen_recipe.json` → 之后才计算保留集分数。保留集不参与任何选择。
- 最终报告的"全数据 OOF AUC"只是**描述性指标**：所用数据与调参开发集重叠，不是无偏估计。README / REPORT 中明确标注了这一点。

---

## 5. 单目标模型与调参

对 `lgb` / `xgb` / `cat` × 两个目标，共 6 组配置，各自独立用 Optuna（TPE）在开发集 3 折上最大化该目标的 OOF AUROC。

- 搜索预算：LightGBM 14 次、XGBoost 8 次、CatBoost 6 次，每目标 → **56 组参数，每组 3 折**。
- 每折用 2200 轮上限 + 90 轮早停（监控该折 AUROC），记录最佳轮数。
- 搜索完成后取 3 折最佳轮数的**中位数**作为该配置的迭代轮数，避免被单折极值带偏。
- Optuna 结果持久化在 `artifacts/search.sqlite`，支持断点续跑；每折 OOF 预测缓存为 `artifacts/tune_*.npz`。
- LightGBM 显式设置 `subsample_freq=1`。这点很关键：旧 notebook 只给了 `subsample` 而没给 `subsample_freq`，行采样实际从未生效。

选中的 6 组配置见 `artifacts/frozen_recipe.json`（含参数、迭代轮数、开发集 AUC）。

---

## 6. 融合

融合在**开发集 OOF** 上进行，且只在一个很小的、预先定义的候选集合里选：

- 权重网格：`[1/3,1/3,1/3]`、`[.5,.25,.25]`、`[.25,.5,.25]`、`[.25,.25,.5]`、`[.4,.2,.4]`、`[.2,.4,.4]`；
- 融合空间两种：概率平均（`probability`）与对数几率平均（`logit`）。

两种空间各自选最优，共 12 个候选。最终两个目标都选中了**等权 + 概率平均**（开发集 AUC 0.870775 / 0.864953）。限制在凸组合小网格里，是为了避免"选出某个单模型 + 极端权重"这种容易在测试集上崩掉的解。

---

## 7. 最终训练与三种打包

配方冻结后，对每个配置做两种训练路径：

| 路径 | 做法 | 产物 |
|---|---|---|
| 折模型 | 全量训练集上 5 折，每折用 `rounds = 1.12 × 冻结轮数` 训练（**不再早停**），预测测试集后取 5 折平均 | `submission_foldbag.csv` |
| 全量模型 | 用**全部**训练数据、3 个种子（`SEED, 42, 2026`），轮数再乘 1.08 | `submission_fullbag.csv` |
| 稳健打包 | 上面两者各占 50% | `submission_robust.csv` |

- 折模型提供"每条测试样本被多个不同训练子集预测"的 bagging 收益；全量模型提供更多数据带来的偏差下降。1:1 混合是**在看保留集之前**就定好的规则，不是事后调的。
- 规模：6 配置 × 5 折 = 30 个折模型；6 配置 × 3 种子 = 18 个全量模型，共 48 个。18 个全量模型用 joblib 保存在 `artifacts/model_*.joblib`，折模型预测缓存于 `artifacts/final_*.npz`。
- 基线对照 `baseline.py` 在同一保留集上复现 Smith-MMA 的 LightGBM 参数（one-hot + 有序编码管道），作为"起点"。

---

## 8. 联合四状态模型

两个标签并不独立，直接用两个独立二分类器会丢掉这部分信息。

`joint.py` 把标签合成为 4 类 `state = 2*h1n1 + seasonal`，训练一个多分类 LightGBM（`objective='multiclass', num_class=4`），再从类别概率边际化：

```
P(h1n1=1)    = P(state 2) + P(state 3)
P(seasonal=1)= P(state 1) + P(state 3)
```

注意：**真实标签从不作为输入特征**，只作为训练目标，因此不存在目标泄漏。

流程：

1. 6 组候选（特征工程开/关 × `num_leaves ∈ {7,15,25}`），开发集 3 折早停评估，选中 `engineering=False, num_leaves=25`（开发集 AUC 0.867793）。
2. 对每个目标，在开发集 OOF 上从 `{0, 0.15, 0.3, 0.5}` 里选联合模型权重（`0` 表示完全不用）。两个目标都选中 **0.5**。
3. 冻结后同样做 5 折 + 3 个全量种子，再与三模型预测按权重融合，产出 `submission_joint_{robust,foldbag,fullbag}.csv`。

**首选提交是 `submission_joint_robust.csv`。**

---

## 9. 结果

同一批 5876 条保留样本，参数和融合权重均先在开发集选择：

| 模型 | 保留集平均 AUROC |
|---|---:|
| Smith-MMA LightGBM 参数复现（基线） | 0.86840886 |
| LightGBM + XGBoost + CatBoost 等权融合 | 0.86960108 |
| 三模型 + 四分类联合目标 | **0.87003861** |

- 相对基线的提升：三模型 **+0.001192**，加联合模型 **+0.001630**。
- 配对行 bootstrap（1000 次）差值 95% 区间：`[0.000599, 0.002720]`（联合 vs 基线）。区间不跨 0，但**只反映固定模型下的行抽样波动**，不包含训练波动和官方测试集分布差异。
- 开发集：三模型 0.867864 → 加联合 0.868903。
- 全数据描述性 OOF：0.870489（非无偏估计，仅作参考）。
- 单目标保留集 AUC（三模型）：h1n1 0.873763，seasonal 0.865439。

**乐观程度要诚实看待**：保留集只针对"本轮搜索"隔离；项目更早的实验已经用过全部训练数据，所以它不是整个项目从未见过的数据。0.0016 量级的提升在 5876 行上的标准误并不小，官方成绩很可能低于本地值。

---

## 10. 复现

本机解释器：`/Users/f.fantasiachopin/miniconda3/bin/python`。环境版本见 `requirements.lock.txt`。

仓库只包含本实验目录；脚本从父目录读取比赛数据，因此需把官方 `data/` 放在仓库目录的上一级（即 `../data`）。

```bash
python experiment.py --lgb-trials 14 --xgb-trials 8 --cat-trials 6 --folds 5
python baseline.py
python joint.py
python verify.py
python predict.py --limit 512
python report.py
```

- `experiment.py` 自动以脚本位置定位 `../data`，可从任意工作目录调用。
- 各阶段都会读取 `artifacts/` 缓存（`search.sqlite`、`tune_*.npz`、`final_*.npz`、`joint_search_*.npz`），中断后可续跑。
- **更换参数搜索空间、折数以外的训练设置或数据时，请新建实验目录**，否则会误用旧缓存，导致"结果复现不出来"。
- `predict.py` 不训练，只从保存的模型重新推理，并用 `np.allclose(atol=1e-9)` 断言与已导出的 `submission_fullbag.csv` / `submission_joint_fullbag.csv` 完全一致（结果见 `inference_verification.json`），用于证明"提交文件确实来自这些模型"。
- `verify.py` 校验全部 6 份提交文件的列名、26708 行、ID 顺序与唯一性、概率范围与有限值，记录 SHA-256，并计算配对 bootstrap。

---

## 11. 文件清单

| 文件 | 说明 |
|---|---|
| `experiment.py` | 主流程：特征、三模型调参、保留集验证、5 折/全量训练、生成提交 |
| `baseline.py` | 复现 Smith-MMA LightGBM 参数，作为同保留集基线 |
| `joint.py` | 四分类联合目标模型，`--screen` 只做候选筛选 |
| `verify.py` | 提交文件格式校验 + 配对 bootstrap |
| `predict.py` | 从已保存模型重新推理，校验导出一致性 |
| `report.py` | 由各 JSON 结果生成 `REPORT.md` |
| `run.log` / `joint_screen.log` / `joint_final.log` | 逐轮搜索、逐折训练日志 |
| `metrics.json` | 三模型保留集 / 描述性 OOF / 官方结果状态 |
| `joint_metrics.json` | 联合模型各阶段 AUC 与权重 |
| `baseline_metrics.json` | 基线成绩 |
| `verification.json` | 六份提交文件的哈希、取值范围、bootstrap 区间 |
| `inference_verification.json` | 重新推理与导出文件的最大绝对误差 |
| `artifacts/frozen_recipe.json` | 保留集评估**之前**冻结的模型与融合配方 |
| `artifacts/joint_frozen_recipe.json` | 联合模型选择结果与融合权重 |
| `artifacts/provenance.json` | 数据 SHA-256、Python 与依赖版本、种子 |
| `artifacts/model_*.joblib` | 21 个全量训练模型（**只加载自己生成且可信的模型**） |
| `artifacts/split.npz` | 开发/保留划分索引 |
| `artifacts/predictions.npz` | 保留集、OOF、测试集预测数组 |
| `submission_joint_robust.csv` | **首选提交**：三模型 + 联合模型，折平均与全量种子平均各 50% |
| `submission_joint_foldbag.csv` / `submission_joint_fullbag.csv` | 联合模型的两个备选打包 |
| `submission_robust.csv` / `submission_foldbag.csv` / `submission_fullbag.csv` | 未加联合模型的对应版本 |

首选文件 SHA-256：`39f6df72167682c1c20e7634ce801a873f15d6fa7b3226a57b2316c46f4a114c`。

---

## 12. 旧代码审计：为什么重写而不是沿用

这不是"换个模型再跑一次"，旧流程里存在几处会系统性抬高本地分数或削弱泛化的做法，本轮全部弃用：

1. **早停泄漏**：旧 `src/06_boost_v3.py` 在训练折内部再切 12% 做早停，却直接预测外层验证集，没有按选定轮数重训完整训练折。本轮改为"固定轮数 + 完整折训练"。
2. **跨目标 OOF 泄漏**：旧 `src/07_screen_v4.py` 把另一套划分产生的跨目标 OOF 当作外层 CV 特征，生成模型可能见过外层验证行的标签。严格做法须在每个外层训练折内部重新生成，本轮直接不用。
3. **目标编码机制不一致**：训练行用自身标签统计、验证行用训练集统计，两套编码分布不同。本轮不用目标编码。
4. **融合权重拟合**：对一次生成的全数据 OOF 再划折拟合 stacking 权重，并不等价于对整个底层训练过程做嵌套 CV。本轮改为"先冻结、后评估"的保留集检查。
5. **行采样未生效**：Smith-MMA LightGBM 只设 `subsample` 未设 `subsample_freq`，搜索项形同虚设。本轮显式设置。

规则页面已核对：外部数据默认不允许，每日提交次数有上限；所有官方分数必须来自本人账号的真实提交记录，不能用本地 AUC 推算或替代。

---

## 13. 局限与下一步

- 0.0016 的提升幅度接近噪声量级，需要官方结果验证；本地保留集已被前几轮实验"间接看过"。
- 未做严格的嵌套 CV，因此不能声称本地分数是无偏泛化估计。
- 想继续提升，优先级大致是：**增加随机种子数做更充分的 bagging** > **在联合模型上做更细的类别结构建模（如两阶段条件模型 `P(h1n1)·P(seasonal|h1n1)`）** > **更精细的调参**。单纯扩大搜索空间在本数据规模上收益有限。
- 获得首次官方结果后，应把提交 ID、文件名、哈希、官方分数与排名一起记录下来，再决定后续提交策略；不要用本地分数替代官方反馈。