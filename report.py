"""Render a factual local experiment report after all verification completes."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
m,j,b,v=map(read,['metrics.json','joint_metrics.json','baseline_metrics.json','verification.json'])
c=v['joint_paired_holdout_comparison']
text=f'''# 实验结果：待官方评分

实验目录：flu_top20_20261115。目标为官方 AUROC > 0.8647 且排名 ≤20；当前尚未提交本轮文件，不能声称达标。

## 本地验证

同一批 {m['holdout_n']} 条保留样本，参数和融合权重均先在开发集选择。

| 模型 | 保留集平均 AUROC |
|---|---:|
| Smith-MMA LightGBM 参数复现 | {c['smith_lgb_auc']:.8f} |
| LightGBM + XGBoost + CatBoost | {m['holdout_mean_auc']:.8f} |
| 三模型 + 四分类联合目标模型 | {j['blended_holdout_auc']:.8f} |

联合模型融合相对原 LightGBM 差值：{c['difference']:+.8f}。
配对行 bootstrap 的差值 95% 区间：{c['bootstrap_95pct_difference']}。
该区间只反映固定模型下保留行的抽样波动，不涵盖训练波动或官方测试集分布。

开发集三模型平均 AUC：{j['base_development_auc']:.8f}；加入联合模型后：{j['blended_development_auc']:.8f}。
两个目标的联合模型权重依次为 {j['joint_weights']}。
最终全数据描述性 OOF AUC：{j['blended_descriptive_cv_auc']:.8f}，因参数选择与其中部分数据重叠，不属于独立无偏估计。
保留集仅针对本轮搜索隔离，项目此前的实验已使用过全部训练数据。

## 首选提交

`submission_joint_robust.csv`：三模型预测与联合四分类边际概率融合；每个模型的五折平均与全量三种子平均各占 50%。首选依据是开发集预先冻结的融合选择。

文件 SHA-256：`{v['submission_joint_robust.csv']['sha256']}`。

所有六份候选文件均通过列名、26708 行、ID 顺序、ID 唯一、概率范围和有限值检查。`verification.json` 保存完整校验。
`inference_verification.json` 检查从已保存全量模型重新推理是否与已导出 fullbag 文件一致。

## 工作量与复现

二分类搜索 56 组参数，每组 3 折；联合目标搜索 6 组参数，每组 3 折。
最终二分类模型使用 30 个折模型 +18 个全量模型，联合模型使用 5 个折模型 +3 个全量模型。
21 个全量模型已保存；折模型预测已缓存，可按脚本重训折模型。

运行顺序：`experiment.py` → `baseline.py` → `joint.py` → `verify.py` → `predict.py --limit 512` → `report.py`。
环境见 requirements.lock.txt；原始数据指纹见 artifacts/provenance.json。

## 官方反馈状态

本轮尚未上传或消耗提交机会。当前账号 Fantasia 此前最佳成绩 0.8636、排名127，为浏览器检查时的状态。
排行榜检查时第17–24名均显示0.8647；显示到四位小数的0.8647并不保证进入前20。
获得首次官方结果后，应将提交ID、完整文件名、文件哈希、官方分数与排名一起保存，再决定后续提交。
'''
(R/'REPORT.md').write_text(text)
print(text)
