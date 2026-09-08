# Adaptive VSV law results（待远端运行）

当前提交只包含可复现代码、审计和 CPU 单元测试，没有模型、COCO 数据或实验输出。

运行后应填写：

- fixed `lambda=0`、fixed `0.17` 与各 adaptive controller 的 CHAIRs/CHAIRi/recall/precision/F1；
- calibration/held-out 的 oracle/MED dose 相关性与 image-bootstrap CI；
- `theta` 分布是否比 raw lambda 更集中；
- legacy/off gate 的 effective lambda 统计；
- SLA-off law discovery 与 SLA-on (`logits_alpha=0.3`) confirmation 分开报告。

若 held-out 相关性弱或 safe set 非单调，不应强行宣称存在 universal dose law。
