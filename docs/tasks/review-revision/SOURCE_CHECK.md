# 本轮公开材料核对（2026-10-05）

本记录由 root 在构建前核对；不将用户 review 的百分比评级当作测量结果。当前 PR 已有 `spectral-camera-v2` 的 61 波段链路，因此 review 中“三个 delta 波长”的批评针对较早快照，不能据此撤回已有光谱实现。

| 证据 | 可以确认 | 仍不能确认 |
| --- | --- | --- |
| [作者原仓库 README](https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/blob/main/README.md#data-flow) 的连续数据流与方法描述 | 全局单应、参考/跨帧约束、可靠局部对应、最高四次二维残差并在控制不足时降阶；learned/legacy 使用相同 LCA 几何；a_C 从初始 legacy、a_R 从变换后 legacy 重新计算 | 特征选择、匹配阈值、残差基、正则项、镜头校准拟合及边界处理公式。新增代码应标为独立装配 |
| 同一 README 的轴向控制、色差和阶梯环段落 | 半像素通道位移、14 帧无噪声/精确运动；轴向控制 0.20/0.35 周每原生像素；固定几何 29 DN 档、暗部加密、每像素 16 点积分；分别衡量非线性残差与恒等误差 | 原始连续靶坐标、具体 29 个 DN 码值、所有作者输入文件、15 个像差条件，不能编造“exact author fixtures” |
| 同一 README 的暗部实验 | S5M2 的等曝光短连拍，独立长曝光参考；九档是重建后显示提升而非传感器增加九档 | 本地没有原始连拍/长曝文件，截图不能用于相同传感器线性定量复测 |
| 用户提供的知乎讲稿 | 是本轮七帧晚融合 Transformer 的机制依据 | 无可验证的原文 URL、完整网络/权重/训练集/PTC 库；公开旧核心移植与讲稿启发的模型继续分开 |

补充公共数据源已经核对：[DBSR 官方仓库](https://github.com/goutamgmb/deep-burst-sr)、[CVPR 2021 论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf) 使用合成 RAW 预训练及真实 BurstSR；真实手机 burst 与 DSLR GT 是跨相机评价，需要其空间/颜色评价协议，不能直接当成 S5M2 的独立长曝参考。inverse-sRGB 是相关工作的合成代理，无法恢复已经剪裁的高光或真实场景光谱。

[HDR+ 官方数据详情](https://hdrplusdata.org/dataset.html) 提供 Android 相机的 2–10 帧 DNG 连拍、元数据和融合输出；该融合输出并非独立高分辨率长曝光 GT。官方 curated subset 为 153 bursts、37 GiB，完整集为 765 GiB，本轮没有批量下载它，也不将它记作已执行的 S5M2 九档实验。该来源可用于另行声明的真实 RAW 工程测试。

GitHub API 确認 fork 为 public（private=false），本轮起点 PR #1 HEAD 为 `2d4148bd8d9f237e2d4ef9e21425e862928968ce`，main 为 `6ab0f5bf76d2ccfbf7e21798cc575232ef9b272b`。修改前本地测试：`python -m pytest -q -p no:cacheprovider --basetemp ../pytest-revision-baseline`，63 passed。
