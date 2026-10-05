# 训练数据构造调研记录

日期：2026-10-05（Asia/Shanghai）。[正式方案](../TRAINING_DATA_IMPLEMENTATION_PROPOSAL.md)整合这些来源；[第一阶段计划](../superpowers/plans/2026-10-05-spectral-jsr-phase1.md)记录待执行的工程任务。

- [传感器/PTC 方法与覆盖范围](sensor/notes.md)
- [公开噪声参数和校准资产核验](sensor/assets/notes.md)
- [PMN Sony A7S II ISO100/200/400/800 数值快照](sensor/assets/pmn_sony_selected_iso_coefficients.json)
- [光学实现、处方审计与计算量](optics/notes.md)
- [实测有效相机核](optics/empirical/notes.md)
- [OpticStudio 官方代码与依赖边界](optics/empirical/zemax/notes.md)
- [光谱来源访问记录](spectral/fetch_results.json)、[补充记录](spectral/fetch_additional.json)

这些文件是研究记录与小型参数快照，没有引入光学软件依赖、真实 RAW 数据集或完整商业镜头/PTC 库。记录中的 HTTP 状态、文件头和字节数说明本次实际核验范围；外部数据包未下载时不以链接存在代替内容验证。采集数量和数值验收阈值属于建议，尚未执行或达到。

JSON 日志的 `local_cache_name` 是当时下载缓存的相对索引，所指的第三方源码/页面缓存没有在此重新发布；其 URL、状态、大小及可用的哈希用于追溯。代码、第三方资产和数据集的许可分别记录。PMN 系数快照引用固定提交及具体函数，不代表已验证逐档独立 PTC 测量或已测满阱。
