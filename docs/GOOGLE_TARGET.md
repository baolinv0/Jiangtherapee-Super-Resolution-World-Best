# Google 2019 启发的重建目标与协议迁移

本工程将新光谱实验的默认目标改为 `spectral-camera-v3` / `target_stage: post_pixel`：在参考帧 0 坐标中，以原生像元的感光面积对同一套波长/视场 PSF 作用后的图像积分，再在 2× 网格上密集取样。保留 `post_optics` 和 `pre_optics` 两个显式消融选项。七帧 Transformer 架构不变；这是一项独立工程目标选择，不是 Google 算法、其训练 GT 或 JSR 作者意图的复现。

## 事实、推断与工程选择

一手来源为 Wronski 等，*Handheld Multi-Frame Super-Resolution*，ACM TOG 2019，[Google 托管论文 PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/5211.pdf)。

| 类型 | 可以成立的陈述 | 不能由它直接推出的结论 |
|---|---|---|
| 论文事实，§5.1，PDF 第 6 页 | 按颜色通道，把不同帧的 RAW 测量视为对连续信号的不规则、有噪采样，用加权核重建并归一化 | 论文没有因此规定本工程的监督学习 GT 公式 |
| 论文事实，§6.3，PDF 第 11 页 | 实拍比较中的线性输出仍受镜头模糊；后续单独使用 unsharp mask 和全局色调映射 | 不能将该描述称为不可超越的理论极限，也不能等同于一个精确的像元反卷积结论 |
| 工程推断 | 将光学与原生像元响应保留在目标中，可以把采样重建与显式去模糊的训练职责分开 | 这不证明 Google 的输出恰好等于下述 `post_pixel` 数学定义 |
| 本工程选择 | 默认 `post_pixel`，同时提供两个上游阶段作为受控消融 | 选型改变监督任务，不代表新目标更接近所有应用需要的最终显示图像 |

本说明不把提高清晰度、去镜头模糊、去像元响应混为同一任务。模型仍可能从学习先验产生额外细节；验证应检查相应目标的误差与伪影，而不是把 2× 输出尺寸解释为两倍可恢复频带。

## 三个目标阶段

令 `Lλ` 为场景增益后的参考光谱辐亮度，`Oλ` 为参考帧的波长/视场光学算子，`Sc` 为 D65 白归一化的相机光谱响应积分，`P` 为原生方形有效像元面积的归一化积分，`D2` 为 2× 网格取样。下面的算子只是本工程定义。

| `target_stage` | 目标定义 | 训练任务所要求的额外反演 | 光学改变时 GT | 填充率改变时 GT |
|---|---|---|---|---|
| `pre_optics` | `D2 Sc[Lλ]` | 去新增光学模糊及像元面积响应 | 不变 | 不变 |
| `post_optics` | `D2 Sc[Oλ Lλ]` | 去像元面积响应 | 可改变 | 不变 |
| `post_pixel`（v3 默认） | `D2 P Sc[Oλ Lλ]` | 重建已有光学和像元响应后的连续图像 | 可改变 | 可改变 |

“可改变”要求使用有空间结构且相应退化非退化的输入来验证；常量平场本来就应保持不变。GT 与 RAW 使用同一采样的 61 波段 PSF、视场节点、相机 SRF、像元参数及参考帧 0；不得为 GT 再抽取一次光学状态，也不得先合成 RGB 后套一个替代三通道 PSF。逐帧输入继续先移动光谱场景、再施加固定在传感器坐标中的视场光学。

三个阶段的目标均为参考曝光的 **D65 归一化 camera RGB**，可以大于 1。它们均不含 CFA、逐帧曝光倍数、通道透光率、噪声、满阱/ADC 截断或量化。场景增益属于场景辐亮度，会作用于 GT；它与观测曝光倍数不同。模型输入仍为带这些观测效应的七帧 RAW，曝光和透光率校正回目标的参考颜色/亮度坐标。

## 网格中心与像元面积

设原生像元间距为 `p` µm，场景网格间距为 `p/2`，margin 为 `m` 个场景网格单元。所有下式使用场景数组的索引坐标，x、y 方向独立适用：

- 原生观测中心：`q_native(j) = m + 2(j + 1/2) - 1/2 = m + 2j + 1/2`。
- 2× 目标中心：`q_dense(k) = m + k`，间隔为一个场景网格单元。
- 面积填充率为 `f` 时，感光边长是 `a = p sqrt(f)`，即 `2 sqrt(f)` 个场景网格单元。
- 每轴 `Q=4` 点中点求积的偏移：`δr = ((r+1/2)/Q - 1/2) · 2 sqrt(f)`，`r=0,…,Q−1`。
- 归一化像元积分：`P[I](q) ≈ Q^-2 Σr,s I(q + (δr, δs))`。

因此 dense 2× 仅加密中心位置，**不把像元边长改成 `p/2`，也不再追加一次面积积分**。目标从参考帧光学后的 RGB 场直接按上述原生 footprint 积分；不能把原生 RAW/RGB 简单上采样作为它。对 `align_corners=False`，网格坐标转为 `g_x = 2(q_x+1/2)/W − 1`。帧移位约定保持为 `frame(x,y)` 观察 `reference(x+dx,y+dy)`；参考帧位移为零。

原生中心位于相邻 dense 中心之间，因此 `target[..., ::2, ::2]` 不是原生干净观测。用 bilinear 对 dense target 二次取样一般也不能严格复现原生积分，应从共同的积分前场分别求值，并用中心位置、线性斜坡、平场与冲激响应检查相位和面积。

## 协议与迁移

- `spectral-camera-v3` 默认 `post_pixel`，允许三个阶段；建议配置显式写出 `target_stage`。
- `spectral-camera-v2` 固定保留历史 `pre_optics`。对 v2 指定其他阶段应报错，避免同一版本名下悄悄改变监督目标。
- 随机数 namespace 保持历史 v2 序列。相同源、seed、其余选项与资产下，切换协议/目标阶段只改变目标及其身份元数据，RAW、曝光、运动、光学抽样和噪声逐值保持一致。这使阶段消融和旧实验复查有共同观测。
- 数据身份、样本元数据、离线 manifest、checkpoint 及评测记录应带协议和目标语义。样本键包括 `target_stage`、`target_space`、`pixel_aperture`（`retained` / `excluded`）、`area_quadrature: 4`、`output_sample_pitch_native: 0.5` 和 `sampling_phase`。v3 的阶段不同即为不同训练任务；恢复训练不得悄悄跨阶段。
- 旧 v2 checkpoint 可用原 v2 recipe 继续训练/评测，不要求转换成 v3。旧配置没有 `target_stage` 时仍按 v2 `pre_optics` 解释。新目标从新运行目录开始训练；已有导出 NPZ 不自动改写。跨目标加载权重是另一个需要明确记录的迁移实验，不是严格 resume。

`configs/revision_explicit_smoke.yaml` 保留 v2 历史 recipe；`configs/spectral_smoke.yaml` 和 `configs/spectral_k7.yaml` 切为 v3 / `post_pixel`。历史 [spectral-validation](spectral-validation/) 和 [revision-validation](revision-validation/) 证据属于其原有协议，不因默认配置变更而重命名或覆盖。

## 实验设置与三个关键选型问题

1. **希望重建哪一个物理阶段？** 默认输出含原生面积响应的 `post_pixel`；需要反演像元响应选 `post_optics`；明确要去新增镜头模糊时选 `pre_optics`。三者须分别报告，不能共用一个含糊的“清晰 GT”名称。
2. **2× 表示更密的采样还是更小的像元？** 本次固定为更密采样，原生 footprint 不变。真实更小像元需要重新定义光子数量、面积响应和采样，不属于本次改动。
3. **需要等曝光重建控制还是 HDR 扩展？** `google_equal_exposure_smoke.yaml` 保持同一 K7 架构、静态场景平移，使用 `bracket_interval_ev: [0,0]`、`scene_gain: [0.05,0.4]`，用于主要未饱和条件下的目标检查；“主要未饱和”是设计目的，须实测饱和比例。原 spectral 配置保留包围曝光/高亮饱和作为本工程 HDR 任务扩展，不称为 Google 全流程复现。等曝光配置也没有复现 Google 配准、核回归、运动鲁棒性或真实采集统计。

## 强反例与最小验证

| 容易误通过的实现 | 必须能推翻它的检查 |
|---|---|
| 只改 metadata，GT 仍来自光学前 | 同一结构场景、同一 seed 的三阶段 RAW 完全一致；默认 GT 随有效 PSF 干预变化，`pre_optics` 保持不变 |
| 对 GT 使用三通道近似或另抽 PSF | 波长依赖 PSF/非均匀光谱与视场变化测试；直接使用参考帧同一光学 RGB 场核对目标 |
| dense 2× 时误把 footprint 缩半 | 独立求积、平场、斜坡、冲激/高频结构检查；`post_pixel` 应与缩小面积的反例有可测差异 |
| 光学后目标被积分两次 | `post_optics` 保持填充率不变；`post_pixel` 对照一次积分的参考值，不对已经积分的 native 图像重复滤波 |
| GT 混入曝光/透光率/噪声/截断 | 固定场景与光学，只改变这些观测设置，GT 不变；高增益场景的 GT 可超过 1 |
| 改 protocol 时顺带重新抽场景或噪声 | v2 原序列保留；v2 与 v3 三阶段比较 RAW、位移、曝光和光学状态一致 |
| 新旧 checkpoint 在不同目标间静默 resume | v2 无新增字段仍可严格恢复；v3 阶段改变被身份/配置检查拒绝 |

最低验收包括上述不变量/反例、v3 的 8 步 CPU 训练与 oracle/estimated 两种评测、单样本导出/推理、v2 严格恢复以及完整既有测试。当前变更的实测记录单独保存在 [google-target-validation/VALIDATION.md](google-target-validation/VALIDATION.md)；其中数值以实际运行结果为准。训练损失下降只证明优化可运行；不构成对目标物理正确性或自然图像质量的充分证据。

## 可运行命令

在仓库根目录运行；导出及验证使用新的空目录：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m jsr_repro.validate_spectral --config configs/spectral_smoke.yaml --output runs/spectral-v3-validation
python -m jsr_repro.validate_spectral --config configs/google_equal_exposure_smoke.yaml --output runs/google-equal-exposure-validation
```

独立准备、训练、导出与评测（均使用 procedural fixtures，只检查链路）：

```bash
python -m jsr_repro.prepare --procedural 12 --output data/smoke
python -m jsr_repro.train_transformer --config configs/google_equal_exposure_smoke.yaml
python -m jsr_repro.spectral_data --config configs/google_equal_exposure_smoke.yaml --split test --count 1 --output data/google-equal-exposure-export
python -m jsr_repro.evaluate_transformer --checkpoint runs/google-equal-exposure-smoke/last.pt --manifest data/smoke/manifest.jsonl --alignment oracle --output runs/google-equal-exposure-smoke/test-oracle.json
python -m jsr_repro.evaluate_transformer --checkpoint runs/google-equal-exposure-smoke/last.pt --manifest data/smoke/manifest.jsonl --alignment estimated --output runs/google-equal-exposure-smoke/test-estimated.json
```

阶段消融需复制 v3 配置，仅修改 `target_stage` 与 `output`；相同 seed、manifest 与其余选项保持相同 RAW。每阶段单独训练/评测。正式自然图像配置与资产接口见 [SPECTRAL_DATA.md](SPECTRAL_DATA.md)。

## 近似与证据边界

场景只在原生 2× 网格离散，亚像素运动和求积依赖 bilinear 插值；这会引入额外平滑并限制可模拟频带，不能当作连续物理真值。有限 PSF 支持、FFT 采样、4×4 求积和四节点视场插值均为数值近似；强像差、边缘、高频靶及更高精度需求应做网格、支持和求积收敛研究。RGB 提升谱仍有不可辨识性，源图像可能已有光学与 ISP 痕迹；PTC、镜头像差和参数联合分布仍含假设。

当前范围只有静态场景的帧间平移，不含动态物体、曝光轨迹、rolling shutter 或完整真实相机标定。该修改定义并验证一个更明确的合成重建任务，不证明其为 Google 的理论上限，也不代表已完成自然图像训练或真实 RAW 画质验证。
