# JSR 独立工程复现

本文记录公开v9.8模块及推断Tap的`inferred-jsr-v1`路径。新增的七帧多曝光`speech-inspired-transformer-v1`是独立入口，见[讲稿机制与Transformer说明](TRANSFORMER.md)，checkpoint不可互换。

审计日期：2026-10-05（Asia/Shanghai）。作者仓库与本 fork 的审计基线均为 [`6ab0f5bf76d2ccfbf7e21798cc575232ef9b272b`](https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/tree/6ab0f5bf76d2ccfbf7e21798cc575232ef9b272b)。

**这里交付可训练、可测试的工程实现，不宣称恢复了未公开的原训练过程，也不宣称达到原作者画质或世界最好。** 可确认的网络模块移植到 PyTorch；缺失的采样、统计、融合和训练环节采用明确命名的独立方案。原有 `Core-Only-Source-Code/` 与权重不修改。

## 证据边界

| 环节 | 公开证据 | 本实现及状态 |
|---|---|---|
| 任务域 | 原 README：扣黑 RAW burst → 高分辨率线性 RGB，位于 WB/TM 之前 | 相同任务域；不是完整手机 HDR 曝光/显示管线 |
| Controller | `wgpu_unet_tiled.py`：151→36，width32，depth3，GELU、平均池化、双线性上采样 | 可训练结构与严格 NPZ 导入；K4 模块 WGPU 数值对照通过 |
| 特征完成 | `local_v7_frontend.py`：144+RGB3+结构张量3+noise1 | 按公开定义移植，与 NumPy/OpenCV 逐值对照 |
| 局部强度 | RGB²均值、radius16/sigma4 高斯平滑、开方；特征按0/1/2次齐次阶归一化 | 保留定义、REFLECT_101 边界及零输入处理 |
| RefineNet | `wgpu_greenfilm_rarm.py`、实际 K4 NPZ | width116、8块、绿色引导 FiLM、颜色差分残差；WGPU 数值对照通过 |
| 幅度恢复 | 官方 reference amplitude=`0.34523068818006963`；归一化域 RGB clip 后乘回幅度 | 保留；不能把它误写成完全无裁剪 |
| K路由 | `weights/manifest.json`：1–14；13复用14 Controller；8–14复用14 RefineNet | 官方导入仅针对独立模块；新模型按配置训练，支持1–14输入但不代表任意K泛化已验证 |
| RAW配准 | 核心包无完整实现 | **推断**：训练为已知平移；独立估计轨道为绿色平面相位相关，不是原配准器 |
| 144维Tap生产 | 核心包只接收 phase；不能反推出完整公式 | **推断**：下述 `independent-phase-splat-v1` |
| 36控制量含义 | 未公开完整消费者 | **推断**：每色12个空间tap的softmax权重 |
| 静态仿射 | 官方 mean/std 和 calibration_ids 已公开，校准数据未公开 | 新管线默认identity；可用自身训练集校准buffer，**不套用官方统计量** |
| 原训练集/损失/优化器/日程 | 所审计快照未提供；论文全文仍标注将发布 | **推断**：公开HR图像代理RAW、L1/色差/梯度损失、AdamW+cosine |
| 知乎陈述 | 定向检索未取得可核验的 JSR 原文全文与URL | **未核实**；不把搜索摘要或第三方转述写成作者事实。见 [来源审计](SOURCES.md) |

权重维度从实际文件读取，而非沿用此前讨论中的猜测。公开 RefineNet 宽度是116，不是常见的64。

## 独立补全的数学定义

输入 `raw[B,K,1,H,W]` 为扣黑、归一化的 RGGB mosaic；`shifts[B,K,2]` 存 `(dx,dy)`，单位为原生传感器像素。帧k的 `(x,y)` 观测参考坐标 `(x+dx,y+dy)`，第0帧位移必须为零。输出为 `rgb[B,3,sH,sW]`。默认s=2；packed Bayer 的宽高只有原生mosaic的一半，因此这里的原生×2等于packed尺寸×4。

原生像素中心投影到HR格点使用 `s*(x+dx+0.5)-0.5`，两个坐标均同此式。采样坐标的native小数相位分成4×4桶。逐颜色、逐相位进行双线性splat，保存计数C、值和S、平方和Q，并以radius=`2s`、sigma=`s`的高斯核平滑。每色48个特征为：

`[C/14 (16 channels), S/C (16), sqrt(Q/C) (16)]`。

最后一组取RMS以满足公开的1次齐次阶，这是**工程选择，不是由源码证明的原公式**。零计数用安全分母；融合缺测回退到该色的burst均值。原始平方和Q不会直接作为1次齐次特征使用。

Legacy按各相位求和后的S/C生成。Controller的36个输出独立解释为每色12个tap，位于HR偏移 `dy∈{-1,0,1}, dx∈{-1,0,1,2}`。softmax后的权重分别作用于局部S和C再取比值。该操作对固定权重为线性强度响应。这一不对称tap网格及权重映射均为推断。

RefineNet接收 `[learned RGB, legacy RGB]`。两路绿色作为引导。每块先做pre-GELU残差卷积，再乘 `1+0.1*tanh(gamma)`、加beta。输出 `dG,d(R-G),d(B-G)`；第一路RGB作为残差基底。第一路的语义在新装配中选为learned，原完整装配顺序未由公开核心证明。

在固定运动、整块依赖域共同缩放、无新sensor clipping/量化时，归一化使正强度缩放保持一致。数据重新加噪/截断、重新估计运动或仅局部改变曝光不在这个契约内。

## 快速运行

Python 3.10+；实际验证为Windows、Python3.11、PyTorch CPU。Linux CI使用3.11。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
python -m jsr_repro.validate --output runs/verification
```

最后一条命令在新目录生成12个独立procedural场景、32步小模型训练、held-out测试、oracle/estimated双轨、亮度/条纹诊断、NPZ burst、线性结果和预览。**这些图仅用于工程测试。** 再次运行请使用新目录，避免覆盖已保存的checkpoint。

直接依赖的已验证版本见 [requirements-validated.txt](../requirements-validated.txt)。需要相同CPU软件版本时先安装该文件，再editable安装本项目；GPU请按PyTorch官方方式选对应CUDA构建。本次未验证CUDA训练/显存/速度；确定性操作不支持时显式报错。

## 数据与训练

见 [数据协议](DATA.md)。示例采用DIV2K官方train_HR和valid_HR；这只是合理的公开代理，不是作者原训练集，也不是官方BurstSR评分协议。

```bash
python -m jsr_repro.prepare --source /datasets/DIV2K --layout div2k --output data/train --encoding srgb
python -m jsr_repro.train --config configs/train_k14.yaml
```

正式配置采用32/116/8模块规模、原生64×64 crop、14帧、100000步。所有超参数均属推断；没有声称本提交已完成该训练。CPU调试可先把device改为cpu并用 `--stop-after 1`；正式实验应按资源和验证表现调整，并保存改动后的配置。

```bash
python -m jsr_repro.train --config configs/smoke.yaml --stop-after 16
python -m jsr_repro.train --config configs/smoke.yaml --resume runs/smoke/last.pt
```

上述smoke独立命令需要先运行 `python -m jsr_repro.prepare --procedural 12 --output data/smoke`。`--stop-after`保留完整LR日程；resume要求模型、数据、训练配置和manifest哈希相同，并保存/恢复optimizer、scheduler、RNG、epoch/batch位置。

默认联合损失：`L(rgb,target) + 0.1*L(learned,target)`；`L = RGB L1 + 0.1*色差L1 + 0.05*差分梯度L1`。均在线性域裁边后计算。没有感知/GAN损失，也不声称原作者未使用它们。

`train.stage`可选 `controller`、`refine`、`joint`。controller阶段按learned输出训练和选best；refine阶段冻结Controller。顺序训练时，在新的输出目录使用 `--initialize previous/best.pt` 转入下一阶段（仅加载模型，重置optimizer/scheduler）；`--resume`只用于同阶段连续恢复。直接选refine且不initialize会冻结随机Controller，只能作为明确标注的消融。

消融请复制配置到新的输出目录：`model.normalization: false`、`model.refinement: false`，或分别设置K=1/4/14；每项应重新训练和评估。禁用模块直接推理属于干预诊断，不能冒充公平重训消融。对应的mean/std通过 `set_feature_affine` 持久化；本版没有拟合原作者校准集的接口。

## 测试与真实输入

```bash
python -m jsr_repro.evaluate --checkpoint runs/k14/best.pt --manifest data/train/manifest.jsonl --alignment oracle --output runs/k14/test_oracle.json
python -m jsr_repro.evaluate --checkpoint runs/k14/best.pt --manifest data/train/manifest.jsonl --alignment estimated --output runs/k14/test_estimated.json
python -m jsr_repro.probes --checkpoint runs/k14/best.pt --output runs/k14/probes.json
```

同一输入、输出尺寸、线性域、固定data_range=1、同一crop，报告单帧bilinear、legacy、learned、最终输出。指标有PSNR、RMSE、R−G/B−G误差、暗部误差、均值偏置；PSNR为逐图dB平均，完全相等时封顶120dB。暗部阈值是GT均值<0.02；无暗部像素返回null。没有逐模型增益拟合或颜色校正。若使用曝光较低的样本，固定峰值PSNR自然增高，须同时看绝对误差与数据分布。

`probes`采用新定义的29级常量强度和0.20/0.35周期每native像素的轴向条纹，分别报告仿射残差与identity误差、错误正交色度能量和真实色度幅度。**它不是作者阶梯环、15像差、九档实拍或质量预测实验的逐图复现。**

```bash
python -m pip install -e ".[raw]"
python -m jsr_repro.import_raw --output data/real/burst.npz --crop 200 200 128 128 frame01.dng frame02.dng frame03.dng frame04.dng
python -m jsr_repro.infer --checkpoint runs/k14/best.pt --burst data/real/burst.npz --alignment estimated --output runs/real
```

RAW导入器扣除逐CFA通道黑电平、按对应白电平归一化，保留负读噪声；将RGB Bayer排列通过裁掉最多一行/列规范为RGGB，记录origin，不旋转、不WB、不做tone mapping。输入须同相机、相同曝光且第0帧为参考。`rawpy`适配器可选，**本次没有真实相机文件做端到端验证**。

已有NPZ可直接推理：`raw[K,1,H,W] float32`，另可含 `shifts[K,2]`；有可信运动时使用 `--alignment provided`。输出HWC float32 `linear_rgb.npy`。PNG仅作相同sRGB编码的诊断预览，不具备相机颜色校正，不能当最终摄影渲染。

这是全幅参考实现，151维HR特征内存成本高；CLI默认上限65536 native像素（256×256），默认建议128×128裁块。可以显式提高 `--max-native-pixels`，但未实现低内存全相机图像分块，**不宣称tile/full等价或手机实时部署**。简单全局平移估计不处理局部运动、遮挡、旋转、滚快门或LCA。原始RAW裁块需要留出运动和评分边界。

## 验证与下一步证据

已执行的检查和未经执行的部分见 [验证报告](VALIDATION.md)。额外的WGPU对照命令：

```bash
python -m pip install wgpu
python scripts/compare_public_core.py --output runs/public-core-parity.json
```

要升级成原方法的完整数值复现，仍需作者原始Tap生产/消费者、配准/LCA、训练集与校准集、完整损失/日程、原实验输入与评测脚本。现有工程将这些缺口集中在明确模块中，便于替换。后续应在独立真实scene与相机上正式训练、对照DBSR/BurstM等原实现；不能凭当前小样本结果裁定作者宣称。

本fork没有为上游源码或权重推定新的许可；原有第三方声明保留。使用公开数据请遵守各数据源条款。


Review revisions add optional robust dense alignment, shared v1 LCA, explicit temporal capture ranks, spherical spectral OPD and reproducible diagnostics. See [conventions, provenance, conformance and evidence](REVIEW_REVISION.md) for the independent definitions, default compatibility and real RAW NOT_RUN protocol.
