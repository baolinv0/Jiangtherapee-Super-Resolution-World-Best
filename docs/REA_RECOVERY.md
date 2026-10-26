# JSR 发布程序逆向补全

本次以 `reproduce/jsr-pytorch-engineering` 的 `cb3e010170a069c8139c2596860d8f8240aaf824` 为基础，补齐旧实现中靠推断的统计生产、连续融合和导引协方差，并恢复给定系数的径向 LCA warp。新入口使用作者原权重；旧 `JSRModel` 和历史 checkpoint 保持原来的独立定义。

**验收范围是 v9.8 的重建模块边界，不是完整论文实验或 v17 全部模式。** 配准流、RAW 导引图及 LCA 估计仍有尚未恢复的部分。

## 实际取证

直接下载并校验作者 [v9.8 发布程序](https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/releases/tag/v9.8)：

| 对象 | SHA-256 / 规格 |
|---|---|
| `JSR_V9_8.exe` | `e4fbe9a44ba508e368a7802bb5b28be1e8a49fc666d28f8306f41d34cf3576c3`，318,034,141 bytes |
| 内嵌后端 | 外层 PE 资源 `10/101/1033`，文件偏移 1,921,816，大小 312,162,740 |
| Python | 内嵌 PyInstaller archive，Python 3.10；主要算法为 Nuitka 编译扩展 |
| streaming fusion PYD | `6724796493551c05ef53f8b40196fcd55924c78963bdc24bd9d401b90d344150` |
| covariance PYD | `20acf56aa47dd483759bf43b2d948a8bf1459936050b65f3fcd96737bafedbe4` |
| LCA PYD | `5def15dc2da130048718dfa4cf3d385d40ff337f1541c3d4b8c1276cbb2bccc8` |
| local frontend PYD | `55a86032a7f040eb24f605fea2d64773d926c34a629e5ba1de00579913eef284` |

从融合扩展中直接提取四个完整 WGSL 常量，运行原 shader，与新 PyTorch 实现逐阶段比较。静态提取器不启动 EXE、不导入 PYD，也不需要 PyInstaller 或反编译器依赖。协方差和 LCA 的辨识使用独立 Python 3.10 子进程调用原扩展，固定输入并保存数值输出。

发布包的 `cv2/version.py` 明确为 OpenCV 5.0.0.93，NumPy METADATA 为 2.2.6；原扩展探针环境固定这两个版本。这个细节会改变 LCA：OpenCV 5 的浮点 remap 使用连续三次权重，旧版 OpenCV 的 1/32 像素插值表不能替代它。

第三方 Apple Silicon port 提供了定位线索，最后验收参考是作者发布程序中的 shader / 扩展。未把第三方声称的全图误差当作本项目实验结果。

## 已替换的假设

| 环节 | 历史独立路径 | 本次恢复的发布版定义 |
|---|---|---|
| 样本访问 | RAW 前向双线性 splat，再平滑统计 | 每个 HR 输出反查各帧，围绕 `floor(my),floor(mx)` 的 4×4 RAW 样本 gather |
| 输出坐标 | 感光单元中心模型 | 原程序 `oy=y/2+y0, ox=x/2+x0`，不自动插入半像素偏移 |
| phase | 投影坐标小数部分，周期 1 | 样本相对偏移 `dy,dx`，周期 2，每轴四桶 |
| 每色 48 通道 | count/14、mean、RMS | weighted count×14/K、weighted mean、weighted population standard deviation |
| Controller 36 输出 | 每色 12 个空间 tap 的 softmax | 每色 12 个连续 log-kernel 修正系数 |
| 无支持 | 回退到自行设计的 legacy | phase 空桶分母下界 1e-9；legacy 无颜色支持报错；learned 无支持为零 |
| LCA | 固定位移、双线性 | 给定 R/B 径向尺度，整图中心坐标、三次插值、BORDER_REFLECT；启用时输出裁到非负 |

4×4 访问的数组索引越界时 clamp；计算核与 phase 时仍使用未 clamp 的原格点坐标。因此边界可能重复计入同一个感光单元。

对颜色 c，Controller 系数与以下基函数点积：

```
[1, dy, dx, dy², dx², dy*dx, r², r⁴,
 cos(2πdy), sin(2πdy), cos(2πdx), sin(2πdx)]
```

`r²=dy²+dx²`，`delta=cap*tanh(dot/cap)`，默认 cap=5，权重为 `exp(-z/2+delta)`。这些权重直接作用于 RAW 值与分母；并非对一个预先融合的 RGB 做空间滤波。

## 协方差恢复

`recovered_covariance.py` 接收已经生成的灰度导引图，恢复 `Sobel×0.25 → 3×3 非归一化 box structure tensor → inverse covariance`。边界为 REFLECT_101。对结构张量，特征值使用原 EPS=1e-12 与退化分支，anisotropy 为：

```
a = sqrt((lambda1-lambda2)/(lambda1+lambda2+EPS))
blend = (1+a)/2
sigma1 = k_detail*(1+(k_stretch-1)*blend)
sigma2 = max(k_detail*(1+(1/k_shrink-1)*blend), min_k)
```

实际矩阵方向、稳定特征向量选择与零张量回退均由原扩展探针确定；不能把其 docstring 中简化的特征值比公式当成实现。默认 k_detail=.12、min_k=.2、k_stretch=4、k_shrink=2。与原入口一样，本实现拒绝改变 min_k 和 k_stretch。

数值 fixture 覆盖 1,664 个 PSD / 旋转 / 各向同性 / 秩一 / 平坦张量以及五组允许参数，共 8,320 次原扩展结果。相同结构张量下，本机比较最大绝对误差为 0。导引图产生结构张量再转协方差时会出现浮点累积误差，不能据此声称所有输入 bitwise 相等。

另有五组原 local frontend 扩展 fixture，覆盖零输入以及 1e-4 到 7 的强度尺度，使用发布包对应的 NumPy / OpenCV 版本验证特征完成、静态 affine 与局部幅度。

**没有确认 RAW→导引图的全部分支，也没有确认 native covariance→HR 输出网格的完整 repeat / crop 规则。** 本模块输出仍位于提供的导引网格；重建入口要求调用者显式提供输出网格 covariance，避免自动填入新的猜测。

## LCA 恢复

`recovered_lca.warp_rgb_tile` 接收完整线性 RGB 和外部提供的 R/B 系数。以 `(W-1)/2,(H-1)/2` 为整图中心，采样坐标为 `center+(destination-center)*(1+coefficient)`。tile 使用整图坐标。绿色不重采样。

原程序有一个重要分支：`applied=False` 返回 ROI copy，保留负值；`applied=True` 即便系数为零也将所有通道裁到非负。48 个原扩展 fixture 覆盖正负值、不同系数、整图 / ROI、边界和单像素维度。PyTorch 实现支持 RGB 与外部系数的梯度。

`RecoveredJSR` 的可选 `lca_coefficients_rb` 会对 learned / legacy 两路使用同一校正，然后重新计算 RefineNet 幅度。集成入口仅允许 origin=0 且输出覆盖完整 reference grid；完整 RGB 上的 tile warp 可单独调用，避免用 ROI 中心代替整图中心。

**自动 LCA 估计器尚未完整复现。** 已辨识部分 bandpass、网格、响应阈值与稳健拟合步骤，但尚未通过无干预 oracle 核验，不把这些片段拼成一个号称作者原实现的估计器。

## 使用接口

```python
from jsr_repro.recovered_model import load_recovered_jsr

model = load_recovered_jsr("Core-Only-Source-Code/weights", k=4)
# frames: [B,K,Hframe,Wframe]，扣黑线性 RAW，保留负值
# flow: [B,K,Hreference,Wreference,2]，reference→frame 的 dx,dy
# covariance: [B,3,Hout,Wout]，xx,xy,yy；坐标单位仍为 native pixel
result = model(frames, flow, covariance, diagnostics=True)
# result["rgb"] / ["learned"] / ["legacy"]: [B,3,Hout,Wout]
```

`fshift[B,K,4]` 为每帧 y offset、x offset、有效 height、有效 width；默认零 offset、完整帧大小。`color_index[Hframe,Wframe]` 默认 RGGB，显式取值 R=0,G=1,B=2。所有坐标均按发布版边界定义，不能直接把旧接口的 frame→reference shifts 填入 flow。

权重加载器验证 manifest 记录的大小与 SHA-256，并恢复 K 路由及静态 affine。加载 K=13 时使用发布的 K14 Controller / RefineNet。运行帧数必须匹配加载路由。全新训练可构造 `RecoveredJSR`；这是已通过梯度检查的计算图，不代表恢复了作者的训练集、损失或优化历史。

此入口要求 Controller 输出网格的高、宽均至少为 8，不采用历史独立 Controller 为极小 ROI 增加的补零扩展。各模块的小型探针仍可直接调用统计 / 融合函数。

CLI 读取不含 batch 维的 NPZ：`frames[K,H,W]`、`flow[K,N,M,2]`、`covariance[3,Ho,Wo]`。可选 fshift、color_index、origin_yx、lca_coefficients_rb。

```bash
python -m jsr_repro.infer_recovered --input boundary.npz --output runs/recovered
```

输出 `stages.npz` 中 RGB 为 HWC，phase / Controller 输入与 qmap 为 CHW；`provenance.json` 记录输入和权重 manifest 指纹及 LCA 状态。示例 boundary fixture 是程序性模块测试输入，不是自然场景或论文测试数据。

## 可复跑验证

```bash
python scripts/extract_released_shaders.py JSR_V9_8.exe --output runs/release-shaders
python scripts/compare_recovered_release.py --release-exe JSR_V9_8.exe --shader-dir runs/release-shaders --output runs/release-parity.json
python -m pytest -q
```

三份数值 fixture 也可以重新生成：另建 Windows x64 Python 3.10 环境，安装 `scripts/requirements-released-oracle.txt`，再运行：

```bash
python scripts/generate_released_oracles.py --release-exe JSR_V9_8.exe --output runs/original-oracles
```

该命令先验证 EXE、三个原 PYD、依赖版本与 affine，随后在该独立进程中导入原扩展。三份输出与本次 fixture 已逐字节核对一致。原扩展只留在选定输出目录的 `_modules/`，不作为仓库交付物。

提取器只输出 shader 到调用者选定的本地目录，仓库不重新打包 EXE、原扩展或 shader。测试 fixture 只含数值输入与原模块输出。

本机 Intel HD Graphics 620 / Vulkan、PyTorch 2.14.1 CPU 的原 shader 对照：

融合参考直接运行原 WGSL，网络参考为公开 WGPU Core 和原 NPZ；此组合测试的公开 Python frontend 在主机 OpenCV 4.14 下执行，原 PYD + 打包依赖的前端另有上述冻结 fixture 核验。此组合不是启动整个 EXE 进行端到端对照。

| 检查 | 最大绝对误差 |
|---|---:|
| counts，K=1/4/14 | 7.63e-6 |
| means，K=1/4/14 | 1.79e-7 |
| standard deviation，K=1/4/14 | 1.31e-4 |
| legacy RGB | 1.50e-7 |
| 固定相同 qmap 的 learned RGB | 1.70e-6 |
| 相同 phase / legacy 下的 Controller 输入 | 3.82e-6 |
| 不同实现逐段传播后的 K4 qmap | 1.45e-4 |
| 逐段传播后的 K4 learned RGB | 1.14e-5 |
| 逐段传播后的 K4 RefineNet RGB | 3.40e-6 |

**保留的失败诊断：** 两个统计实现各自产生 phase 后，归一化 Controller 输入的严格 atol=3e-4 检查最大差 0.00380689，未通过。根因是近零方差 `E[x²]-mean²` 的 FP32 消减，再被 amplitude / frozen std 归一化放大。首次失败报告保留；后来增加相同 phase 的前端检查，分离公式错误与误差传播。没有放宽原有统计 / qmap / 最终 RGB 容差；最终报告使用 `passed_with_intermediate_deviation`，不声称逐位一致或严格中间特征一致。

这些测试使用固定受控输入和相同原权重，不重新训练来掩盖实现差异，也不证明真实照片画质、PSNR 表格或跨设备误差上界。

## v17 版本边界与剩余缺口

另外读取了 [v17.3.1 发布包](https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/releases/tag/jsr-v17.3.1) 的 ZIP 中央目录、README、小型 Fast backend 与一份权重，逐项校验 ZIP CRC。未下载并验证整个 517,284,175-byte ZIP 的 SHA；不把此项列为完整取证。

包内 README 明确区分 Stormer 七帧、Fast 十四帧与 K14 十四帧三个模型；目录中也有三个独立 backend。Stormer 配准已改为全局单应＋高光掩膜四次残差场。这不能用旧版 U-Net 或旧讲稿启发 Transformer 的验证替代。

仍未完成：

- 原始 RAW 导引构造、配准 / 拒帧的完整可训练或独立实现。
- covariance 网格映射、全图 tile / halo 的完整原程序调度验证。
- 自动 LCA 系数估计器的完整恢复。
- v17 Stormer / Fast / K14 各路径的完整识别与数值复现。
- 原作者训练数据、标定划分、loss / schedule 与论文表格。当前作者仓库仍未提供完整论文全文，不能以现有程序测试代替论文实验复现。

以上状态不会通过“可运行”或重新训练来改写成“完整复现”。
