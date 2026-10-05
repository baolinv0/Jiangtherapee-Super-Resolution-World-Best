# JSR 光谱—光学—传感器数据构造

新实验协议：`spectral-camera-v3`，默认 `target_stage: post_pixel`；旧 `spectral-camera-v2` 固定保留 `pre_optics`。这是从用户提供的知乎讲稿与流程图出发、显式修改目标阶段的可运行独立数据方案：RGB→61 波段假设光谱→波长/视场相关光学→相机响应→像元积分→七帧 RAW。七帧 Transformer 的训练、评测、导出和标定入口已接通；v1 Controller/RefineNet 的训练、验证及推理尚未接入这套校准数据适配。**流程层面已经实现；作者原始光谱算法、镜头/PTC 库、图像集合与采样分布没有恢复，不能称为原训练集的数值复现。**

默认 GT 保留参考帧同一套波长/视场光学及原生像元面积响应，2× 仅指采样网格加密；另提供 `post_optics` / `pre_optics` 消融。[目标定义、坐标公式、Google 2019 依据与迁移](GOOGLE_TARGET.md)明确区分论文事实和本工程选择，不声称该目标是 Google 的 GT 公式或理论极限。原 `inferred-synthetic-v1` 数据（用于 `inferred-jsr-v1` 模型）和三波段 `speech-camera-proxy-v1` 保留用于旧实验；不同阶段的 GT 不得混用。七帧网络仍是[讲稿启发的独立 Transformer](TRANSFORMER.md)，没有改变为作者未公开权重。

## v1 Controller/RefineNet 接入改进方案（待实施）

详见 [训练数据实现方案与来源](TRAINING_DATA_IMPLEMENTATION_PROPOSAL.md) 和 [第一阶段实施计划](superpowers/plans/2026-10-05-spectral-jsr-phase1.md)。推荐将同机身实测标定与抽象域扩增分别登记。PMN 的 Sony A7S II ISO100–800 参数可作为有来源的 donor 噪声参考；DeepLens 的审计后专利处方可生成离线 PSF；实测高光谱场景用于验证 RGB 光谱先验。这些资源不构成已测量的完整相机/定焦镜头库。

第一阶段拟新增 `spectral-jsr-v3`（scale2、K4/7/14、等曝光），同步适配训练、训练内验证、评估、档案导出和推理，修正 masked fallback，版本化选择输出组装，并检查校准/资产身份。现有 `spectral-camera-v2` 和旧 checkpoint 默认行为保留。验收重点是整组 burst 及依赖域都缺失某颜色时的最终 RGB 恢复；不能以仍有短曝光同色观测或只统计 GT>1 的指标替代。这些代码调整和专门训练尚未执行。

注意：该拟议的 `spectral-jsr-v3` 面向 v1 Controller/RefineNet；本次已实现的 `spectral-camera-v3` 面向七帧 Transformer 的目标阶段，两者是不同协议与工作范围。


## 原讲稿启发的链路与当前工程选择

| 讲稿机制或工程选择 | 已实现 | 依据与差异 |
|---|---|---|
| 图片上采样到全光谱；图中为 400–700 nm、5 nm | 61 波段非负谱；也可直接读用户提供的光谱辐亮度 | 使用 Mallett–Yuksel 2019 光谱基构造一个与 RGB 对应的假设谱。61 个采样点不意味着从 RGB 恢复了 61 个独立自由度，也不是覆盖全部电磁光谱 |
| CMF 库、不同相机 | 打包 28 台相机的公开实测相对光谱响应；默认选 4 台全画幅相机 | Jiang 等 WACV 2013 数据，不是作者的库。28 台含非全画幅设备，不能声称“覆盖一切全画幅” |
| 波长与像场相关 PSF，f/2–f/8 | 圆孔标量衍射、OPD 像差、径向 LCA、四个视场节点插值；外部光学 PSF 导入 | 默认是物理参数模型，不是 Canon RF135 等真实镜头设计/实测库 |
| 像元 3–5.76 µm、大填充率 | 连续采样像元尺寸，面积填充率 0.9–1；有限像元面积积分 | 范围对应讲稿；具体分布和数值积分精度为工程选择 |
| ISO 100–800 PTC、暗场 | 100/200/400/800 档案；Poisson/读噪声/ADC；实拍平场对与暗场拟合工具 | 默认 PTC 是解析假设。实测输入可替换，随文件记录来源和哈希；没有冒称下载到了作者实测库 |
| 同一 PTC、不同透光率导致通道不同速度饱和 | 同一电子转换/读噪声模型；RGB 独立透光率；满阱与 ADC 两处截断；部分通道饱和训练 | 默认透光率 `[0.7,1,0.55]` 为假设。剩余未饱和观测提供约束，不能保证恢复任意已丢失颜色 |
| 七张，同曝光到 1 档间隔包围曝光 | 7 帧，亚像素平移；间隔 d∈[0,1] EV，曝光比 `[1,2^-3d,2^-2d,2^-d,2^d,2^2d,2^3d]` | 参考帧放首位；无运动物体、曝光期间轨迹或 rolling shutter |
| v3 的目标阶段选择 | 默认 `post_pixel`：参考帧 PSF → SRF → 原生面积积分 → 2× 密集采样；显式支持上游两个阶段 | 本工程修改，不推定作者意图。v2 的 `pre_optics` 历史行为保留；源照片自带的光学/ISP 痕迹在所有阶段仍可能存在 |

## 实际构造顺序和单位

1. **先划分源场景，再裁 patch。** Manifest 记录源文件 SHA256、scene、encoding、split。相同内容与同场景不得跨划分；训练、评测、导出检查哈希。训练 seed 由总 seed、epoch、样本序号和源哈希确定；scene/optics/noise 使用独立 RNG，保持 v2 随机序列；改变目标阶段不会改变 RAW，改变 PSF 是否影响 GT 由目标阶段决定。
2. **线性化和光谱提升。** sRGB 使用逆传递函数；线性 RGB 直接输入。Mallett 非负基乘 RGB，再乘 D65 形成假设辐亮度。原基 380–780 nm 被裁到 400–700 nm，使用梯形积分。当前原色/白/灰探针往返 sRGB 最大绝对误差 0.003279、RMSE 0.001454；没有强行宣称裁带之后仍精确色度匹配。逆 sRGB 不会撤销原相机的 tone mapping、锐化或降噪。
3. **相机颜色与目标阶段。** 相机曲线由 10 nm 插值到 5 nm，积分权重按该通道的 D65 白归一化。GT 是这套 **D65 归一化 camera RGB**，不是显示 sRGB，也不是原图 RGB。源谱乘 log-uniform 场景增益（默认 0.125–8），GT 允许超过 1。v3 默认目标在参考帧光学及原生像元积分后分出；`post_optics` 只保留光学；`pre_optics`（也是固定 v2 行为）在新增光学之前分出。三个阶段均不含 CFA、逐帧曝光、透光率、噪声、量化或截断。
4. **逐帧运动和光学。** 先对光谱场景做平移，再应用固定在传感器坐标的光学 PSF；避免把视场像差随物体错误地一起移动。圆孔复振幅含 defocus、astigmatism、coma 的未归一化 Zernike 项，系数单位 nm OPD；FFT 标度由波长、光圈和空间采样共同决定。四角 PSF 双线性混合是视变卷积的近似；LCA 为另设径向位移。
5. **投影、像元积分、CFA。** 光学处理后的谱按相机响应积分为 RGB；对方形有效感光面积做 4×4 求积，再按 RGGB 取样。默认 `post_pixel` GT 复用参考帧的光学 RGB 场，以更密中心执行同一次原生面积积分，不含 CFA。原生中心 `m+2j+0.5`、dense 中心 `m+j`；面积边长均为 `2√fill_factor` 个 HR 单元，不能随 2× 输出缩半。面积填充率控制边长的平方。光学核已在 HR 重建网格单元上求积，但尚未包含原生像元面积积分，两者不可混为一谈。
6. **曝光与电子噪声。** 设相机积分结果为 x，曝光比 e，通道透光率 t，电子/DN 为 g，则预期电子数 `μ=x·e·t·(white_dn−black_dn)·g`。先 Poisson(μ)，再按 full_well_e 截断，加读噪声，除以 g 加黑电平，再 ADC 四舍五入并截到 `[0,white_dn]`。顺序保留“噪声可能把接近满阱的观测推入饱和”的行为。
7. **生成网络条件。** RAW 扣黑归一化，`saturation` 根据观测 DN 判定；`variance` 是观测信号的 plug-in 噪声估计。干净电子数和真实信号饱和标记仅作诊断，模型不读取它们。曝光和透光率用于把观测校正回 GT 坐标。

**ISO 约定：** x=1 对应每个 ISO 档位的参考白，因此不同 ISO 的单位参考电子数可能不同。本实现不是固定绝对光子通量、固定快门后只改变模拟增益的 ISO 扫描；相关约定和 `electrons_per_reference_unit` 写进每个样本。相机曲线为相对响应，不能从其推导绝对 QE。默认独立抽取响应曲线、PTC、像元和像差，是组合域随机化，不是特定真实机身—镜头的完整标定。

原生 RAW 张量为 `[7,1,H,W]`，去掉单通道轴后的马赛克为 `[7,H,W]`；packed RAW 为 `[7,4,H/2,W/2]`，GT 为 `[3,2H,2W]`。每条样本另存协议/目标阶段及采样语义、7 组 native-pixel 位移、曝光、观测噪声/饱和、ISO、光圈、像元、像差、光谱/标定哈希、源哈希等。光学支持与运动之外保留固定 `scene_margin_hr`；图像边缘复制和源图放大都会标记。`field_extent` 表示含 margin 的完整模拟区域半宽，不表示一块 patch 固定占据真实相机的多少视场。

## 使用公开 RGB 或自有光谱数据

安装并执行完整小规模验证（无需下载自然图像集）：

```bash
python -m pip install -e ".[test]"
python -m jsr_repro.validate_spectral --output runs/spectral-v3-validation
# 或 scripts/verify_spectral.sh / scripts/verify_spectral.ps1
```

准备自行从[官方站](https://data.vision.ee.ethz.ch/cvl/DIV2K/)取得的 DIV2K HR 图像：

```bash
python -m jsr_repro.prepare --source /datasets/DIV2K --layout div2k --encoding srgb --output data/train
python -m jsr_repro.spectral_data --config configs/spectral_k7.yaml --split train --count 100 --output data/generated
python -m jsr_repro.train_transformer --config configs/spectral_k7.yaml
python -m jsr_repro.evaluate_transformer --checkpoint runs/spectral-k7/last.pt --manifest data/train/manifest.jsonl --alignment oracle --output runs/spectral-test-oracle.json
python -m jsr_repro.evaluate_transformer --checkpoint runs/spectral-k7/last.pt --manifest data/train/manifest.jsonl --alignment estimated --output runs/spectral-test-estimated.json
```

DIV2K 0001–0800 为训练；官方 validation 的 0801–0850/0851–0900 分成本工程 val/test，**不是官方隐藏 test**。自有图片用 `--layout custom`，同场景多视角用 `--group-by parent`。不自动下载和重新分发图像集。离线导出产生 NPZ 和含哈希的 JSON manifest；在线训练直接从源 manifest 调用同一生成器。`--count` 最大为该 split 的场景数乘 `samples_per_scene`，实际输出数会报告。

光谱输入为安全 NPZ：`radiance[L,H,W]`（非负）、`wavelengths_nm[L]`（严格递增，必须覆盖 400–700 nm）。不外推缺失波段，也不根据扩展名认证其为实测数据。每个 scene 至少独立划分，示例：

```bash
python -m jsr_repro.prepare --source /datasets/spectra --encoding spectral_radiance --output data/train
```

默认训练配置为 native64、K7、100000 步、GPU；这是可运行配置，**没有完成自然图像全集训练**。历史 v2 验证执行了小模型 CPU 8 步及测试中的恢复训练；v3 实测结果单列在 [google-target-validation/VALIDATION.md](google-target-validation/VALIDATION.md)。61 波段逐帧光学在 CPU 构造，成本高于三波段代理；没有宣称生产训练吞吐或作者速度。

## 接入实测 PTC 与暗场

使用同一 ISO、温度、读出模式和 ROI 的 RAW DN 数据。每个亮度等级拍两张平场；暗场至少 3 张，实际建议更多。输入未做白平衡/伽马的浮点数组，保留黑电平：

```json
{
  "name": "my-camera-readout",
  "provenance": "camera/readout/temperature/exposure and capture notes",
  "channel_transmission": [0.7, 1.0, 0.55],
  "captures": [{
    "iso": 100,
    "flat_pairs": "flats-100.npy",
    "dark": "dark-100.npy",
    "white_dn": 16383,
    "full_well_e": 60000,
    "full_well_source": "replace with measured or independently documented value"
  }]
}
```

以上数值仅为格式示例，不能作为实测结果。`flat_pairs` 为 `[levels,2,H,W]`、`dark` 为 `[T,H,W]`。工具用平场对差分方差的一半消除静态空间非均匀性，取扣黑信号的 5%–70% 区间且排除达到白电平的等级，至少保留 3 级。由暗场时间方差固定截距（假设均匀量化，最低 1/12 DN²），以预测方差平方的倒数作权重拟合正斜率，得到电子/DN；保留自由回归截距与残差诊断，避免因有限样本的负外推截距误拒绝有效标定。权重假设各亮度等级采样数相同。full well 由用户提供，不能由线性 PTC 段唯一推定。

```bash
python -m jsr_repro.calibrate_stacks --manifest calibration/captures.json --output calibration/fitted
```

将 YAML `data.profile` 设为生成的 `profile.json`；只抽样实际存在的 ISO。可选 `data.options.read_noise_bank` 指向 `read-bank-iso100.npz`，此时 `iso: [100]`。暗场残差去除逐像素时间均值，校正有限帧方差并近似扣除 ADC 的 1/12 DN²；保留其空间相关性，替换 Gaussian 读噪声，不额外再叠加一次。bank 强制检查电子单位、ISO 和完整 profile 哈希。此处理不会保留 DSNU，没有显式模拟随曝光时间变化的暗电流，低码值离散噪声的量化方差校正只是近似。

## 接入外部光学 PSF

NPZ 包含 `kernels[nodes,61,2r+1,2r+1]`、`field_xy[nodes,2]`、`wavelengths_nm[61]` 和标量字符串 `metadata`。元数据 JSON 必须含：

```json
{"schema":"jsr-spectral-psf-v1","spatial_unit":"um","sampling_um":2.0,"f_number":4.0,"contains_pixel_integration":false,"provenance":"actual source and processing"}
```

示例 sampling_um=2 要求 pitch_um 固定为 4（HR/native 倍率 2）。设置 `psf_library`，固定匹配的 f-number、pitch、field_center、field_extent 和 psf_radius。单节点用于 extent=0；四节点必须按左上、右上、左下、右下与配置逐值匹配。代码拒绝尺寸/单位不符或已包含像元积分的核，不静默重采样。真实机身—镜头组合应同时固定 camera_id、PTC 和对应光学库。

## 相关引用的核验与积分边界

此前版本引用 [RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](https://arxiv.org/html/2603.12493v1)。本次调研未能独立取得该全文及实现，不把其具体方法作为已核验的实施依据。已核验的光学实现、实测核方法和来源范围见 [光学调研](training-data-research/optics/notes.md)。

[Delbracio 的作者说明](https://github.com/mdelbra/psf-estim)明确指出其估计的相机模糊包含传感器及抗混叠滤镜等效应。这样的有效相机核不能直接当本接口的纯光学核后重复做像元积分。纯光学核与有效相机核应使用各自清晰的协议，并在相同测量域内验证；这些资料不能补出作者 JSR 的原始光谱或 PTC 文件。

## 公开资产、出处与许可

| 资产/方法 | 一手来源 | 本工程使用及许可 |
|---|---|---|
| Mallett & Yuksel 2019 | [作者项目](https://graphics.geometrian.com/research/spectral-primaries.html)、[论文](https://www.cemyuksel.com/research/papers/spectral_primary_decomposition.pdf)；[Colour v0.4.6 数值实现](https://github.com/colour-science/colour/blob/v0.4.6/colour/recovery/datasets/mallett2019.py) | 数值基取自固定版本；BSD-3-Clause 原文随附 |
| Jiang 等 2013，相机相对响应 | [作者数据库及数据](https://www.gujinwei.org/research/camspec/db.html) | 28 条目，插值处理；CC BY-NC-SA 4.0 原文随附，商业用途需另行取得适当权利 |
| CIE 1931 2°、D65 | [CMF 官方表](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer)、[D65 官方表](https://cie.co.at/datatable/cie-standard-illuminant-d65) | 裁带与积分，CC BY-SA 4.0；完整官方归属/许可元数据随附 |
| PTC/噪声物理 | [EMVA 1288 Linear 4.0](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf) | 用于独立拟合与单位约定，不表示经过 EMVA 认证 |
| Fourier pupil / OPD | [POPPY 官方说明](https://poppy-optics.readthedocs.io/en/latest/overview.html)、[波前误差说明](https://poppy-optics.readthedocs.io/en/latest/wfe.html) | 原理参考；本工程独立代码，不依赖 POPPY 或其镜头库 |

`spectral_assets/source_manifest.json` 保存下载 URL、字节数和 SHA256。复建资产：`python scripts/fetch_spectral_sources.py --output /tmp/jsr-sources`，再 `python scripts/build_spectral_assets.py --source /tmp/jsr-sources`（具体可用参数见 `--help`）。下载脚本校验固定内容；构建脚本只用 AST 字面量解析公开基数据，不执行下载代码。生成的 basis/cameras 哈希进入 checkpoint；恢复训练/评测拒绝资产被悄悄替换。

## 验证及仍未恢复的内容

**下列数值为历史 v2 / `pre_optics` 证据，不适用于 v3 默认目标。** 当时全部 **63 项自动化测试通过**；实际 8 步验证和图见 [spectral-validation](spectral-validation/)。历史光圈、像元、填充率、PSF 支持和视场干预中，v2 GT 最大绝对变化均为 0，RAW 相应改变；测试覆盖非负谱、单位平场、PTC 拟合、shot/read 方差、部分通道饱和、真实光谱导入格式、严格恢复、资产变更拒绝、NPZ→TIFF。7 个资产文件重建后逐字节一致，wheel 构建及资产/许可内容检查通过。

历史 v2 的 8 步 procedural test：oracle 条件下 corrected merge 16.858 dB，Transformer 16.854 dB；估计对齐时分别 16.749、16.745 dB。峰值固定 1，GT 可超过 1，没有每图拟合增益。这些结果只验证链路，短训网络未超过融合基线，不能当作作者画质或自然图像基准。

v3 检查阶段对应的 GT 变化、同 RAW 三阶段消融、固定原生面积、观测因素不进入 GT、旧 v2 恢复和跨阶段拒绝；运行命令、反例及设计见 [GOOGLE_TARGET.md](GOOGLE_TARGET.md)，新增证据单列记录，历史 JSON 不改写。

后续 [独立 review 与可运行参考](GOOGLE_TARGET_REVIEW.md) 量化了运动重采样、有限 PSF 支持和读噪声后饱和漏判的限制，并给出等曝光整组缺色的验收方案。参考算子尚未接入正式生成器。

尚缺作者原始图像清单、RGB→谱算法、全部相机 PTC/暗场、实测/设计镜头 PSF、各参数联合分布与原始训练。默认只含静态场景帧间平移，没有曝光期间运动积分、动态物体、rolling shutter、lens shading、完整 sensor crosstalk 或反 ISP。场景仅在原生 2× 网格离散，亚像素运动和积分依赖 bilinear 插值，不是连续光学真值。有限 FFT/PSF 支持会截掉衍射尾部后重新归一化，四节点视场近似和固定求积也有误差；需按目标空间分辨率加密、扩大支持并做收敛评估。后续取得真实资产可通过上述接口替换，不需要重写数据链。


Review revisions add optional robust dense alignment, shared v1 LCA, explicit temporal capture ranks, spherical spectral OPD and reproducible diagnostics. See [conventions, provenance, conformance and evidence](REVIEW_REVISION.md) for the independent definitions, default compatibility and real RAW NOT_RUN protocol.
