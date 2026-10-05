# JSR 训练数据构造：公开资源与可实施方案

调研及方案更新日期：2026-10-05（Asia/Shanghai）。审查分支：`reproduce/jsr-pytorch-engineering`。原调研基线为历史提交 `0e764de8bb026b1e08da3655d8d741e299018108`；本轮目标审查基线为 `0bc29f14dd0e4be8a05cfc5787719203fe43d900`。

**建议采用“实测标定作基准，公开参数与物理模型扩展覆盖范围”的方案。** 现有分支已提供可运行的光谱—光学—传感器骨架，可以继续使用。需要补齐的是资产的物理对应关系、真实标定、原版 JSR 训练入口，以及证明部分通道饱和恢复的评估。公开资源足以支持一个有来源、可验证的独立实现；本次没有取得作者的原始 PTC/镜头库，不能据此声称复现了其全部训练分布。

本文区分已核验的公开事实与建议实施的工作。原调研只更新文档；此后代码已增加 `spectral-camera-v3` 的三阶段目标，默认 `post_pixel`，见 [GOOGLE_TARGET.md](GOOGLE_TARGET.md) 和[本轮审查](GOOGLE_TARGET_REVIEW.md)。本次同步这项独立工程选择，并保留历史 v2 语义。以下采集数量、采样网格、阈值和 v1 接入安排仍为工程建议；真实拍摄、完整训练及未落实的代码改造均待执行。作者实际 GT 的物理阶段没有因此得到确定。

## 1. 对作者描述的逐项落实

| 作者描述 | 当前分支 | 建议实现 | 达到什么条件才可以称完整 |
| --- | --- | --- | --- |
| 图片上采样到全光谱 | Mallett–Yuksel 基函数生成 400–700 nm、5 nm 间隔的 61 波段假定光谱；也可导入外部光谱 | 保留快速 RGB 光谱先验，加入实测高光谱场景；显式记录来源、照明和波长轴 | 明确“全光谱”指覆盖的可见光范围；不能把 RGB 预测或插值称为真实光谱恢复 |
| 相机链路下采样及 PSF | 单色衍射/Zernike OPD、视场混合、像元积分、RGGB 和噪声均已串联 | 加入审计后的镜头处方 PSF、实测相机核验证；像元积分只执行一次 | 波长、视场、光圈、焦距/对焦距离、采样单位与真实镜头对应，并有留出状态验证 |
| 全画幅 ISO 100–800 PTC 库 | 默认一个解析电子/DN profile；已有 flat/dark 拟合工具 | PMN/ELD 的公开参数作为明确标注的参考，再采集同机身 RAW flats/darks | 每个声称覆盖的机身、读出模式、ISO 都有测量或注明推断的条目、拟合诊断和留出验证 |
| 主流定焦 f/2–f/8 | 连续随机光圈配通用像差；不是实测定焦镜头库 | DeepLens/DiffOptics 的专利处方用于物理扩增，小规模实测用于锚定 | 只用镜头实际可达到的光圈，验证像场与色散；公布镜头/光圈覆盖表 |
| 像元 3–5.76 µm、高填充率 | 独立随机 pitch 及 0.9–1.0 填充率 | 分开“有物理对应的具名机身”和“抽象参数扩增”两种采样 | 具名相机的 pitch、响应、噪声模式一致；微透镜/有效感光孔径假设明确 |
| 同 PTC、不同透光率、通道依次饱和 | 已实现通道透光率影响电子数、共享 PTC、满阱/ADC 截断 | 先验证各 CFA 电子增益能否共用；透光率与光谱响应归一化一致 | 在整组连拍都缺失某颜色的样本上，最终 RGB 相比基线确实改善 |

代码依据：[spectral.py](../reproduction/jsr_repro/spectral.py#L1)、[spectral_data.py](../reproduction/jsr_repro/spectral_data.py#L53)、[calibration.py](../reproduction/jsr_repro/calibration.py#L14)、[physical_sensor.py](../reproduction/jsr_repro/physical_sensor.py#L54)。28 条相机光谱响应曲线是相对波长响应，不能替代 28 台相机的 PTC、绝对 QE 或满阱测量。

## 2. 可以实际利用的公开来源

| 来源 | 本次核验内容 | 适合承担的工作 | 需要保留的限制 |
| --- | --- | --- | --- |
| [Mallett–Yuksel 2019 / Colour](https://github.com/colour-science/colour/blob/v0.4.6/colour/recovery/mallett2019.py) | 固定版本基函数及转换实现；论文 *Spectral Primary Decomposition for Rendering with sRGB Reflectance*，DOI 10.2312/SR.20191216 | 大量 RGB 场景的快速、非负光谱先验 | 一个 RGB 对应许多同色异谱；目前仓库截取 400–700 nm，原始基函数范围更宽 |
| [MST++ / NTIRE 2022](https://github.com/caiyuanhao1998/MST-plus-plus) | 作者 README 提供成对光谱/RGB 数据入口，代码读取 `.mat` 光谱 cube | 实测高光谱场景补充；可选离线学习型光谱先验 | 数据包没有下载；必须验证波长、量纲、照明和许可；学习结果仍为预测 |
| [Google Research Unprocessing](https://github.com/google-research/google-research/tree/master/unprocessing) | CVPR 2019 官方说明和 `unprocess.py`，近似逆 tone/gamma/CCM/WB | JPEG 数据的近似相机线性化，作为扩增而非干净真值 | 逆 gamma 不能撤销全部 ISP、剪裁、局部色调映射及原图光学模糊 |
| [PMN](https://github.com/megvii-research/PMN) | Sony A7S II 逐 ISO 噪声参数，包含 100/200/400/800 和中间档位 | 第一阶段有来源的电子增益/读噪声参考 | 是另一台实体机身的标定；缺少匹配的物理满阱、透光率和原始逐 ISO PTC 栈 |
| [ELD](https://github.com/Vandermode/ELD) | Canon 5D IV、Nikon D850、Sony A7S II 等小型拟合参数资产 | 扩展具名参考噪声分布和重尾/行噪声思路 | 公开参数没有 ISO 索引表；评估数据为 ISO 800/1600/3200；公开基线与受限制的完整模型须区分 |
| [EMVA 1288](https://github.com/EMVA1288/emva1288) | 官方参考实现的增益、噪声、线性度等计算；[真实参考数据](https://github.com/EMVA1288/datasets)的元数据 | 自建 RAW PTC 工具及测量流程的参考 | 参考数据是机器视觉 CCD/CMOS，不是全画幅消费相机库；有限拟合不能称符合整套标准 |
| [DeepLens](https://github.com/vccimaging/DeepLens)、[DiffOptics](https://github.com/vccimaging/DiffOptics) | 官方光线追迹、相干传播/衍射实现和若干公开定焦处方 | 离线物理 PSF 库、色散/视场模拟、参数拟合 | 专利例子不等于量产镜头实测；默认几何 PSF 不包含完整衍射 |
| [Eboli 实测镜头 PSF](https://github.com/teboli/fast_two_stage_psf_correction) | 作者提供几十款镜头实测 PSF 入口；解析代码有 Canon 5DSR 与 EF24 mm 文件示例 | DSLR 的有效相机核验证或独立扩增路线 | 外部数据未取得；RGB 有效核不能直接充当 61 波段、未积分的纯光学核 |

本次成功访问并保存的是 GitHub 作者仓库的说明、源文件和小型参数资产。EMVA 官网 PDF、PhotonsToPhotos、部分论文站及外部网盘被当前代理返回 403；这不代表它们不存在，但本报告不声称已经读取那些全文或下载大数据包。代码许可与第三方数据/镜头处方许可应分别记录；已核验软件许可包括 Colour BSD、PMN/DeepLens Apache-2.0、ELD/DiffOptics/Eboli MIT，外部数据许可并未因此自动得到确认。

### 2.1 最直接可用的 ISO 参考值

PMN 固定提交 `d207d3eb62e3a5106861992c0562054adf9a9c70` 的 [process.py](https://raw.githubusercontent.com/megvii-research/PMN/d207d3eb62e3a5106861992c0562054adf9a9c70/data_process/process.py) 中，`get_camera_noisy_params_max` 提供：

| ISO | K：DN/e | 转为仓库 g：e/DN | Gaussian 读噪声 σ：DN |
| --- | ---: | ---: | ---: |
| 100 | 0.09563 | 10.456970 | 1.0067395 |
| 200 | 0.19126 | 5.228485 | 1.2926387 |
| 400 | 0.38252 | 2.614242 | 2.0595572 |
| 800 | 0.76504 | 1.307121 | 3.5475867 |

换算 `g = 1/K`，Gaussian 分量的电子域 σ 为 `σ_DN / K`。PMN 还给出行噪声与 Tukey-lambda 参数，不能把 Tukey scale 直接当标准差，也不能把整套模型压成一个 Gaussian 后声称完全等价。源表所有增益遵循 `K = 0.0009563 × ISO`，因此“有逐 ISO 条目”不能证明每档都独立测量过 PTC 斜率。

[PMN README](https://raw.githubusercontent.com/megvii-research/PMN/d207d3eb62e3a5106861992c0562054adf9a9c70/README.md) 明确写明标定机身与公共数据机身传感器相同、实体相机不同。参考系数可以替换无来源的解析噪声，但如果使用另一机型的光谱曲线或随机 pitch，这仍是混合扩增，不能标为“实测 A7S II 相机”。参考的 wp=16383、bl=512 也不能自动推广到所有 RAW 模式。实际数值快照保存在 [pmn_sony_selected_iso_coefficients.json](training-data-research/sensor/assets/pmn_sony_selected_iso_coefficients.json)。

## 3. 推荐的数据生成与标定路线

### 3.1 光谱与源图像

主训练源可继续采用当前快速基函数，但优先加入线性 HDR/高质量线性 RGB，减少 JPEG 既有锐化、去噪、局部色调映射被当作干净真值的影响。原始源图的光学模糊仍在；仅 `pre_optics` 消融的 GT 排除新模拟 PSF，也不意味着已经获得真实无模糊场景。

加入 NTIRE/ARAD 等实测高光谱子集，用于约束光谱分布和留出测试。读取真实波长轴后重采样到当前 61 波段；插值增加数值采样点，不增加真实测量的信息。分清输入是反射率还是辐亮度：反射率需要指定照明，已包含照明的辐亮度不能再乘 D65。RGB 先验及高光谱场景都应记录照明/曝光尺度。若扩展到 400–700 nm 之外，需要同时扩展基函数、响应、PSF 和积分轴，不能只补零后宣称扩展完成。

不建议先把训练链路升级成另一套大型 RGB→HSI 网络。MST++ 可以离线生成不同先验作为对照，但重要的第一步是加入真实光谱评估，并测量先验偏差会如何影响跨相机、跨光源的恢复。

### 3.2 相机/PTC 标定

先选少量代表机身和 RAW 模式，ISO 100/200/400/800 为首批锚点；若要声称覆盖全部可选 ISO，再加入 125/160/250/320/500/640 等。记录机身编号、固件、位深、压缩/读出模式、温度、曝光时间与 RAW 哈希。中间档位的插值只在经过验证的同一增益/读出模式内进行，不应跨转换增益切换强制按 ISO 成比例。具名相机固定其真实 pitch 和响应关联；3–5.76 µm 连续范围另外作为抽象扩增。

建议每 ISO 采约 20 个均匀照明水平，每水平 4–8 张 RAW，另采约 64 张短曝光 dark 和代表性的长曝光 dark。数量是初始建议，应按拟合不确定度调整。采用稳定无频闪均匀光源，避免白平衡、去噪、demosaic、tone curve 和有损 RAW。覆盖线性区，并另做接近/超过饱和的扫描；只测线性区不能识别物理满阱。

每个 CFA 平面分别计算 dark-subtracted mean μ 和 `Var(flat1−flat2)/2`，拟合：

`v_DN² = K_DN/e × μ_DN + v_dark`，`g_e/DN = 1/K_DN/e`。

验证 R/G1/G2/B 的斜率和残差是否支持共用一套电子规律，再合并成作者所述“相同 PTC”。不同通道的滤光片/光谱响应改变收集电子数；电子增益相同并不意味着相同入射光必然同时饱和。透光率必须有归一化/照明定义，不能仅从不同颜色的噪声拟合中推断。

**满阱与 ADC 白电平必须分开。** ADC 电子余量为 `g × (white_dn−black_dn)`，可用电子上限近似为它与 `full_well_e` 的较小值。高 ISO 可能先碰到 ADC 上限。缺少物理满阱测量时，资产应记录未知或已知下界；若原型采用 ADC-only 或解析满阱假设，必须明确该模型范围。现有 profile 强制填写 full_well，直接塞入一个猜测值会掩盖这个缺口。

先复用 [calibrate_stacks.py](../reproduction/jsr_repro/calibrate_stacks.py)，导出配对 flats/dark manifest 和拟合诊断。随后按实测需要扩展 per-CFA、读出模式、未知满阱、行/列噪声、PRNU/DSNU 及字段级来源。当前 residual dark bank 去除逐像素均值，可保留部分时空噪声，但不包含被去除的 DSNU；PMN dark-shading 系数图也不是随机 dark-frame bank。定量比较应使用留出的曝光/拍摄会话，检查均值—方差曲线、低信号尾部、协方差和饱和位置。

### 3.3 定焦镜头 PSF

优先采用 DeepLens 的少量、逐一审计的公开处方，离线生成 PSF，再用真实 body/lens 组合校准。光线追迹提供场曲、像差、色散、畸变；相干传播/衍射负责几何光线不能覆盖的衍射效应。默认几何 PSF 不能当完整 ray-wave 结果。

必须核查处方单位、有效焦距、入口瞳、实际 f-number、像场范围与玻璃色散。已核验的两个反例说明不能按文件名批量导入：`rf24mm_f1.8.json` 存储约 39.54 mm、f/5.24；`rf16mm_f2.8.json` 明确注明原专利不能覆盖全画幅。EF35 标名 f/2 的例子存储约 f/2.31。它们是文件元数据观察，尚未运行光学重算；不能由此断言项目算法错误。EF50 的公开专利例子约 50.52 mm、f/1.79，较适合作为需验证的起点。

先从稀疏视场、波长和可达到的光圈开始：例如 3×3 或 5×5 视场、f/2、2.8、4、5.6、8 中该镜头支持的档位。物距和对焦距离分开记录。选少量 ray-wave 参考点，拟合当前 OPD/FFT 模型或插值库，再根据留出误差补点。三张 RGB PSF 不足以证明 61 波段准确；各波长的质心要相对同一参考保留，分别居中会抹掉横向色差。

DeepLens 已核验的相干路径要求 float64、至少 100 万光线，默认约 1678 万。25 视场×61 波段×5 光圈等于每镜头/物距 **7625 次** PSF 调用，最低也超过 76 亿条光线，尚未计算镜片表面传播。因此建议离线 GPU 预计算与验证后的插值，当前 CPU 环境只做几何粗网格和极小相干参考检查。此处没有实际速度基准，不能承诺生成耗时。

实测相机核通常含像元孔径、OLPF、盖玻璃、有限光源以及可能的 ISP。当前 [physical_optics.py](../reproduction/jsr_repro/physical_optics.py) 外部库要求不含 pixel integration。可选两条清晰路线：通过包含像元/光源的测量模型拟合纯光学参数；或新增有效相机核协议，明确不再重复同一像元积分。不能把实测有效核的 integration 标记改为 false 来绕过检查。

PSF 库至少保存波长/光谱带宽、物理采样、视场、光圈、焦点/物距、来源及版本、共同质心参考、归一化前通量、核支持截断能量、像元/光源积分状态和误差范围。归一化的 PSF 只表示形状；渐晕、T-stop/总透过率独立保存。建议用 Airy 解析极限、双倍数值采样与独立种子做收敛检查，再比较留出实测核的质心、包围能量及二维 OTF/MTF。仅 MTF 幅值不能唯一确定二维 PSF。

### 3.4 统一物理单位与资产身份

推荐顺序为：

`线性场景/光谱 → 帧运动映射 → 各波长视场 PSF → 光谱响应及一次像元积分 → RGGB → 透光率/曝光形成期望电子数 → Poisson → 满阱 → 读出噪声 → DN/量化/ADC → 观测掩码`。

空间均匀的线性光谱响应与像元积分可交换；条件不满足时必须按真实传感器空间响应实施。GT 按现有 `spectral-camera-v3` 的 `target_stage` 解释：默认 `post_pixel`，取参考帧同一光谱 PSF/SRF 作用后、保留原生像元 footprint 的密集 2× 采样；`post_optics` 排除像元面积响应，`pre_optics` 才排除新增光学退化。三者均在参考曝光的 camera RGB 中，允许大于参考白，不含 CFA、观测透光率、噪声或剪裁。f-stop/PSF/pitch/fill 改变时 GT 全部不变的断言只适用于 `pre_optics`；`post_optics` 可随光学变化但不随 fill 变化，默认 `post_pixel` 可同时随光学和 fill 变化。三阶段分别训练评估，不能合并成一个“清晰 GT”任务。

当前阶段切换沿用历史 v2 的观测 RNG namespace；在其余选项、资产和观测算子相同的条件下，共享 RAW 以作目标消融。**待做：** 评估 coordinate-first 前向计算，避免低分辨率场景先 bilinear warp 再积分产生额外平滑；同时检查视场 PSF 固定在传感器坐标的语义，不能假定运动和空间变化光学可交换。应做场景网格、FFT/瞳孔、面积求积及视场插值的数值收敛，并记录 PSF 归一化前的有限支持能量损失。未来改变这些观测算子时应新增独立 `forward_model` 身份；目标阶段与前向模型分开记录，不承诺新旧 RAW 逐值相同。这些改造尚未实施。

当前响应按每通道 D65 白归一化，再独立使用 transmission，因此额外透光率有明确位置。如果改用绝对 QE、含 CFA 透过率的响应，需重新定义归一化，避免重复计入同一透过率。光圈改变 PSF 形状不自动改变曝光：物理通量模式需要加入 T-stop、曝光时间、像元面积等；沿用当前每 ISO 白电平归一化模式也可以，但不能把它报告成固定光子通量的 ISO 扫描。

资产 registry 分成两类：

- **具名标定组合**：机身/模式/ISO/PTC/black-white/pitch/光谱响应/透光率/镜头与状态相互绑定；缺失字段显式标注测量、公开拟合、推断或未知。
- **抽象扩增组合**：连续 pitch、填充率、OPD、donor 噪声等用于覆盖未见域；记录混合来源，不声称对应某台量产相机。

登记内容哈希、来源 URL/commit、单位、测量/推断状态、适用域和校准不确定度。训练、评估、推理及 checkpoint 使用同一身份；尺寸相同不能说明资产或训练协议相同。

## 4. 接入当前 JSR 的必要调整

本节是尚未实施的 v1 JSR 接入建议；现有 Transformer 的 camera-v3 目标定义已实现，不能据此声称 v1 adapter 也已完成。保持 151→36 Controller、六通道 RefineNet 拓扑可行；但只替换 Dataset 不能满足任务。

| 位置 | 建议 | 原因 |
| --- | --- | --- |
| [train.py](../reproduction/jsr_repro/train.py#L70)、[evaluate.py](../reproduction/jsr_repro/evaluate.py#L29) | 小范围复用 `make_burst_dataset`，同一适配器覆盖训练、训练内验证、独立评估、推理和验证档案导出 | 当前 v1 直接使用 `SyntheticBurstDataset`；光谱链路只接入 Transformer；factory 的原默认是 Transformer 的 proxy 协议 |
| [spectral_data.py](../reproduction/jsr_repro/spectral_data.py#L56) | 新协议支持 K4/7/14、等曝光；保留旧 K7 多曝光协议 | 当前硬编码 Transformer K7；v1 初期不应混入未适配的曝光差异 |
| [model.py](../reproduction/jsr_repro/model.py#L131) | 在原有 positional 参数之后增加可选 validity，传给 `phase_splat` | 底层已有 mask 支持，模型入口未传递；保留旧调用兼容 |
| [frontend.py](../reproduction/jsr_repro/frontend.py#L169) | 对 masked 路径计算只含有效样本的 per-color fallback，完全无支持时明确处理 | 当前 fallback 平均全部 RAW，饱和值会漏回已经清空的通道 |
| RAW adapter / [infer.py](../reproduction/jsr_repro/infer.py#L24) | 在观测 DN 域生成饱和 mask，再按 CFA 透光率校正；新档案要求校准元数据 | 校正后值 >1 仍可能有效；推理目前只读 raw/shifts，不应用校准 |
| [model.py](../reproduction/jsr_repro/model.py#L86) | 对新训练协议显式选择未截断的线性输出组装，旧权重保留原 policy | 现有 normalized RGB 的 `[0,1]` 截断限制缺失颜色可恢复的幅值 |
| checkpoint identity | 给新 recipe 增加光谱、PTC、PSF、dark bank 哈希及 output policy；明确推理不匹配时的默认拒绝或显式跨域策略 | 避免旧权重/新物理协议混用；Transformer 的现有推理仅报告 domain match，不会因不匹配自动拒绝 |

v1 未填写 protocol 时仍应选择原 `inferred-synthetic-v1`，Transformer 保留其原缺省；不能直接复用 factory 的 proxy 默认而改变旧配置。当前 [validate.py](../reproduction/jsr_repro/validate.py#L39) 导出的旧 NPZ 只含 RAW/shifts/target，新协议必须同步导出 mask、透光率、曝光及校准/资产身份，使验证和推理使用一致的解释。

例如中性 GT=1.4、transmission=`[0.5,1,0.25]`、等曝光：观测 red=0.7、green 已饱和、blue=0.35；校正后 red/blue 都为 1.4，仍须保留为有效观测。不能用校正 RAW>1 检测饱和，也不能默认 RAW>0 为有效性条件：丢弃负的 black-subtracted 读噪声会给暗部引入偏差。真值饱和状态只用于离线诊断，模型 mask 应来自观测和已知标定。

输出截断还有一个可计算的接口范围限制：RefineNet 当前每颜色输出上界为 `amplitude / 0.34523068818`。若缺失 green 被置零、red/blue 各 0.05，则 amplitude≈0.040825，上界≈0.1183。接口目标 green=8 时，无论训练多久都不可达。这个反例用于检查模型的数值可表示范围，不代表已经验证某特定实测光谱能产生该组合。若继续保留截断，必须说明并验证目标分布落在可表示范围；若要支持更宽 HDR 恢复，则版本化修改输出组装，重新训练/微调，不把新行为称作原始权重的数值复现。

某颜色在完整依赖域内没有有效观测支持、对应 accepted counts/sums 均为零时，Controller 线性融合不能生成该色的非零测量信号；缺色预测需要 RefineNet 的统计先验。只有 sums=0 不足以判断缺色：有效零信号或正负读噪声抵消也会产生零和。因此 controller-only 训练或融合层指标不能证明这项能力。现有最终 RGB/chroma/gradient 监督可继续用，额外 residual head 并非必要；辅助融合 loss 应区分有支持与无支持的颜色。

## 5. 如何验证“靠未饱和通道恢复”

必须按整组 burst 的颜色支持情况划分，而非仅统计 GT>1：

| 测试组 | 判定 | 主要验证目标 |
| --- | --- | --- |
| A | 所有颜色有正常观测 | 常规恢复与颜色稳定性 |
| B | 有饱和帧，但该色仍有短曝光/其他帧未饱和 | 曝光/多帧融合，不作为完全缺色能力证据 |
| C | 某颜色在整个 burst 均无未饱和支持，其余颜色有支持 | 真正跨通道、依赖统计先验的恢复 |
| D | 两色全 burst 饱和，仅一色有支持 | 更强欠定性及误色风险 |
| E | 所有颜色/曝光均饱和 | 输入仅提供截断界限，不能唯一确定真实幅值；单独报告不确定性与失败 |

C/D 组的支持判定要在配准后的共同评价区域进行，并覆盖模型依赖域。同色观测可能通过邻域平滑、融合 taps 或网络上下文到达输出；存在这些支持时，改善不能直接归因于跨通道恢复。建议先用整个输入及上下文都缺失某 CFA 颜色的受控案例验证，再扩展局部饱和的自然场景。

核心验收须包含**等曝光、部分颜色在整组输入及上下文中都缺失、其余颜色有观测支持**的受控样本，单列 C/D 组缺色误差和未缺失通道损伤。现有低增益等曝光 smoke 主要检查目标，bracket 的高光 RMSE 也不能替代这项验收。**待做：** 当前满阱后加读噪声，再按观测 DN 阈值判饱和，负读噪声可能把实际 clipped 样本移到阈值下；需标定仅依赖观测噪声/相机信息的 guardband 或 softmask，报告不同 ISO 下漏判与误拒率。clean `signal_saturation` 只能用于离线诊断，不得作为部署 mask；新规则应版本化，不改写历史 capture mask。

首先做接口验证：改变 masked 值不得改变 accepted statistics/fallback；透光率校正保留有效 HDR 值；零支持处理明确；固定 masks/geometry 后的输入增益测试保持预期比例关系。再验证物理资产：用留出 ISO/照明水平/视场/波长检查 PTC/PSF，测试单色质心、均匀场、delta PSF 和一次积分。

完成足够训练后，C/D 组至少对比：透光率校正后的 masked merge、旧数据 recipe 的模型、新 recipe 的模型，以及无光谱/无饱和扩增的消融。判断数据扩增本身的收益，必须另设相同拓扑、相同 output policy 的消融，避免把取消截断带来的可表示范围扩大误归因于光谱建模；旧 checkpoint 原行为另列为对照。各方法共享相机 RGB 单位、mask 和评估域，不额外对每个模型拟合增益。报告缺失颜色 RMSE、R−G/B−G 色差、未饱和通道受损、边缘/光晕，以及不同 ISO、镜头、亮度区间的结果；不能只用全图 PSNR 稀释缺色区域误差。

划分以场景、实体机身/拍摄会话、镜头为单位，保证留出与训练无同场景 patch 泄漏。真实 HDR 参考可以使用离线短曝光/包围曝光采集，但不得把参考帧作为“全 burst 缺色”测试的输入。以留出场景为单位统计误差区间；先证明相对基线的稳定改善，再扩大参数范围。

还应构造歧义对照：剩余通道观测相同、被剪裁通道真值不同的两个场景可能产生相同输入，因此不存在对任意场景的唯一精确恢复。成功结论应描述在声明的光谱/场景分布内改善，而非任意光谱都可还原。

仓库保存的 8-step spectral 检查只是执行验证。其报告中，oracle 条件下 highlight RMSE：corrected merge≈0.157495，Transformer≈0.157584，并没有展示对该基线的改善；该指标也未专门限定为 C 组。报告位置：[validation.json](spectral-validation/validation.json)。该归档记录当时 63 项测试；后续云环境复核曾运行 79 项测试。测试通过能证明相应工程协议通过检查，不能代替实测库验证或完整训练效果。

## 6. 建议的落地顺序与交付边界

1. **先解决数据到模型的接口。** 增加新 recipe 和等曝光 v1 数据入口，正确传 mask/透光率，修正 masked fallback，确定新 output policy、档案/权重身份。保留旧协议和原始模块核对结果。
2. **建立第一批可追踪资产。** 导入 PMN 的参考系数并注明 donor/简化范围；缺失满阱等字段显式处理。导入少量实测高光谱场景，审计一到几个专利镜头，完成离线 PSF 的数值验证。
3. **建立缺色基准并训练。** C 组作为核心，B 组单独列出；证明最终 RGB 改善与未饱和通道不受明显损害。同步采集代表机身/镜头的真实 RAW 校准，替换抽象假设。
4. **扩大有证据的覆盖表。** 增加真实机身、读出模式、ISO、镜头/光圈和实测状态，按留出误差补采。未测量区域仍可扩增，但继续标记推断。

这条路线能够实施与作者描述相对应的过程，允许分阶段验证，并适配现有代码。仅把随机范围设成 ISO100–800、f/2–8、3–5.76 µm，或者导入几张曲线，不足以支持“完整覆盖所有全画幅和主流定焦”的表述。要达到作者原始数据分布的逐项复现，仍需作者资产、采样协议及对应训练证据。

### 第一阶段的明确合同（待实施）

第一阶段只交付现有假设资产下的数据—模型—档案闭环，真实 PTC、PMN 导入、实测高光谱及镜头库分别进入后续资产阶段。

- `spectral-jsr-v3` 是先前建议的、尚未实现的 **v1 architecture adapter family**，不是现有相机数据协议名。它应消费 `spectral-camera-v3` 数据，默认显式 `target_stage: post_pixel`；其余阶段是分别训练评估的独立消融。待扩展 v1 recipe 至 RGGB、scale=2、K4/7/14、偶数 native_size≥16、等曝光；现有 camera-v3 生成器仍为 K7，旧数据协议和 Transformer 默认不变。
- 新 `prepare_v1_inputs` 只消费 batched 观测 RAW、曝光、透光率及 valid/saturation，按 CFA 校正，输出 RAW 与 validity；clean signal/GT 不参与模型条件。档案保存 unbatched 数据，推理显式补 batch 维。
- `JSRModel.forward` 在既有参数之后增加 keyword-only validity；Controller/RefineNet 权重布局不变。缺省 `legacy-clipped-v1`，新 recipe 显式选择 `linear-unclipped-v1`，不把新行为写成原权重复现。
- 待新增 v1 档案名为 `jsr-burst-v2`，pipeline 身份 schema 为 `jsr-pipeline-identity-v1`；分别绑定相机数据 protocol、adapter_family、target_stage/目标空间、output policy、独立 forward_model、光谱/profile/PSF/dark bank 的内容身份及 recipe。新 v1 checkpoint 使用 format_version=2；旧 format_version=1 保留原加载行为。档案保留原始 `observed_dn`，核对 RAW 归一化及观测饱和 mask；不从 float32 归一化 RAW 逆算 DN 来保证无量化时的精确阈值。旧 capture 继续使用原 DN 运算顺序。
- Profile 哈希保留原 canonical JSON，tensor 校验先转换为相同 float32；recipe 身份排除 split、seed/epoch/index、采样次数等迭代字段，保留成像/噪声分布、K、单位及资产身份。旧 checkpoint 的跨 recipe 对照只能作为显式 model-only 离线实验，不能绕过正式推理的身份检查。
- 首期 K4/7/14 分别使用明确的 recipe 训练/评估；新协议不能靠 `--frames` 静默跨越训练 recipe 的 K。参数布局相同不说明校准/数据身份或泛 K 恢复能力相同。
- 第一阶段训练用 joint/refine、learned_weight=0，仅监督最终 RGB；controller-only 及缺色融合辅助 loss 不在首期支持范围。新评估用 oracle，推理用 provided shifts；未适配掩码的 estimated/robust 对齐对新协议明确拒绝，旧路径保留。
- 8-step smoke 验收执行、数据/推理一致和 resume；质量状态保持 `not_established`，直到有留出 C/D 组及相同 output policy 的公平消融证据。
- coordinate-first 前向模型、网格/求积/PSF 支持收敛以及噪声后满阱的观测 guardband/softmask 均 pending；阶段切换保留相同观测 RNG，前向算子改版则另记身份并重新验证，不以 RAW parity 阻止有依据的数值修正。

## 7. 第一阶段计划与可复查资料

第一阶段的执行文件：[等曝光 JSR 光谱接入与缺色评估实施计划](superpowers/plans/2026-10-05-spectral-jsr-phase1.md)。后续真实标定和资产扩展按第 6 节分阶段进行。

- [传感器/PTC 调研、来源和采集建议](training-data-research/sensor/notes.md)
- [PMN/ELD/SIDD 等公开资产核验](training-data-research/sensor/assets/notes.md)
- [光学项目、处方审计与计算量](training-data-research/optics/notes.md)
- [实测有效核与 OpticStudio 路线](training-data-research/optics/empirical/notes.md)
- [OpticStudio 官方代码与依赖边界](training-data-research/optics/empirical/zemax/notes.md)
- [光谱主来源访问记录](training-data-research/spectral/fetch_results.json)
- [补充光谱/Unprocessing 访问记录](training-data-research/spectral/fetch_additional.json)

本报告引用作者提供的知乎段落作为待落实的描述，未独立核验作者原始资产。作者：姜尧耕（渔樵耕牍）；[原文链接](https://zhuanlan.zhihu.com/p/2088805454843585359)。
