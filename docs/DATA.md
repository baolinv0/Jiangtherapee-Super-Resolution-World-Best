# 数据构造与协议

新七帧 Transformer 光谱实验使用 [spectral-camera-v3 构造](SPECTRAL_DATA.md)：默认 GT 保留参考光学与原生像元面积响应（`post_pixel`），另有两个目标阶段消融；旧 v2 固定保留 `pre_optics`。61 波段、公开相机曲线、PTC/暗场与包围曝光链路保留，并新增等曝光控制配置；[目标定义与迁移](GOOGLE_TARGET.md)包含坐标、运行命令和验证依据。v1 Controller/RefineNet 的训练和评估仍直接使用旧合成数据，不能把 Transformer 的接入状态归到 v1。

调研后的改进路线见 [训练数据实现方案](TRAINING_DATA_IMPLEMENTATION_PROPOSAL.md) 和 [第一阶段实施计划](superpowers/plans/2026-10-05-spectral-jsr-phase1.md)。拟新增 `spectral-jsr-v3` 等曝光协议，保留已有数据协议与模型默认行为；目前该面向 v1 的新协议尚未实现，与已实现的七帧 Transformer `spectral-camera-v3` 不同。旧数据协议名 `inferred-synthetic-v1` 与 checkpoint 的 implementation 名 `inferred-jsr-v1` 分别标识数据和模型，不能混为一谈。

本文下面的合成链/未模拟范围仅适用于原`inferred-jsr-v1`。新七帧数据链的PTC/暗场接口、ISO档案、三波段PSF、像元积分、包围曝光与信号饱和见[TRANSFORMER.md](TRANSFORMER.md)。两者均为代理数据，作者原训练集未恢复。

所有退化范围均为本工程的假设。原JSR训练图像、camera模型、损失与校准子集未从公开快照中确认。

## 可用数据

| 来源 | 用途 | 本工程处理 |
|---|---|---|
| [DIV2K 官方](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | HR RGB合成代理 | 实现`--layout div2k`；train_HR 0001–0800训练，valid_HR 0801–0850用于验证、0851–0900作为本工程held-out test。这50张**不是官方DIV2K测试集**。官方条款为学术研究用途 |
| [DBSR官方](https://github.com/goutamgmb/deep-burst-sr)、[Zurich loader](https://github.com/goutamgmb/NTIRE21_BURSTSR/blob/master/datasets/zurich_raw2rgb_dataset.py) | 更贴近burst文献的训练参考 | 原DBSR使用Zurich Canon RGB合成；可以准备本地RGB manifest，但本工程没有自动克隆其完整unprocessing、划分或隐藏test协议 |
| [BurstSR/DBSR](https://github.com/goutamgmb/deep-burst-sr) | 真实跨相机评价 | 应用官方空间/颜色对齐评分；本工程的synthetic指标不能直接宣称官方BurstSR分数。未提供可混淆二者的转换器 |
| [HDR+ 数据](https://www.hdrplusdata.org/dataset.html) | 同曝光真实RAW鲁棒性 | 可选RAW导入与裁块推理；处理后的HDR+输出不是独立干净HR线性GT |
| 自有线性RGB或相机RAW | 域匹配实验 | 浮点HWC `.npy`，明确sensor/working-RGB空间；相机RAW需扣黑、白电平、CFA、曝光元数据 |

下载由使用者从原站进行，本工程不自动获取大型数据或重新分发图片。DIV2K只需要官方`Train Data (HR images)`与`Validation Data (HR images)`，解压为：

```text
DIV2K/
  DIV2K_train_HR/0001.png ... 0800.png
  DIV2K_valid_HR/0801.png ... 0900.png
```

```bash
python -m jsr_repro.prepare --source /datasets/DIV2K --layout div2k --encoding srgb --output data/train
```

其他本地图像使用`--layout custom`（默认），固定seed在**源scene**层面随机分成约80/10/10，至少各1组。多张曝光/视角属于同scene时，放同一子目录并指定`--group-by parent`。独立照片则用默认file。该工具不能识别文件名未知的同场景、近重复或已经裁好的关联patch，须在导入时正确分组。训练后不能把重新划分的数据当独立test。

## Manifest

`manifest.jsonl`逐行含 `path,sha256,scene_id,encoding,split`，路径相对于manifest或为本机绝对路径。训练/测试启动时检查内容SHA256与跨split的scene/hash泄漏。相同文件内容去重，裁patch在划分后执行。`dataset.json`保存布局、seed、实际分组计数和manifest哈希。

```json
{"path":"sources/scene_0001.npy","sha256":"实际文件的64位十六进制SHA256","scene_id":"scene_0001","encoding":"linear","split":"train"}
```

PNG/JPEG等由Pillow读为8-bit RGB；线性高位深数据请用float32 HWC `.npy`，范围[0,1]。`srgb`使用标准逆sRGB曲线。**逆sRGB并不能撤销未知tone mapping、锐化、降噪或恢复相机光谱响应，因此这是linear RGB proxy而非实测sensor RAW。**

## 默认合成链

1. 根据`seed/epoch/index/source_sha256`产生独立稳定seed；crop前保留足够运动/PSF边缘。
2. 若原图小于所需crop则明确双线性放大并写入`source_resized`；正式数据应避免这种情况。
3. 在CFA前做90度旋转/翻转增强；apply `2^EV`与R/B随机gain的逆增益，G不变。
4. HR高斯光学blur；目标是该参考空间、blur后的RGB。
5. 逐帧参考平移，第0帧固定0；双线性场景采样，每个native sensel对s×s HR单元做面积积分。
6. RGGB取样；Poisson shot noise（方差`shot_coefficient*signal`）与Gaussian read noise（方差`read_std²`）。
7. 可选sensor[0,1]截断和14-bit量化；crop/运动/曝光/噪声/PSF/量化/饱和比例均保存在metadata。

`train_k14.yaml`默认native64、s2、K14、位移±2 native像素、HR blur sigma 0.2–1.2、EV −5–0、R/B gain1–2.5、shot系数1e−5–4e−3、read标准差1e−4–3e−3。均匀采样范围是本工程选择，不能写成JSR原超参数。

未模拟lens shading、realistic motion trajectories、rolling shutter、PRNU/DSNU、量化相关噪声、动态物体、光谱crosstalk或完整反ISP；未实现原作者LCA/15类像差实验。高ISO/真实暗部能力需单独评价。

## 统一评价边界

- 同一物理区域、同一native倍率与裁边，固定线性峰值1，不拟合各模型曝光/颜色。
- oracle几何只衡量给定平移的后端；estimated轨道使用相同输入估计平移。两者分文件报告。
- 学习模型、legacy及单帧baseline共用退化/target，且默认只用`test`split。
- 帧数改变、恢复归一化/细化开关的干预不等于重新训练的公平消融。
- 512倍数字提亮不等于传感器动态范围增加9档。这里只测条件性响应，不造原作者实验数字。
