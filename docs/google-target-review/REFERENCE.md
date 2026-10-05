# Coordinate-first discrete-field reference

文件：`coordinate_first_reference.py`；实际数值：`coordinate_first_results.json`。

在仓库根目录运行（JSON 证据写在脚本旁边）：

```bash
python -m pip install -e ".[test]"
python docs/google-target-review/coordinate_first_reference.py
```

这是独立的可运行数学参考，不改生产数据协议，不运行网络训练。源场/PSF仍离散，空间查询仍是 bilinear，像元积分仍是中点求积；它不是连续物理真值，也不能替代场景、PSF、面积积分的收敛研究。

## 对四节点近似的改写

对当前采用的空间变光学近似，节点核为固定的 `h_iλ`，输出传感器位置的混合权重为 `w_i(x)`。场景移动 `T_u L(x)=L(x+u)`。在连续解释中，每个固定节点卷积满足：

`O[T_u L](x) = Σ_i w_i(x) (h_i * T_u L)(x) = Σ_i w_i(x) C_i(x+u)`，其中 `C_i=h_i*L`。

参考先对原场景产生各节点卷积场 `C_i`，每个传感器求积坐标 `q=center+δ` 查询一次 `C_i(q+2u)`，混合用 `w_i(q)`。然后按共同 SRF 和 footprint 积分。这样消除“先 warp 至同样低密度网格，再 bilinear 求积”的中间重建；权重始终在传感器坐标，不随场景移动。

这只在每个节点内部是平移不变核、移动是全局平移、SRF及相同面积响应保持线性的假设下成立。旋转、空间变几何、曝光轨迹或复杂 ray-traced 位置相关 PSF 需要重新推导，不应直接套用。

## RAW 与 target

- RAW 干净观测：`output_scale=1`；native中心 `m+2j+0.5`；共同原生边长 `2sqrt(fill)`。
- post_pixel target：同一 fields，零场景位移，`output_scale=2`；dense中心 `m+k`；footprint仍是同一个原生面积。
- pre_optics / post_optics 需要分别显式从上游场取样；当前参考只演示 post_pixel 的观测算子，不偷偷改变目标定义。
- 曝光、透光率、CFA、Poisson、满阱和 ADC 全部在上述干净 camera RGB 原生观测后接原有 capture 链路。干净 dense target 不接这些观测作用。

## 已通过的验收

| 检查 | 实测 |
|---|---|
| 单节点 coordinate-first vs 对同一光学场直接偏移中心积分 | max差 0 |
| 单节点零位移 vs 当前生产链路 | max差 5.36e-7（当前零位移 warp 有 float32 网格舍入） |
| 0.25cycles/HR、位移0.25native 的当前对比度/参考对比度 | 0.7499988；coordinate-first 与直接积分一致 |
| 四节点 native 与正确对齐采样中心的 dense even samples | 逐值相同，max差 0 |
| dense保留原生面积 vs错误把边长缩半 | max差 0.0121616；强反例未误通过 |
| ramp场、固定field权重、位移[2,1]native 的解析亮度差 | max误差 2.20e-7 |
| 错误先混合四节点场，再整体移动，ramp解析误差 | 0.0019306；对应错移权重 max变化0.0431378 |
| SRF 与共同面积响应/查询的线性可交换检查 | max差 1.79e-7 |
| 61波段、4节点、Canon5DII SRF、K7 native输出 | 有限非负 `[7,3,16,16]` |
| 同一61波段fields的post_pixel target | `[1,3,32,32]`，对齐采样native一致max差0 |

## 可用于工程接入的安全优化

每个节点仍先完成 **61个波长各自的卷积**，之后立刻执行 camera SRF，缓存该节点的camera RGB场。因为查询、field权重和footprint都不依赖波长，SRF可以与这些线性操作交换。`prepare_camera_fields` 实测与保存完整光谱后最后积分的结果 max差 **1.79e-7**。

这不是用3个替代RGB PSF：61个波长核完全保留。示例缓存从 3,060,736 个值降到 150,528 个值；每burst只做4×61次卷积（K7不再做7×4×61），后面只查询3通道。实际性能仍需按训练patch、batch、device测量，不能由该小probe推断GPU吞吐。

## 协议与未解决的数值限制

- 四节点时 `interpolate(w*C)` 与 `w(q)*interpolate(C)` 不完全相同。基于pupil模拟的四节点在零运动的max差约 **9.41e-5**，非零运动 vs当前链路max差约0.01089/0.00936。因此新采样器必须有新protocol或明确的forward-model identity，不能静默替换v2/v3以冒称parity或strict resume。
- `Q4→Q64` 的差为0.0014933，`Q16→Q64`降为8.79e-5；这里只支持“面积求积在该样本上趋于稳定”，不证明Q64是连续相机真值。formal recipe应有代表频带、fill、PSF、运动的收敛预算。
- 这项修复避免额外运动重建，却不能恢复源场原本缺失的高频。若目标是严肃的2× SR训练，应继续把场景模拟密度与输出2×分开，并验证更密源场的forward观测收敛。
- 主瓣离开有限核支持后的能量归一化问题仍需单独处理；该参考不宣称已修复生产PSF支持检查。


## 其他诊断与实测摘要

```bash
python docs/google-target-review/diagnose_pixel_optics.py
python docs/google-target-review/diagnose_full_well_mask.py
python docs/google-target-review/diagnose_burst_support.py
```

`diagnose_pixel_optics.py` 重建运动相位、有限PSF支持和FFT收敛诊断；`diagnose_full_well_mask.py` 重建观测饱和阈值、候选guard band与误拒控制；`diagnose_burst_support.py` 重建等曝光/包围曝光颜色支持的原控制JSON并核对。均只写数值证据，不改生产模块或运行网络训练。脚本调用本仓库实现，审查基线为提交`0bc29f1`，后续修改后应重新检查其合同与解释。完整review见[GOOGLE_TARGET_REVIEW.md](../GOOGLE_TARGET_REVIEW.md)。

`full_well_mask_probe.json` 是主审查的seed2026/native64记录；带脚本的独立复核使用seed1189/native128，结果存于`full_well_mask_independent.json`。`saturation-protocol-probe.json` 为关闭噪声/量化、沿用解析profile的ISO800等曝光/包围曝光控制记录（seed43、native16、零运动、线性白场、scene_gain=1.4、Canon 5DMarkII SRF、Mallett lift、post_pixel）；候选策略尚未接入生产生成器。`validation-summary.json` 与`pytest.txt` 保存本轮两个8步CPU工作流和完整110项测试摘要。所有结果都不构成实拍质量或完整作者训练分布的证明。
