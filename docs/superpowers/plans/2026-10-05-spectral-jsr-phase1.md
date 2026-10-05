# 光谱数据接入 v1 JSR：第一阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有光谱相机生成器接入独立 v1 JSR，形成兼容旧协议、可追溯且能验证整组缺色条件的数据—模型—档案闭环。

**Architecture:** 新增 `spectral-jsr-v3` 等曝光协议，通过一个 RAW adapter 将观测饱和掩码和 CFA 透光率传入现有模型。保留 Controller/RefineNet 的层结构；以显式 output policy 区分旧截断与新线性 HDR 输出。训练、验证、评估、推理共享数据身份和档案解释。

**Tech Stack:** 现有 Python/PyTorch/NumPy/PyYAML/pytest；CPU 工程验证；不增加依赖。

**Spec:** [训练数据实施方案](../../TRAINING_DATA_IMPLEMENTATION_PROPOSAL.md)。本计划对应其第一阶段，所有复选框均待实施；本次文档更新没有改产品代码或运行新方案。

## Global Constraints

- 数据协议：保留 `inferred-synthetic-v1`、`speech-camera-proxy-v1`、`spectral-camera-v2`；新增 `spectral-jsr-v3`。
- v1 未指定数据协议仍用 `inferred-synthetic-v1`；Transformer factory 未指定协议仍用 `speech-camera-proxy-v1`。
- 新协议仅支持 RGGB、scale=2、K∈{4,7,14}、偶数 native_size≥16、所有 exposure ratio=1；不把多曝光输入偷偷用于 v1。
- 新 checkpoint 绑定训练 K；K4/7/14 分别以明确 recipe 训练和评估。相同模型维度不证明跨 K 泛化，首期评估/推理拒绝改变 K。
- 模型保留 Controller 151→36、RefineNet 输入六通道和相同 state_dict 张量布局；旧 positional 参数兼容。
- 输出策略：缺省 `legacy-clipped-v1`；新 recipe 显式 `linear-unclipped-v1`。新行为不声称原作者权重的完整数值复现。
- 不将 clean_raw、signal_e、signal_saturation、GT 或 GT 导出的 mask 输入模型；有效负 black-subtracted RAW 不得按正值筛掉。
- 第一阶段仅用现有解析/公开光谱资产，明确标注混合假设；真实 PTC、PMN 导入、实测光谱、镜头 PSF 和拍摄校准仍为后续依赖。
- 新协议首期只允许 joint/refine、learned_weight=0；最终 RGB/chroma/gradient loss 不变。Controller-only 及缺色融合辅助 loss 后续另行设计。

## Review Focus

- 未指定 protocol 的旧 v1/Transformer 配置不能互换默认数据；Task 1 固定 factory 默认与旧种子。
- 校正后 RAW>1 和负读噪声均可能是有效观测；Task 2 固定掩码先于透光率校正。
- 无有效颜色支持或全无支持不得从 masked 值/其他颜色生成 fallback；Task 3 固定该路径和零输出。
- 同尺寸但不同校准、PSF、output policy 的档案不能静默匹配；Task 5 固定完整身份与缺字段拒绝。
- 短曝光/邻域同色证据不能被算作全 burst 缺色恢复；Task 6 用整个输入缺色的受控案例固定分组。

## 文件与责任

| 文件（相对仓库根） | 责任 |
| --- | --- |
| `reproduction/jsr_repro/{bracket_data,spectral_data}.py` | 协议 factory、兼容的等曝光生成与 dataset identity |
| `reproduction/jsr_repro/raw_adapter.py`（新增） | 批量 RAW/观测 mask/透光率到 v1 输入 |
| `reproduction/jsr_repro/{frontend,model}.py` | masked fallback、validity 传递、版本化输出组装 |
| `reproduction/jsr_repro/{train,evaluate,infer,validate}.py` | 同一 factory/adapter/checkpoint/档案合同 |
| `reproduction/jsr_repro/burst_archive.py`（新增） | 新档案读写、观测 mask 校验和 pipeline identity |
| `reproduction/jsr_repro/physical_sensor.py` | 提取共享观测饱和阈值计算，保持旧 capture 数值 |
| `reproduction/jsr_repro/saturation_benchmark.py`（新增） | 整组颜色支持分组和缺失通道指标 |
| `configs/spectral_jsr_smoke.yaml`（新增） | 小模型、K4、等曝光、joint、线性输出、8-step smoke |
| `tests/test_spectral_jsr.py`（新增）、现有相关 tests | 新合同、集成、resume 与兼容回归 |

## Task 1: 版本化 factory 和等曝光光谱生成

**Files:** Modify `reproduction/jsr_repro/bracket_data.py` 的 `make_burst_dataset`；`spectral_data.py` 的 `_rng`、`synthesize_spectral`、`SpectralBurstDataset`。Test `tests/test_spectral_jsr.py`（待创建）和现有 `tests/test_spectral_data.py`。

**Interfaces:** `make_burst_dataset(manifest, split, options, seed=1234, profile=None, *, default_protocol="speech-camera-proxy-v1", samples_per_scene=None) -> Dataset`。v1 显式传 `default_protocol="inferred-synthetic-v1"`；旧 Transformer 调用不变。`synthesize_spectral(source, options, seed, profile=None, assets=None)` 签名不变，按 options.protocol 分派；sample raw[K,1,H,W]、shifts[K,2]、target[3,2H,2W]、exposure[K]、transmission[3]。

- [ ] 在新 tests 写 `test_factory_defaults_remain_distinct`：分别断言旧 v1 为 SyntheticBurstDataset、旧 Transformer 为 BracketBurstDataset；同 seed/epoch/index 的旧 v1 raw/target 与直接构造旧 Dataset `torch.equal`。
- [ ] 写 `test_spectral_equal_exposure_k_and_gt_invariance`，参数 K=4/7/14：exposure 与 ones 严格相等；raw/target 维度如上；frame0 shift 严格零；同 seed 的 GT 在 f-number/PSF/noise 改变后严格不变。K=6、scale=1、多曝光配置应 ValueError。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k 'factory or equal_exposure' -q`，确认新协议/接口尚未支持而失败。
- [ ] 实现分派；v2 保留 `_rng` 原 token、七帧 draws 和 bracket 构造；v3 使用独立 protocol token 和 ones，不消耗旧协议的 RNG 路径。保留 v1 顶层 samples_per_scene 训练设置及 val=1 语义；仅 v3 metadata 增加完整 profile、quantize 和标准化 capture_recipe 快照，供 Task 5 校验。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py tests/test_spectral_data.py tests/test_data_training.py -q`，新合同与旧谱协议均通过。
- [ ] Commit `feat: add versioned equal-exposure spectral JSR dataset`，只暂存本任务文件。

## Task 2: 唯一的 observed-mask / transmission adapter

**Files:** Create `reproduction/jsr_repro/raw_adapter.py`；Test `tests/test_spectral_jsr.py`。

**Interfaces:** `prepare_v1_inputs(sample: Mapping[str, Tensor], protocol: str) -> tuple[Tensor, Tensor | None]`。仅接受 batched raw[B,K,1,H,W]；新协议还要求 exposure[B,K]、transmission[B,3]、saturation/valid 与 raw 同形、有限且为二元 mask。输出 raw 同形、validity 同形；legacy 原样 raw 和 None。

- [ ] 写 `test_adapter_preserves_hdr_and_negative_observations`：观测 RGGB R=.7/G=1/B=.35、transmission=[.5,1,.25]、G saturation=1，断言校正 R/B≈1.4 且有效、G 无效；R=-.01 且 saturation=0 时校正=-.02 且有效。
- [ ] 写 `test_adapter_uses_observed_fields_only`：改变/删除 signal_saturation、clean_raw、signal_e、target，不改变两个输出；新协议 missing exposure/transmission/mask、非二元 mask、NaN、transmission≤0/>1、exposure≠1 均 ValueError。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k adapter -q`，确认 adapter 不存在而失败。
- [ ] 实现 validity=`valid*(1-saturation)`，按 RGGB 的 R/G1/G2/B 布局除以 transmission；校正前的 saturation 已由观测域产生。不得依据校正值范围或 black_invalid 拒绝负读噪声。
- [ ] Run 同一 adapter 命令，确认通过且输入 tensors 未被原地修改；旧协议直接路径输出与输入对象一致。
- [ ] Commit `feat: adapt observed spectral RAW to v1 model inputs`。

## Task 3: 掩码封闭与版本化 RefineNet 组装

**Files:** Modify `reproduction/jsr_repro/frontend.py` 的 `phase_splat`；`model.py` 的 `JSRModel`、`RefineNet.factorized`。Test 新 tests 及现有 `tests/test_model.py`。

**Interfaces:** `JSRModel.__init__(..., lca_offsets_native=None, *, output_policy="legacy-clipped-v1")`；`JSRModel.forward(raw, shifts, geometry=None, diagnostics=False, *, validity=None)`。`RefineNet.factorized(value, amplitude, reference_amplitude=REFERENCE_AMPLITUDE, *, output_policy="legacy-clipped-v1")`。无新增权重、buffer 或 Controller 特征维度。

- [ ] 写 `test_masked_splat_is_independent_of_rejected_values`：所有 G 无效时，把它的 raw 从1改为100，phase/count/sum/legacy 严格相同且 G count/sum/legacy 全零；极端出界 shift 下 fallback 仅取有效同色样本；全无效时所有统计/模型输出严格零。
- [ ] 写 `test_unclipped_policy_removes_amplitude_ceiling`：RefineNet 所有参数归零、out.bias=[80,0,0]，输入六通道前RGB=[.05,0,.05]，amplitude=sqrt(2/3)*.05；旧 G≤amplitude/REFERENCE_AMPLITUDE，新 G=80*amplitude/REFERENCE_AMPLITUDE>8。两 policy state_dict keys/shapes 严格相同；非法 policy 拒绝。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k 'masked_splat or unclipped' -q`，确认 mask fallback/新 policy 测试失败。
- [ ] masked 分支用有效同色的加权均值作 fallback，无同色支持返回0；unmasked 分支保留原数值路径。linear policy 保留当前色差组装和 amplitude 缩放，仅不执行 normalized RGB 的 `[0,1]` clamp；零 amplitude 输出仍为0。
- [ ] Run `python -m pytest tests/test_model.py tests/test_spectral_jsr.py -q`；固定 validity/geometry 时输入×{.125,3.7}输出同比例，rtol=2e-5/atol=2e-7；旧 positional `model(raw,shifts,geometry,True)` 保持可调用及旧核对结果。
- [ ] Commit `feat: close masked fusion and version HDR output assembly`。

## Task 4: 接通训练、内部验证与独立评估

**Files:** Modify `reproduction/jsr_repro/train.py` 的 `run_training`、`validate_model`；`evaluate.py` 的 `evaluate`；`validate.py` 的 sample/model 调用；Create `configs/spectral_jsr_smoke.yaml`。Test 新 tests。

**Interfaces:** 以上公开函数/CLI 参数不变；所有 dataset 由 Task 1 factory 创建，所有模型调用先用 Task 2 adapter。新配置 model 使用现有 smoke 宽度4/8、blocks2、scale2 与 linear policy；data 复制谱 smoke 的小 PSF 设置但 K4/exposure=ones；train joint、steps8、validate_every4、learned_weight0。

- [ ] 写 `test_v1_training_validation_evaluation_use_identical_adapter`：spy 证明三条路径收到校正 raw/validity，且不收到 clean/GT mask；相同样本 train 内验证与独立 oracle 评估的最终 RGB 指标相同；旧默认 config 保持直接 raw 路径。
- [ ] 写 `test_spectral_training_rejects_unsupported_stage`：新协议 stage=controller 或 learned_weight>0 明确 ValueError；joint/refine 有有限 loss 与有效 RefineNet 非零梯度，不用 Controller 缺色输出宣称恢复。
- [ ] 写 `test_spectral_evaluation_cannot_change_training_k`：K4 checkpoint 的 frames=None/4 可以评估，frames=7/14 均 ValueError；K7/K14 recipe 同样仅评估自身 K；旧协议原 override 行为保持。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k 'training or evaluation' -q`，确认 v1 未接入谱数据而失败。
- [ ] 接入 factory/adapter；新协议配置必须显式 linear policy，否则启动拒绝以防误选。旧 train loss、stage、frame override 与数据 defaults 不变；新 evaluate 的 frames override 仅可省略或等于 checkpoint 的训练 K，alignment 必须 oracle。v3 validate 不调用现有 estimated 路径，报告其待支持；旧 validate 两种对齐都保留。
- [ ] Run 上述 tests；8-step 训练只检查执行、有限值和输出维度，不要求质量改善；Task 5 完成后再跑完整 CLI archive 闭环。
- [ ] Commit `feat: train and evaluate v1 on equal-exposure spectral bursts`。

## Task 5: 档案与 checkpoint 的身份合同

**Files:** Create `reproduction/jsr_repro/burst_archive.py`；Modify `physical_sensor.py`、`spectral_data.py` identity、`train.py` checkpoint、`infer.py`、`validate.py` archive。Test 新 tests 与现有 CLI/resume tests。

**Interfaces:** `save_v1_burst(path: str | Path, sample: Mapping, pipeline_identity: dict) -> None`；`load_v1_burst(path: str | Path, expected_identity: dict | None) -> dict`；`observed_saturation_mask(observed_dn: Tensor, camera: dict, quantize: bool) -> Tensor`（physical_sensor）；`build_pipeline_identity(dataset, output_policy: str) -> dict`（burst_archive）。

**Schema:** `jsr-burst-v2` NPZ 保存未校正 raw[K,1,H,W]、捕获时的 observed_dn（同 raw 形状且保留其 dtype）、shifts[K,2]、exposure[K]、transmission[3]、valid/saturation 同 raw，以及纯 JSON metadata：archive_schema、protocol、iso、完整 profile、选中 camera 参数、quantize、capture_recipe、pipeline_identity。target 为可选离线评估字段，不影响推理。identity schema=`jsr-pipeline-identity-v1`，字段为 protocol、output_policy、target_space、spectral_assets、profile_sha256、psf_identity、read_noise_bank_sha256、recipe_sha256；缺失可选外部资产以 null 表示，解析 PSF 使用明确模型版本与参数哈希，不冒充实测库。

**Metadata contract:** save 接受 unbatched sample，其 metadata 可以是单个 JSON 字符串或 dict；输出 metadata 为 NPZ 标量 Unicode JSON，load 返回 unbatched tensor dict 及单个 metadata 字符串。infer 在 reader 后仅给数字 tensors 增加 batch 维，再交 adapter；训练 DataLoader 的 metadata 则是 list[str]，逐项解析用于校验而不当作数值张量或模型特征。生成器提供完整 profile/capture_recipe 快照，save 校验并补 archive_schema/pipeline_identity；profile 使用现有 `profile_identity` 的 canonical 内容哈希，不将JSON中的 .7 改写成tensor的float32近似值。

recipe_sha256 排除路径/manifest/output、split、seed/epoch/index、samples_per_scene 等样本迭代字段；保留训练K、成像分布、单位、噪声/量化和内容资产身份。单样本随机状态另存 metadata。reader 校验 profile 哈希、camera=profile.iso[iso]、ISO 位于 capture_recipe.iso、quantize 与 recipe 相同、recipe 哈希与身份相同；transmission 严格相等比较之前将 profile.channel_transmission 转为与 tensor 相同的 float32 dtype/device，再 `torch.equal`。

- [ ] 写 `test_archive_roundtrip_and_identity_guard`：以 allow_pickle=False 往返 tensors 严格相等；分别更改谱资产/profile/PSF/dark bank/recipe/output policy 的身份字段，加载均拒绝；缺 metadata/mask、非法曝光、伪造 saturation、NaN、ISO 越出 recipe、camera/transmission 与 profile 不符均拒绝。JSON 字符串和dict的 save 输入产生等价 metadata；load→batch→adapter 与 DataLoader→adapter 数值严格相同。保存后 sample 不被改动。
- [ ] 写 `test_float32_transmission_and_iteration_independent_identity`：profile [.7,1,.55]→同dtype float32 tensor 时通过，profile JSON 哈希不变；任一 transmission 值改一个float32 ULP即拒绝。train samples_per_scene=2、val=1以及不同split/seed/epoch/index具有相同pipeline identity；K或noise/quantize改变则身份不同。
- [ ] 写 `test_new_checkpoint_resume_and_legacy_load`：v3 format_version=2、implementation=`inferred-jsr-v1`、必含 pipeline_identity；8 steps 全程与4+resume4的模型/optimizer/scheduler/RNG一致；资产内容改变但路径相同也拒绝 resume。旧 format_version=1 缺 output_policy 仍按 legacy policy 载入。
- [ ] 写 `test_dn_saturation_boundary_and_legacy_masks`：quantize=false 的阈值正下方/恰好阈值/正上方（以 torch.nextafter 构造）分别 false/true/true；分别覆盖满阱先截断、ADC先截断和quantize=true半码阈值。共享函数与原 capture 比较语句的 mask `torch.equal`；DN/raw不一致的档案拒绝，不能通过 RAW 逆算、round 或任意 epsilon 改变边界。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k 'archive or checkpoint' -q`，确认新 schema/身份校验不存在而失败。
- [ ] 抽取 sensor 原阈值为共享函数，capture 直接传原 observed_dn tensor，保留原 dtype、量化/截断次序及比较顺序；阈值=`min(white,black+full_well/gain)-(0.5 if quantize else 0)`。reader 从保存的 DN 快照依同一运算计算 `(observed_dn-black)/(white-black)` 再 float32，要求与 raw 严格相等，再由原 DN 判 sat 并验证存储 saturation。不从 normalized RAW 逆算 DN 判阈值，不额外 round/epsilon。构造 canonical JSON 内容身份，checkpoint/档案严格匹配；旧档案仅配旧 checkpoint。
- [ ] 新 v3 推理首期仅支持 `--alignment provided`，且输入 K 必须等于训练 recipe 的 K，保证校正/mask 路径一致；estimated/robust 清晰拒绝，避免现有无 mask 配准器改变缺色实验。推理 float32 linear_rgb.npy 保留>1/负值，preview 可继续按旧显示规则截断。新 policy 与旧 weights 的初始化只在显式 model-only initialize、相同参数拓扑且记录 policy 变更时允许；resume 不允许跨策略。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py tests/test_data_training.py tests/test_cli_contracts.py -q`；然后 `python -m jsr_repro.validate --config configs/spectral_jsr_smoke.yaml --output runs/spectral-jsr-phase1`（全新目录），断言 checkpoint、NPZ、provided inference 身份一致且 offline 与 archived inference 输出一致。
- [ ] Commit `feat: bind spectral archives and checkpoints to pipeline identity`。

## Task 6: 缺色基准与最终验收

**Files:** Create `reproduction/jsr_repro/saturation_benchmark.py`；Modify `evaluate.py`/`validate.py` 的新协议报告；Test 新 tests；更新 `docs/SPECTRAL_DATA.md` 与本计划状态。

**Interfaces:** `classify_burst_support(saturation: Tensor, valid: Tensor) -> list[str]`，输入[B,K,1,H,W]，返回每 burst 的A/B/C/D/E；`missing_channel_metrics(prediction: Tensor, target: Tensor, missing_channels: Tensor, border: int = 4) -> dict[str, float | None]`，prediction/target[B,3,2H,2W]、missing_channels[B,3] bool。受控缺色覆盖整个输入/上下文，不只中心 crop。

- [ ] 写 `test_support_groups_distinguish_burst_evidence`：所有色有效→A；仅一帧G饱和、其他帧G有效→B；所有帧全输入G饱和→C；G/B同样缺失→D；全色饱和→E。valid=0 的值不算支持；若存在邻域同色有效观测，不给它“完整依赖域缺色”的标签。
- [ ] 写 `test_missing_channel_metric_does_not_hide_color_error`：GT三色均1.4、prediction R/B=1.4/G=0、missing=[False,True,False]，断言 missing_rmse≈1.4、supported_rmse=0、chroma_rmse≈1.4；GT整体>1但所有色均有效时 missing_rmse=None，不当作C组。
- [ ] Run `python -m pytest tests/test_spectral_jsr.py -k 'support_groups or missing_channel_metric' -q`，确认分组/指标不存在而失败。
- [ ] 实现固定单位/裁剪的报告，至少列 corrected masked merge、legacy checkpoint、新 checkpoint；A/B/C/D/E分开列数目，E标明无法唯一恢复。legacy checkpoint 对照为显式离线 model-only 跨recipe实验，记录权重哈希、校正raw/validity适配、跨域输入和 `legacy-clipped-v1` output policy；正式 infer 仍遵守 Task 5 身份拒绝，不增加绕过参数。后续质量实验另加同拓扑同linear policy的无谱/无饱和消融，排除取消截断带来的混淆。
- [ ] 写 `test_legacy_benchmark_is_explicit_cross_recipe_experiment`：旧checkpoint/新档案的正式 infer 仍拒绝；显式离线对照报告必须包含 cross_recipe=true、checkpoint_sha256、input_adapter=`spectral-jsr-v3`、output_policy=`legacy-clipped-v1`，不声称该输入属于旧训练分布。
- [ ] Run `python -m pytest -q`；依次运行旧 `python -m jsr_repro.validate --output runs/phase1-regression-v1`、`python -m jsr_repro.validate_transformer --output runs/phase1-regression-transformer`、`python -m jsr_repro.validate_spectral --output runs/phase1-regression-spectral`；所有输出使用全新目录，不覆盖历史证据。
- [ ] 将新 CLI 证据与旧回归分列保存。只有接口/恢复训练闭环通过才写 engineering_status=passed；质量状态仍为 not_established，直到留出场景C/D组相对同策略基线稳定改善并报告未缺失颜色损伤。实测 PTC/PSF/PTC字段来源和完整机身覆盖不属于本阶段完成标准。
- [ ] Commit `test: add burst-wide missing-channel benchmark and phase-one evidence`。

## 执行环境与交付

在仓库根以 Python 3.12 建立环境并安装已记录依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-validated.txt
python -m pip install -e '.[test]'
```

CUDA 环境先安装平台对应的 PyTorch，再安装其余依赖；本阶段验收使用 CPU。当前已配置云环境可选用 `source /workspace/jsr-repro-cloud/env.sh`，该路径不是 GitHub 使用者的前置条件。以上新 tests/模块/config 均为待创建，不是当前可运行入口；旧 tests/CLI 已存在。约定每任务先记录失败再实现并复测；不要因本计划的文档批准步骤重复阻塞已有用户授权。

交付应包括变更摘要、精确验证结果、pipeline schema 示例、默认兼容情况和后续资产依赖。质量实验不设8-step通过阈值，不把解析/公开 donor 混合链路写成作者的全画幅/定焦实测库。
