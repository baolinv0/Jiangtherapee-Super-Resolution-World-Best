# Additional measured-noise assets verified 2026-10-05

This research only saved source documentation, small calibration files, JSON extracts, and request results under `/workspace/jsr-research/sensor/assets`. No source/configuration files were changed and no large dataset was downloaded. Raw GitHub and GitHub HTML requests succeeded; YorkU, arXiv, CVF, api.github.com, and the PMN project host returned proxy tunnel 403. A proxy denial is an environment restriction, not evidence that an upstream asset is unavailable publicly.

## PMN — immediately obtainable fitted calibration coefficients and maps

Official project: https://github.com/megvii-research/PMN

Consistent usable main snapshot: `d207d3eb62e3a5106861992c0562054adf9a9c70` (verified GitHub tree HTML). TPAMI snapshot: `9f985ba24f3e605338359d0c770148b5eb08ee48`. The initial Sony dictionary of `get_camera_noisy_params_max` is exactly equal between those two snapshots (AST literal comparison, all 28 entries).

Primary source and provenance:
- README: https://raw.githubusercontent.com/megvii-research/PMN/d207d3eb62e3a5106861992c0562054adf9a9c70/README.md
- Parameters: https://raw.githubusercontent.com/megvii-research/PMN/d207d3eb62e3a5106861992c0562054adf9a9c70/data_process/process.py
- Resource tree: https://github.com/megvii-research/PMN/tree/d207d3eb62e3a5106861992c0562054adf9a9c70/resources
- Main README explicitly states that dark shading and noise calibration require massive dark frames and that it supplies calibration results. It also explicitly warns: “The calibration is based on a SonyA7S2 camera, which has the same sensor as the public datasets but not the same camera.” Thus these are calibrations of one different physical body of that model, not measurements of each SID/ELD body and not a model-family variability study.
- Main LICENSE is Apache-2.0 (pinned main LICENSE downloaded and inspected; HTTP200,10,271B). This establishes repository licensing; it does not establish all rights in independently downloaded upstream SID/ELD/LRID data.

The full-frame Sony A7S II source is a useful measured low-light noise proxy. Its large-pixel sensor is not a measured 3–5.76 µm camera library. The source files do not document physical active-area/pixel-pitch metrology; body class and approximate 8.4 µm pitch are camera model context, not calibration quantities read from these files. No APS-C/MFT calibration is supplied by these PMN resources.

### Exact ISO 100–800 anchors

`generate_noisy_obs` converts normalized clean pixels to DN with `(wp-bl)`, samples `Poisson(y_DN/K)*K`, adds read/row noise in DN, normalizes, and clips. Therefore K is DN per Poisson electron; its inverse is a derived e/DN gain within that model. `sigGs` is Gaussian read standard deviation in DN. `sigTL` is Tukey-lambda *scale*, not its standard deviation. `sigR` is row-noise standard deviation in DN. The rows below are released calibrated coefficients, not independently verified per-ISO raw PTC measurements.

| ISO | K DN/e | derived e/DN | Gaussian read sigma DN | Tukey-lambda scale DN | lambda | row sigma DN |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 0.09563 | 10.456970 | 1.0067395 | 0.70181876 | 0.14875287 | 0.1391465 |
| 200 | 0.19126 | 5.228485 | 1.2926387 | 0.8117464 | 0.07902429 | 0.22815849 |
| 400 | 0.38252 | 2.614242 | 2.0595572 | 1.1816813 | 0.0222538 | 0.36209714 |
| 800 | 0.76504 | 1.307121 | 3.5475867 | 1.9346539 | -0.008199721 | 0.5723769 |

All rows use `wp=16383`, `bl=512`, `q=1/16384`, `bias=0`, and separately publish uncertainty terms (`sigGssig`, `sigTLsig`, `sigRsig`, `biassig`). The exact extract is `pmn_sony_selected_iso_coefficients.json`.

All 28 ISO keys are 50,64,80,100,125,160,200,250,320,400,500,640,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000,12800,16000,20000,25600. The supplied gain values are exactly `K=0.0009563*ISO`: 28 table rows are not 28 independently measured flat-field PTC slopes. Read-distribution parameters vary by ISO, including a high-ISO gain/read regime change. Other functions offer continuous fits in log-gain space. Keep the distinction between a published fitted noise model and original calibration observations.

`wp` is the raw code white normalization/clip point. These assets do **not** independently measure photocharge full-well capacity or identify the physical saturation mechanism. `(wp-bl)/K` is only a derived ADC headroom in model electron units, not a measured full-well capacity. No CFA-color throughput, absolute quantum efficiency, wavelength sensitivity, pixel response non-uniformity, lens PSF, or same-body RGB transmission measurement is present. White balance/color matrices in scene metadata are not substitutes for such measurements.

### Numeric dark shading maps: main is usable, TPAMI tree omits them

These four main-branch files returned HTTP 200. Only the first 256 response bytes were read, enough to verify the NPY header and response size. Each is `float64` with shape `(2848,4256)` and `Content-Length=96,968,832` bytes (~92.5 MiB); all four are 387,875,328 bytes (~369.9 MiB). The arrays themselves were **not** downloaded:
- https://raw.githubusercontent.com/megvii-research/PMN/main/resources/darkshading_lowISO_k.npy
- https://raw.githubusercontent.com/megvii-research/PMN/main/resources/darkshading_lowISO_b.npy
- https://raw.githubusercontent.com/megvii-research/PMN/main/resources/darkshading_highISO_k.npy
- https://raw.githubusercontent.com/megvii-research/PMN/main/resources/darkshading_highISO_b.npy

To freeze reproducibility, replace `main` in those raw URLs with the verified `d207d3eb62e3a5106861992c0562054adf9a9c70` commit. Saved `small_prefix_headers.json` records the successful tested main URLs, sizes, and NPY headers.

The small main `darkshading_BLE.pkl` is downloadable (HTTP200,1318B): https://raw.githubusercontent.com/megvii-research/PMN/main/resources/darkshading_BLE.pkl . It was downloaded and statically disassembled with `pickletools`, never unpickled. The reader uses a low/high-ISO branch (threshold ISO1600), a deterministic fitted map of the form `ds_k*ISO + ds_b + BLE`, and optional exposure-dependent BLE in TPAMI. This gives fixed-pattern/dark-shading calibration maps, not a stochastic bank of original independent dark frames. Original raw flat stacks and raw calibration dark-frame stacks are not in the inspected GitHub resource tree. Do not describe fitted maps as a residual dark bank or use them to claim original PTC measurements.

The TPAMI `resources/SonyA7S2` tree contains one BLE pickle and four PNG illustrations, but no numerical `k/b.npy` files. Exact expected TPAMI `.npy` URLs returned404. Main’s actual `.npy` maps avoid that gap, and main’s same coefficient table avoids cross-branch mixing. TPAMI BLE (47,317B) is separately saved for provenance only.

## PMN/LRID smartphone scope

Official TPAMI README: https://raw.githubusercontent.com/megvii-research/PMN/TPAMI/README.md

It identifies LRID training/evaluation as IMX686; ISO100 bright reference and ISO6400 low-light data, raw white1023 and black64. Its `get_camera_noisy_params_max` contains explicit `IMX686_100` and `IMX686_6400` coefficients, plus another function with different calibration values; these branches should not be silently merged. Camera-body model, sensor full-well, and transmission are not resolved by the inspected README.

Small actual numeric files fetched:
- https://raw.githubusercontent.com/megvii-research/PMN/TPAMI/resources/IMX686/noiseparam-iso-100.h5 : HTTP200,46,592B, saved `pmn_IMX686_iso100.h5`.
- https://raw.githubusercontent.com/megvii-research/PMN/TPAMI/resources/IMX686/noiseparam-iso-6400.h5 : HTTP200,54,272B, saved `pmn_IMX686_iso6400.h5`.
- https://raw.githubusercontent.com/megvii-research/PMN/TPAMI/resources/IMX686/BLE_t.pkl : HTTP200,219B, saved and statically disassembled only.

HDF5 content was not parsed because local h5py was absent; claiming exact internal quantities beyond file availability would require inspecting them with a trusted HDF5 reader. LRID is linked on Baidu: https://pan.baidu.com/s/1fXlb-Q_ofHOtVOufe5cwDg?pwd=vmcl ; README reports full LRID_raw523GB, training/evaluation LRID185.1GB, results19.92GB and metrics59KB. No bulk fetch or Baidu access check was performed. Dataset rights/access restrictions beyond the repository license remain unverified.

## Noise Flow / SIDD: smartphone noise model, not physical electron/full-well calibration

Official repository: https://github.com/BorealisAI/noise_flow
Verified master commit: `48d1e077a07ca702a03f292df2e23dbf95e07b4b`.
- README: https://raw.githubusercontent.com/BorealisAI/noise_flow/master/README.md : HTTP200.
- Actual coefficient file: https://raw.githubusercontent.com/BorealisAI/noise_flow/master/cam_iso_nlf_all.txt : HTTP200,614B, saved.
- Sampling example: https://raw.githubusercontent.com/BorealisAI/noise_flow/master/sample_noise_flow.py : HTTP200,4390B, saved.
- Source raw metadata reader: https://raw.githubusercontent.com/BorealisAI/noise_flow/master/sidd/raw_utils.py : HTTP200,6202B.
- Full learned checkpoint tree: https://github.com/BorealisAI/noise_flow/tree/master/models/NoiseFlow/ckpt : HTTP200; files are model.ckpt.best.data-00000-of-00001(10884B), index(5979B), meta(26,199,564B). Only256B prefixes were checked for size/status; no checkpoint runtime was validated.
- License: https://raw.githubusercontent.com/BorealisAI/noise_flow/master/LICENSE : CC BY-NC-SA4.0, verified; noncommercial/share-alike constraints apply to this repository. Upstream SIDD dataset terms were not independently verified because the official YorkU host was blocked here.

The actual NLF CSV has 22 camera/ISO rows: IP=100,400,800,1600; GP=100,400,800,1600,3200; S6=100,400,800,1600,3200; N6=100,400,800,1600,3200; G4=100,400,800. Codes correspond to Apple/iPhone, Google/Pixel, Samsung/S6 Edge, Motorola/Nexus6, LG/G4 smartphone cameras. The official metadata reader explicitly maps manufacturers Apple,Google,samsung,motorola,LGE to those five camera IDs. The exact iPhone generation is not established by the inspected repository sources; do not assume it from the short code.

The source forms noise standard deviation as `sqrt(nlf0*clean + nlf1)` and reads NLF from RAW metadata; these beta1/beta2 values are normalized raw variance coefficients, not a per-ISO electron gain/read/full-well/white-level calibration. Noise Flow learns a conditional residual distribution on SIDD clean/noisy pairs and conditions on ISO/camera. Its example filters ISO100,400,800,1600,3200 and calls the bundled pretrained model at sampling temperature0.6. Scene-data pairing, code normalization, and calibration metadata conventions must be preserved; arbitrary applying the NLF or learned model to a DSLR is unsupported.

Required SIDD_Medium_Raw download is linked by README through http://bit.ly/2kHT7Yr and the official dataset site https://www.eecs.yorku.ca/~kamel/sidd/ . Neither actual dataset bytes nor dataset license were retrieved here. No DSLR/APS-C/MFT physical PTC scope is established by SIDD.

## SFRN supplemental calibrated coefficients

Official repository: https://github.com/zhangyi-3/noise-synthesis
- README: https://raw.githubusercontent.com/zhangyi-3/noise-synthesis/main/README.md : HTTP200,4320B.
- Actual source parameters: https://raw.githubusercontent.com/zhangyi-3/noise-synthesis/main/synthesize.py : HTTP200,2691B, saved.
- LICENSE: https://raw.githubusercontent.com/zhangyi-3/noise-synthesis/main/LICENSE : MIT, verified.

Source lists camera/ISO sets: iPhone100,200,400,800,1600,2000; SamsungS6Edge100,200,400,800,1600,3200; GooglePixel100,200,400,800,1600,3200,6400; SonyA7S2 800,1600,3200; NikonD850800,1600,3200. It publishes Poisson and read constants, but its current mobile `cread` values are passed directly as `normal_` standard deviation despite the small normalized values and ambiguous convention. Do not silently treat all such numbers as verified physical read variance or mix mobile normalized and DSLR DN conventions. Source does not bundle original measurement flats/dark banks, electron full-well calibration, or CFA throughput. PMN’s explicit source units/normalization are the stronger immediately actionable calibration route.
