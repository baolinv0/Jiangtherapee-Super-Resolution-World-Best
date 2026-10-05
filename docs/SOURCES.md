# JSR public evidence audit

Audit date: 2026-10-05 (Asia/Shanghai). Upstream and fork snapshot verified through GitHub: `6ab0f5bf76d2ccfbf7e21798cc575232ef9b272b`. This audits public sources; it does not independently reproduce reported quality.

## Evidence and limits

| Item | Status | Reproduction consequence |
|---|---|---|
| Purpose and data flow | Author statement in current upstream README | Black-subtracted RAW → alignment → color/phase Tap statistics → learned Controller fusion plus legacy RGB → amplitude-normalized RefineNet → linear RGB. White balance/tone mapping are downstream. |
| Implementation | Public core and K4 weights inspected | `Core-Only-Source-Code` is visible. README's old no-source sentence conflicts with the tree. Use the actual code for implementation details. |
| Intensity equivariance | Conditional author claim | Requires fixed correspondence/LCA transforms, correct black subtraction, no clipping, and scaling the whole dependency support. Not an unconditional capture/alignment property. |
| Frame count | README internally inconsistent | Intro advertises 4–14; detailed text also mentions 3–6. Determine actual core constraints; do not silently duplicate frames. |
| Original data, losses and schedule | Not verified in inspected material | Do not label inferred training choices as original JSR. Full paper is still announced for future release. |
| Nine-stop result | Author-reported constrained experiment | Editing gain is 512, not nine extra sensor DR stops. |
| Overall superiority | Not established here | Screenshots/stress tests do not establish universal or smartphone-pipeline leadership. |

Primary: <https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best>

Core: <https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/tree/main/Core-Only-Source-Code>

Fork: <https://github.com/baolinv0/Jiangtherapee-Super-Resolution-World-Best>

## GitHub and Zhihu attribution

- Subsequent evidence: the user supplied the complete speech “SR相机手持超分辨率16bitraw合成讲稿”, attributed to 姜尧耕, edited2026-10-03. This supports author-statement attribution as supplied text, not independent retrieval/authentication of a Zhihu page. The earlier search limitation below remains a URL discovery limitation. [Speech conformance and new implementation](TRANSFORMER.md) distinguish the seven-frame Transformer description from publicv9.8 Controller/RefineNet. Rechecking upstream main on2026-10-05 still returned6ab0f5b; no original Transformer training/calibration library was established.

- Inspected GitHub issues list and opened [issue 1](https://github.com/y-g-jiang/Jiangtherapee-Super-Resolution-World-Best/issues/1). No training recipe was established from them.
- [Author homepage](https://y-g-jiang.github.io/) identifies the public name 姜尧耕; it is a primary author site, not Zhihu.
- Searched `site:zhihu.com` / `site:zhuanlan.zhihu.com` with Jiangtherapee, JSR, 姜尧耕 and 渔樵耕牍. No verifiable JSR-specific Zhihu URL and full article were retrieved. Homepage HTML yielded no direct Zhihu URL. This is a discovery/access limitation, not proof that such statements do not exist.
- Do not cite third-party Nikon news recaps or search snippets as JSR training evidence, or invent a Zhihu address. Future direct links/text can extend this audit without blocking a documented engineering reproduction.

## Related original sources

| Source | Confirmed information | Implication (our choice, not JSR evidence) |
|---|---|---|
| DBSR, CVPR 2021 | Official toolkit synthesizes training bursts from Zurich RAW-to-RGB Canon RGB training split. Fixed validation has 300 bursts of 14 RAW frames from the source test split. Real BurstSR pairs Galaxy S8 bursts with DSLR targets; scoring aligns space/color. | Split by source scene before crops. Keep synthetic linear scoring separate from real aligned metrics. |
| BSRT, CVPRW 2022 | Separate synthetic training and real fine-tuning; example uses 14 frames, Swin features and flow-guided deformable alignment. | Baseline reference, not JSR's recipe. |
| Burstormer, CVPR 2023 | Multiscale feature alignment, reference enrichment and progressive fusion; official implementation available. | Burst restoration comparison; does not identify JSR training. |
| BurstM, ECCV 2024 | Official repo uses Zurich synthetic training followed by BurstSR fine-tuning; Fourier coefficients plus optical flow; multiscale output. | Match physical output size, K, color space and alignment access. |
| HDR+, SIGGRAPH Asia 2016 | Constant short-exposure Bayer bursts protect highlights; alignment/merging reduce shadow noise before tone mapping. | Equal exposure bursts can support HDR. Reconstruction does not reproduce capture control/TM. |
| Handheld Multi-Frame SR, TOG 2019 | Direct CFA burst-to-RGB exploits natural subpixel hand motion. | Appropriate physical multiframe baseline concept. |
| HDR+ dataset | Official introduction reports 3,640 bursts / 28,461 frames plus intermediate/rendered HDR+ outputs. | Real RAW robustness set; pipeline output is not independent clean HR linear GT. |

Sources:

- DBSR [official code](https://github.com/goutamgmb/deep-burst-sr), [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhat_Deep_Burst_Super-Resolution_CVPR_2021_paper.pdf), [Zurich loader](https://github.com/goutamgmb/NTIRE21_BURSTSR/blob/master/datasets/zurich_raw2rgb_dataset.py).
- [BSRT official code](https://github.com/Algolzw/BSRT).
- Burstormer [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper.html), [official code](https://github.com/akshaydudhane16/Burstormer).
- [BurstM official code](https://github.com/Egkang-Luis/BurstM).
- [HDR+ 2016](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/), [Handheld MFSR 2019](https://research.google/pubs/handheld-multi-frame-super-resolution/).
- [HDR+ dataset introduction](https://research.google/blog/introducing-the-hdr-burst-photography-dataset/), [dataset details](https://www.hdrplusdata.org/dataset.html), [later exposure bracketing](https://research.google/blog/hdr-with-bracketing-on-pixel-phones/).

## Proposed engineering defaults — explicitly inferred

1. Start with licensed local HR RGB, preferably the public Zurich training split. Document conversion to a linear proxy. Inverse sRGB alone does not undo unknown tone mapping, denoising, sharpening or camera colors; do not call it measured camera RAW. Also allow already linear RGB.
2. Synthesize reference-space linear GT, subpixel motion, optical/pixel integration blur, RGGB and shot/read noise. Record seed, source hash, CFA phase, black/white levels, motion sign/units, sampling ratio and every degradation parameter. Ranges are our choices.
3. Begin with static scenes and known translations to isolate backend reconstruction. Call this oracle geometry. Estimated alignment is a separate condition. Treat affine motion, LCA, clipping and moving objects as explicit stress conditions.
4. Distinguish native mosaic H×W, packed Bayer H/2×W/2 and native 2× JSR output. Publish shape contracts; compare physical scale rather than library naming.
5. Document a robust linear loss; label any color-difference/gradient/amplitude terms as inferred. Save config, RNG, optimizer, scheduler and validation metrics in checkpoints.
6. Deterministic procedural data is for smoke/CI only. Generalization needs withheld real scenes and a full dataset run. Do not report smoke learning as reproduced paper quality.
7. Verify zero input, finite output, gradients, unclipped gain equivariance, alignment sign, CFA colors, K variation, checkpoint reload/resume and tile/full equivalence with enough halo. Report fixed-range linear PSNR, color-difference RMSE, shadow response and artifact probes. Do not fit per-model gain/color when testing radiometric accuracy.
8. Keep downloads optional and link original terms; do not redistribute datasets without checking licenses. Real BurstSR needs official spatial/color alignment metrics.

Suggested README wording: “This is an independent engineering reproduction of the public JSR core and described principles. The original complete training recipe and paper are not available in the audited snapshot; our synthetic data generation, losses and schedule are documented assumptions. Included smoke tests establish execution and numerical properties, not the original paper's quality or a state-of-the-art ranking.”
