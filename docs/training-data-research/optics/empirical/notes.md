# Empirical camera PSFs and Zemax: verified primary-source notes

Checked 2026-10-05 with read-only HTTPS and normal TLS verification. No packages, models, lens binaries, or bulk datasets were installed/downloaded. Cached small source pages/text are in this directory. Successful URLs below were actually fetched; external dataset contents and compatibility were not verified.

## 1. Nonparametric calibration from a random target: effective camera PSF

Primary author implementation: https://github.com/mdelbra/psf-estim (observed commit `192ef6aa545e68a77bd90220c0fc666aef11564e`). Verified README: https://raw.githubusercontent.com/mdelbra/psf-estim/master/README.txt

Exact quotes:

> "there is a permanent intrinsic physical camera blur due to light diffraction, sensor resolution, lens aberration, and anti-aliasing filters. Our goal here is to accurately estimate the Point Spread Function - PSF, that models the intrinsic camera blur."

> "theoretical bounds show that a near-optimal accuracy can be achieved by taking a single snapshot of a calibration pattern mimicking a Bernoulli(0.5) white noise."

> "accurate geometric registration and radiometric correction; Once these procedures have been applied, the local PSF can be directly computed by inverting a linear system that is well-posed and consequently its inversion does not require any regularization or prior model."

> "<output file> : Output PSF written into a TXT file as a k x k matrix of floats"

The source supports a concrete local, subpixel, nonparametric calibration method, with a provided sample capture/pattern. It does **not** provide a measured multi-lens, multi-aperture, monochromatic PSF library. The stated estimation target explicitly includes sensor response and anti-aliasing effects: do not label this output optical-only or independently apply the sensor pixel aperture again. Camera sampling/model separation would require additional calibration and identifiability assumptions.

README code license is GNU Affero GPL v3 or later. Existing implementation supports 8/16-bit PGM and requires FFTW3, CBLAS, LAPACK. No installation was attempted. RGB-CFA and narrowband acquisition would need an explicit extension/calibration pipeline.

## 2. Learning Lens Blur Fields: measured camera data + learned representation

Official author repository: https://github.com/estherlin/learning-lens-blur-fields (observed commit `77f0a8db1f33cafae40b54f619dc030bc699055e`). Verified README: https://raw.githubusercontent.com/estherlin/learning-lens-blur-fields/main/README.md

Exact quotes:

> "This repository contains the official implementation of Learning Lens Blur Fields, published in IEEE TPAMI 2025. This work introduces a high-dimensional neural representation of blur—the lens blur field—and a practical method for acquiring it from real-world camera systems."

> "You can download data for both iPhone 12 Pro wide cameras used in the paper"

> "4x downsampled processed iPhone 12 Pro wide lens data (1.06GB)"

> "Full resolution processed iPhone 12 Pro wide lens data (18.3GB)"

> "Unprocessed calibration pattern captures (1.75GB)"

> "This branch contains development code that is not yet cleaned up. The full project has not been maintained since 2023."

Author-linked data endpoint (blocked by CONNECT 403 here; no data downloaded): https://drive.google.com/drive/folders/1zf2p2Bj_Jxhq4-smq1AsnEwlRfKJH0qC?usp=sharing

Verified code evidence:

- https://raw.githubusercontent.com/estherlin/learning-lens-blur-fields/main/train.py: `--dims` defaults to `xyuvp`; `generate_pts` builds two kernel coordinates and appends sensor position and `lens_positions`; training uses `rgb_bayer = bayerfy(rgb)` and `loss = F.mse_loss(rgb_bayer, blur_cut)`.
- https://raw.githubusercontent.com/estherlin/learning-lens-blur-fields/main/util/renderer.py: model has `n_input_dims=5`, `n_output_dims=4`; RGB conversion explicitly stacks `(R, G, G, B)`.
- https://raw.githubusercontent.com/estherlin/learning-lens-blur-fields/main/util/preprocess.py: `load_data(DATA_DIR, distances, patterns, lens_positions, burst_size)` loads `raw.raw_image` and splits four Bayer positions. The stored capture data can span distances, but the reviewed five-coordinate model is **not** an independent wavelength/aperture/object-depth/focus tensor.
- The live `models/` tree lists `iphone12pro0-wide.pth` and `iphone12pro1-wide.pth`; models were not downloaded or evaluated.

Practical implication: strong acquisition/reference source for real phone-camera field/focus blur, but the verified release is four camera-CFA channels, not 61 monochromatic wavelengths. No f-number sweep, exact independent depth/focus mapping, or separation of sensor pixel integration was established. Conservatively treat it as an effective captured-camera blur model unless the paper/supplement or calibration proves otherwise. It cannot be asserted directly compatible with an optical-only spectral PSF loader.

The repository LICENSE is MIT (verified https://raw.githubusercontent.com/estherlin/learning-lens-blur-fields/main/LICENSE), but this alone does not establish the external Drive dataset's independent redistribution terms. README references `notebooks/visualizing_blur_fields.ipynb`, which returned 404; live tree instead contains `psf_comparison_grid.ipynb` (currently an effectively empty file). The training shell references `scripts/train5d.py` whereas README lists `train.py`; no runnable-readiness claim is supported.

## 3. Eboli et al. ECCV 2022: measured PSF collection distinguished from Gaussian approximation

Official repository: https://github.com/teboli/fast_two_stage_psf_correction (observed commit `fbe8e24097900e15d8ce2a165cc9e9958708805e`). Verified README: https://raw.githubusercontent.com/teboli/fast_two_stage_psf_correction/main/README.md

Exact README quote:

> "Download the DIV2K dataset and the PSFs at https://edmond.mpdl.mpg.de/file.xhtml?fileId=101784&version=1.0"

Official author's project HTML source verified via GitHub (served project hostname was blocked): https://raw.githubusercontent.com/teboli/teboli.github.io/master/fast_optical.html

Exact abstract quote:

> "Based on the measurements of the PSFs of dozens of lenses, these blur kernels are modeled as RGB Gaussians defined by seven parameters."

This distinction matters: the measured collection and the seven-parameter Gaussian approximation are different evidence/assets. The code can consume stored nonparametric RGB kernels; do not describe a generic Gaussian sweep as measured optical PSFs.

Verified parser https://raw.githubusercontent.com/teboli/fast_two_stage_psf_correction/main/fast_optics_correction/utils_psf.py includes:

```python
##### Inspired by Mathias Bauer's code from his ICCP18 paper
self.W = 8712  # Canon EOS 5DSR sensor size
self.H = 5808  # Canon EOS 5DSR sensor size
C = loadmat(psf_path)['C']
locations = np.stack([Y, X], axis=-1)
# example filename:
psf_path = './psfs/Canon_EF24mm_f_1.4L_USM_ap_1.4.mat'
```

It reads locations/exposures/RGB kernels and normalizes each channel. The example filename supports at least lens/aperture metadata in filenames, but full coverage/count/aperture range/depth coverage/spectral bandpasses/sensor-aperture separation could not be verified because the Edmond record was blocked by CONNECT 403. External dataset license was likewise not checked. Code is MIT (verified repository LICENSE), which should not be extended automatically to the linked dataset.

## 4. Minimal practical acquisition plan (recommendation, not executed)

For a true lens-specific optical-only library, use a known, reproducible lens prescription and a licensed wave-optics simulator; export monochromatic PSFs on the chosen sensor-plane pitch over an explicit field x/y grid, wavelength grid, iris settings, object distances, and focus settings. Keep the detector pixel aperture outside that optical-only PSF. Zemax evidence/constraints are documented separately below.

For an empirical camera library, lock focus/aperture/exposure/RAW processing; acquire a registered calibrated random target or point-source grid at each field/depth/focus/iris setting, with dark/flat repeats and measured narrowband illumination. RGB CFA responses cannot substitute for independent monochromatic wavelengths. Specify projected target size, pixel pitch, coordinate orientation, source bandwidth, spectral normalization and focus/depth convention. Validate interpolation using held-out captures, and compare centroid/chromatic offsets, encircled energy and directional MTF. Treat estimated camera kernels as including the detector/OLPF unless these components have been separately characterized and safely removed. Use a separate effective-camera-PSF representation rather than falsely declaring optical-only compatibility.

## Access limitations

`github.com` public repository HTML and `raw.githubusercontent.com` succeeded. `api.github.com`, `arxiv.org`, `drive.google.com`, `edmond.mpdl.mpg.de`, `teboli.github.io`, `cs.ubc.ca`, `uni-tuebingen.de`, Google/Bing, `optics.ansys.com` and `www.ansys.com` probes were blocked with proxy CONNECT 403. This is an observed access restriction, not evidence that those sources or data do not exist. No TLS verification bypass, credential request, or network configuration change was used.

## 5. Official Zemax evidence and license-holder route

Full verified Ansys-source findings: [zemax/notes.md](zemax/notes.md); pinned-source status/hashes: [zemax/verified_sources.json](zemax/verified_sources.json).

Verified official repository: https://github.com/ansys/lib-zemax-programming at commit `88a35ec4764bb3042cdf35fd41d444e0d5d7679e`. README calls its contents "sample code shipped with OpticStudio" and requires "An installed and licensed copy of Zemax OpticStudio (any version)". API example code is MIT; this does not grant a public license to installed sample lens prescriptions. Official examples refer to `Cooke 40 degree field.zmx` and `Double Gauss 28 degree field.zmx` in `SamplesDir/Sequential/Objectives`, and construct an explicit N-BK7 singlet. No sample lens binaries were retrieved. Official C# code calls `New_HuygensPsf()` with 128x128 pupil/image sampling, establishing a wave-optics entry point. Numeric PSF-grid export is **not verified** by the inspected examples (MTF DataSeries export is not PSF DataGrid export).

The notes give a prospective licensed Windows workflow: one monochromatic optical intensity PSF per requested wavelength/field/aperture/defocus, separate optical and sensor operators, local API export verification before batching, convergence/normalization/physical-axis checks. Preserve lens geometry for aperture sweeps: the public singlet's rear-radius FNumber solve changes the lens prescription, so it cannot serve as an iris sweep for one fixed lens. Preserve a reference focus then perturb physical image-plane/object depth explicitly; per-condition autofocus would remove the desired defocus.
