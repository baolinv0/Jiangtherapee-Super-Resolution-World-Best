# Official Zemax sources and a prospective optical-only PSF export route

Checked 2026-10-05 by read-only HTTPS. No OpticStudio installation, license use, ray trace, `.zmx`/`.zos` lens download, or numerical PSF export was performed.

## Verified official source and version

Official repository: <https://github.com/ansys/lib-zemax-programming>.
Inspected commit: `88a35ec4764bb3042cdf35fd41d444e0d5d7679e`.
Eight small pinned text sources returned HTTP 200. `verified_sources.json` records their URLs, sizes, and SHA-256 hashes; `pinned_*` files preserve the retrieved contents.

The pinned [README](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/README.md) says:

> “The Ansys Zemax Programming Library centralizes programming solutions provided by the Ansys Zemax team.”

> “This repository maintains boilerplate code, sample code shipped with OpticStudio, and knowledgebase examples for the five ZOS-API-supported languages.”

Its dependencies include:

> “An installed and licensed copy of Zemax OpticStudio (any version)”

> “A programming language with .NET or COM capabilities (required by the ZOS-API)”

## Lens candidates: source references, not downloaded prescriptions

| Candidate | Verified official source and exact evidence | Availability established here |
|---|---|---|
| Cooke triplet/objective | [Python example04](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_04_pull_data_from_FFTMTF.py#L194): `testFile = sampleDir + '\\Sequential\\Objectives\\Cooke 40 degree field.zmx'` | Installed-sample filename and location; numerical lens surface prescription not retrieved. |
| Double Gauss objective | [Python example22](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_22_seq_spot_diagram.py#L179): `file = "Double Gauss 28 degree field.zmx"`, then loads `SamplesDir + "\\Sequential\\Objectives\\" + file` | Installed-sample filename and location; prescription not retrieved. |
| Double Gauss course design | [Python example15](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_15_Seq_Optimization.py#L176): loads `\\Short course\\sc_dbga1.zmx`, falling back to `\\Short course\\Optical System Design Using OpticStudio\\sc_dbga1.zmx`; prints `Double Gauss Design:` | Two version-dependent installed sample paths; prescription not retrieved. |
| Simple N-BK7 singlet | [Python example01](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_01_new_file_and_quickfocus.py#L186) builds a new system: `ApertureValue = 40`, `AddField(0, 5.0, 1.0)`, wavelength preset `d_0p587`, front radius `100.0`, glass thickness `10.0`, `Material = 'N-BK7'`; rear radius receives an `FNumber` solve with `FNumber = 10`, followed by quick focus. | Public MIT code specifies a reproducible construction. Rear radius/image distance depend on solving in licensed OpticStudio; this is not an already exported numerical prescription. |

Sample directory access is explicitly `TheApplication.SamplesDir`; the public repository contains the example code, not evidence that these commercial installed lens files have a public redistribution license. The simple singlet is a useful transparent test case, but has less camera-system realism than the objectives.

## Verified API behavior and boundaries

1. [Example01, lines 75–80](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_01_new_file_and_quickfocus.py#L75) calls `CreateNewApplication()` and checks `IsValidLicenseForAPI`; its exact error is `"License is not valid for ZOSAPI use"`. The example imports `winreg` and locates installed .NET libraries. This is a future licensed Windows workflow, not a demonstrated Linux-cloud capability.
2. [C# SampleExtension1, lines 254–267](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/c_sharp/SampleExtension1.cs#L254) says `// Create a Huygens PSF`, calls `analysisZemax.New_HuygensPsf()`, casts settings to `IAS_HuygensPsf`, sets `ImageSampleSize` and `PupilSampleSize` to `_128x128`, selects `FalseColor`, and applies the analysis. This verifies an official Huygens-PSF analysis entry point, not array/grid export.
3. [Example12](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_12_SEQ_SystemExplorer.py#L175) exposes `SystemData.Wavelengths`, `SystemData.Fields`, field type, field coordinates, and surface thickness. [Example22, lines 276–277](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_22_seq_spot_diagram.py#L276) uses `Field.SetFieldNumber(0)` and `Wavelength.SetWavelengthNumber(0)` for a **spot diagram**. These are not verified PSF-specific selector signatures.
4. [Example04, lines 211–233](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/examples/python/PythonStandalone_04_pull_data_from_FFTMTF.py#L211) runs `ApplyAndWaitForCompletion()`, gets results, and reads `GetDataSeries`, `XData.Data`, and `YData.Data` for **FFT MTF**. It is not evidence of `GetDataGrid` or numerical Huygens-PSF export. A helper comment mentions FFT PSF, but is insufficient to establish a working exporter.

## Code licensing versus lens/output licensing

The pinned [LICENSE](https://github.com/ansys/lib-zemax-programming/blob/88a35ec4764bb3042cdf35fd41d444e0d5d7679e/LICENSE) is MIT; each inspected example also says `SPDX-License-Identifier: MIT`. Exact license conditions include:

> “The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.”

This supports reuse of the inspected public code with its notices. It does not establish redistribution permission for installed OpticStudio `.zmx` prescriptions, proprietary glass catalogs, commercial software, or outputs generated under a particular license. No explicit public license for those installed lens prescriptions was verified. The repository's separate [LegalNotice](https://raw.githubusercontent.com/ansys/lib-zemax-programming/main/LegalNotice) contains general commercial-product licensing text, so it must not be silently conflated with the code's explicit MIT header.

## Future license-holder export plan (not executed or verified)

Use the user's 61 spectral bins individually, with one **monochromatic optical intensity PSF per wavelength** for each fixed lens × field × aperture/f-number × physical defocus condition. Do not substitute one spectrally averaged PSF. Huygens PSF is a prospective optical-only route; verify the licensed model and settings exclude pixel-area integration, pixel-response functions, sensor microlenses, CFA/spectral QE, and sensor noise. An image sampling grid alone must not be labeled a sensor aperture. Any sensor-inclusive kernel should be separately documented and never mixed into the optical-only set.

On a licensed Windows machine:

1. Load a permitted lens (or construct the public singlet), record the OpticStudio release, resolved prescription, glass/catalog versions, units, coordinate convention, and permission status. Preserve the fixed lens geometry while changing pupil/stop aperture. **The example01 rear-radius FNumber solve changes the prescription and is unsuitable as a fixed-lens aperture sweep.**
2. Establish a reference focus once, then apply recorded physical image-plane offsets. Do not run autofocus separately at every requested defocus, as that would erase the perturbation. Define field positions and the 61 exact wavelengths with no unintended spectral weighting/aggregation.
3. Run the Huygens PSF for one field/wavelength, verify the installed release's PSF-specific selectors, data-grid or text-export API, array axis order, physical x/y spacing/origin, intensity normalization, centroid alignment, and throughput. **No verified numeric PSF-grid export call is supplied here.** Confirm it against that release's local ZOS-API reference and compare an exported example with OpticStudio's displayed/text result before batching.
4. Repeat over the requested condition grid and export numerical arrays plus metadata. Preserve raw intensity/throughput separately from sum-normalized kernels. Check pupil/image sampling convergence, wavelength dependence, and a centered near-Airy sanity case. Apply sensor aperture/QE/noise only in a separate downstream sensor model.

## Access restrictions and linked documentation

`https://optics.ansys.com`, `https://www.ansys.com`, `https://support.zemax.com`, `https://community.zemax.com`, `https://help.ansys.com`, `https://ansyshelp.ansys.com`, and `https://ansys.github.io/optical-automation/` probes failed with `Tunnel connection failed: 403 Forbidden`. `api.github.com` also failed; `github.com` and `raw.githubusercontent.com` worked. These are observed proxy/network restrictions, not proof that source pages or licenses do not exist. TLS verification remained enabled; no credentials, allowlist changes, or installation were attempted.

The verified official README links [Getting started with ZOS-API](https://optics.ansys.com/hc/en-us/articles/42661816179731--Tutorial-Series-Getting-Started-with-ZOS-API). Its page content was not accessible here and is not quoted as verified. Outstanding requirements are a licensed installation, prescription/output-sharing rights, and verification of numerical optical-only PSF export in the installed version.
