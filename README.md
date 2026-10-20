<h1 align="center">Long-Tailed 3D Detection<br>via Multi-Modal Late-Fusion</h1>

<p align="center">
  <strong>International Journal of Computer Vision (IJCV)</strong><br>
  Accepted for publication
</p>

<p align="center">
  <strong>Yechi Ma</strong><sup>1,&dagger;</sup> &nbsp;
  <strong>Neehar Peri</strong><sup>2,&dagger;</sup> &nbsp;
  <strong>Achal Dave</strong><sup>3</sup><br>
  <strong>Wei Hua</strong><sup>1</sup> &nbsp;
  <strong>Deva Ramanan</strong><sup>4</sup> &nbsp;
  <strong>Shu Kong</strong><sup>5,*</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University &nbsp;&middot;&nbsp;
  <sup>2</sup>Caltech &nbsp;&middot;&nbsp;
  <sup>3</sup>Toyota Research Institute<br>
  <sup>4</sup>Carnegie Mellon University &nbsp;&middot;&nbsp;
  <sup>5</sup>University of Macau<br>
  <sub>&dagger; Equal contribution &nbsp;&middot;&nbsp; * Corresponding author</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2312.10986v6"><strong>Paper</strong></a> &nbsp;|&nbsp;
  <a href="https://cc50121.github.io/lt3d-lf-io/">Project Page</a> &nbsp;|&nbsp;
  <a href="https://github.com/cc50121/lt3d-lf/releases/tag/ijcv-reproducibility-v1">Code &amp; Models</a> &nbsp;|&nbsp;
  <a href="docs/reproducibility.md">Reproduction Guide</a> &nbsp;|&nbsp;
  <a href="https://cc50121.github.io/lt3d-lf-io/#video">Video</a>
</p>

## Overview

**LT3D** evaluates all 18 annotated nuScenes classes, including rare but
safety-critical objects such as `child`, `stroller`, and `emergency_vehicle`.
**MMLF** combines independently trained LiDAR and RGB detectors through
image-plane association, score adjustment, and probabilistic fusion.

<p align="center">
  <img src="assets/lt3d_long_tail_overview.png" width="100%" alt="Long-tailed class distribution and examples in nuScenes">
</p>

<details>
<summary><strong>Read the abstract</strong></summary>

Contemporary autonomous vehicle benchmarks have significantly advanced
multimodal LiDAR and RGB 3D detection. However, despite the naturally
long-tailed distribution of object classes, existing benchmarking protocols
primarily focus on a small set of frequent categories, largely overlooking
rare but safety-critical classes. We formalize this problem as Long-Tailed 3D
Detection (LT3D), where evaluation encompasses all annotated classes. We
introduce hierarchical losses that promote feature sharing across classes,
diagnostic metrics that assign partial credit to semantically reasonable
mistakes with respect to the semantic hierarchy, and a Multi-Modal Late-Fusion
(MMLF) framework. MMLF combines independently trained LiDAR and RGB detectors,
allowing advanced unimodal models and additional unimodal data to be used
without paired LiDAR-RGB training. We examine three fundamental design choices:
the RGB detector representation, cross-modal association space, and fusion
strategy. Our experiments show that 2D RGB detectors recognize rare classes
more reliably than monocular 3D detectors, image-plane association is robust to
depth errors, and probabilistic fusion of calibrated scores provides the best
performance. Experiments on nuScenes and Argoverse 2 establish a new state of
the art for LT3D.

</details>

## Results

### nuScenes LT3D Benchmark

| Method | Modality | All mAP | Many | Medium | Few |
|:---|:---:|---:|---:|---:|---:|
| CenterPoint | L | 39.2 | 76.4 | 43.1 | 3.5 |
| CMT | C+L | 44.4 | 79.9 | 53.0 | 4.8 |
| CCF | C+L | 47.1 | 79.8 | 56.2 | 9.2 |
| FOMO-3D | C+L | 54.6 | 79.9 | 59.6 | 27.6 |
| **MMLF (CenterPoint + Co-DINO)** | **C+L** | **57.0** | **80.7** | **64.0** | **29.2** |

C: RGB cameras. L: LiDAR. Table values follow the paper's one-decimal rounding.
The released prediction achieves **57.04 mAP** and **62.81 NDS**.

<details>
<summary><strong>Frequency groups and detailed results</strong></summary>

| Split | Number of Classes | Mean LCA0 AP |
|:---|---:|---:|
| Many | 5 | 80.74 |
| Medium | 7 | 63.96 |
| Few | 6 | 29.21 |

- **Many:** car, truck, adult, traffic cone, barrier.
- **Medium:** trailer, bus, construction vehicle, bicycle, motorcycle,
  construction worker, pushable/pullable.
- **Few:** emergency vehicle, child, police officer, stroller,
  personal mobility, debris.

See [detailed results](docs/results.md) for per-class AP, distance breakdowns,
LiDAR-density analysis, and prediction checksums.

</details>

### Argoverse 2 Benchmark

| Method | Modality | All mAP | Many | Medium | Few |
|:---|:---:|---:|---:|---:|---:|
| CenterPoint | L | 44.0 | 77.4 | 46.9 | 30.2 |
| CCF | C+L | 49.9 | 81.8 | 52.0 | 37.6 |
| **MMLF (CenterPoint + Co-DINO)** | **C+L** | **58.6** | **93.9** | **60.7** | **45.3** |

Results use the paper's 26-class, 50 m evaluation protocol.
[Full benchmark tables](https://cc50121.github.io/lt3d-lf-io/#results)
include all compared methods and the separately marked oracle result.

## Method

A LiDAR detector provides accurate 3D localization, while a 2D RGB detector
supplies complementary semantic recognition for long-tail objects.

<p align="center">
  <img src="assets/mmlf_design_choices.png" width="100%" alt="Three design choices in MMLF">
</p>

- **Representation:** use a 2D RGB detector rather than a monocular 3D RGB detector.
- **Association:** match 2D detections and projected LiDAR boxes on the image plane.
- **Fusion:** adjust scores and combine matched predictions probabilistically,
  with class-relaxed matching and semantic relabeling.

The modular pipeline can use stronger unimodal detectors without retraining
a joint multimodal network.

## Quick Start

The RGB detector and legacy 3D/fusion code use **separate environments**.
Follow the [reproduction guide](docs/reproducibility.md#environments) for
dependency versions, GPU compatibility, data preparation, and detector training.

### 1. Download the Release

```bash
python scripts/download_release.py --extract
```

This downloads and verifies the source snapshots, model weights, and cached
LiDAR input. Co-DINO checkpoint parts are reassembled automatically.

### 2. Set the Input Paths

Download the camera detections from [Downloads](#downloads) and extract them.
In the fusion environment, set:

```bash
export ROOT=/absolute/path/to/lt3d-lf
export FRAMEWORK_ROOT="$ROOT/third_party/centerpoint"
export NUSCENES_SDK="$ROOT/third_party/nuscenes-sdk"
export DATA_ROOT=/absolute/path/to/nuscenes
export LIDAR_PREDICTIONS="$ROOT/downloads/centerpoint_fusion_input.pkl"
export CAMERA_RESULTS=/absolute/path/to/extracted/camera/s015
```

`CAMERA_RESULTS` must point to the directory containing the per-camera TXT
files. nuScenes metadata and validation infos must already be prepared.

### 3. Run Fusion and Evaluation

```bash
cd "$ROOT"
bash scripts/run_final_fusion.sh
```

The script uses the released Co-DINO calibration and threshold maps, then
evaluates the fused predictions with the 18-class LCA0 protocol.
See the [full workflow](docs/reproducibility.md) to generate detector outputs
or evaluate an existing prediction JSON.

<a id="released-artifacts"></a>

## Downloads

| Resource | Download | Contents |
|:---|:---|:---|
| Detector code and models | [GitHub Release](https://github.com/cc50121/lt3d-lf/releases/tag/ijcv-reproducibility-v1) | Detector sources, weights, LT3D SDK, and cached LiDAR input |
| Camera detections | [Download](https://drive.google.com/file/d/1NGMGuB7detRJkiHy8hbXvok6rJ2PfLUU/view?usp=drivesdk) | Per-camera 2D detection results |
| Final 3D predictions | [Part 1](https://drive.google.com/file/d/1ZHD2JIQzZpFG9cwTNrm8kUPSVUBW_z_z/view?usp=drivesdk) &middot; [Part 2](https://drive.google.com/file/d/1_Wxu0Mq3L1Hjp3vd0_wedgJuVTlOrARa/view?usp=drivesdk) &middot; [Part 3](https://drive.google.com/file/d/19Cv8Bd2XLfeXlWAbx8Pu25FMk3alM9z-/view?usp=drivesdk) | Released fusion predictions |
| Evaluation summary | [Download](https://drive.google.com/file/d/1Eq27aE-nJlVLYtpO5Mo0bJUip_7O6KRW/view?usp=drivesdk) | Saved evaluation outputs |

Checksums and provenance are documented in the
[release manifest](configs/release_assets.json),
[results](docs/results.md), and
[reproduction guide](docs/reproducibility.md#verification-and-limits).

> The historical CenterPoint checkpoint has not been independently linked to
> the final cached predictions. Use the released cached LiDAR input for the
> final fusion recipe. From-scratch training and full-dataset reruns have not
> been newly verified for this release.

<details>
<summary><strong>Reassemble the final prediction archive</strong></summary>

Linux or macOS:

```bash
cat lt3d_lf_final_3d_predictions.json.gz.part* > lt3d_lf_final_3d_predictions.json.gz
```

Windows:

```bat
copy /b lt3d_lf_final_3d_predictions.json.gz.part001+lt3d_lf_final_3d_predictions.json.gz.part002+lt3d_lf_final_3d_predictions.json.gz.part003 lt3d_lf_final_3d_predictions.json.gz
```

Compare the reconstructed archive's SHA256 with the value in
[the result documentation](docs/results.md).

</details>

## Qualitative Results

MMLF uses RGB evidence to correct semantically plausible LiDAR mistakes,
including `bus` to `truck`, `adult` to `stroller`, and `adult` to `child`,
while retaining the LiDAR detector's 3D localization.

<p align="center">
  <img src="assets/mmlf_qualitative_examples.png" width="100%" alt="MMLF qualitative relabeling examples">
</p>

[Watch the CMT / CCF / MMLF comparison video](https://cc50121.github.io/lt3d-lf-io/#video).

<details>
<summary><strong>Repository structure</strong></summary>

```text
lt3d_lf/       Fusion, calibration, evaluation, and metrics
configs/       Detector configurations and final fusion parameters
scripts/       Download, data preparation, training, and inference
environments/  Observed critical dependency versions
third_party/   Detector source snapshots and license notices
analysis/      Stratified LT3D analysis
tests/         Release-tool tests
docs/          Reproduction guide and detailed results
assets/        Paper figures
```

</details>

## Citation

The paper is accepted by IJCV. Publisher-assigned bibliographic details will
be added when available; the preprint citation is provided below.

```bibtex
@article{ma2023longtailed,
  title={Long-Tailed 3D Detection via Multi-Modal Late-Fusion},
  author={Ma, Yechi and Peri, Neehar and Dave, Achal and Hua, Wei and Ramanan, Deva and Kong, Shu},
  journal={arXiv preprint arXiv:2312.10986},
  year={2023},
  note={Accepted by International Journal of Computer Vision (IJCV)},
  url={https://arxiv.org/pdf/2312.10986v6}
}
```
