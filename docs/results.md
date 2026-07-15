# Final Results

## Best Result

| Field | Value |
|---|---|
| Detector line | Co-DINO Swin-L SWAT-Stage2 nuImages + MMLF |
| mAP | 57.04 |
| NDS | 62.81 |
| Prediction files | Hosted externally; see the download links below. |

Large prediction files are not included in this lightweight source release.
They are provided as external reproducibility artifacts:

| Artifact | Link | SHA256 |
|---|---|---|
| `lt3d_lf_final_3d_predictions.json.gz.part001` | [download](https://drive.google.com/file/d/1ZHD2JIQzZpFG9cwTNrm8kUPSVUBW_z_z/view?usp=drivesdk) | `0FDB1FE143B4DC054F612C038A27ACC9DB2085E6DE17D211A229966BB6F87F22` |
| `lt3d_lf_final_3d_predictions.json.gz.part002` | [download](https://drive.google.com/file/d/1_Wxu0Mq3L1Hjp3vd0_wedgJuVTlOrARa/view?usp=drivesdk) | `4662674ADC106885A66E6B8FE1841F904E1B23BE1D6206C0096DB1EF04B63D8D` |
| `lt3d_lf_final_3d_predictions.json.gz.part003` | [download](https://drive.google.com/file/d/19Cv8Bd2XLfeXlWAbx8Pu25FMk3alM9z-/view?usp=drivesdk) | `4E84F357977BCB5FC3E739BC8AD9AA1CE7DEA8AC03FB683A706D0FD319D9D937` |
| `lt3d_lf_camera_results.tar.gz` | [download](https://drive.google.com/file/d/1NGMGuB7detRJkiHy8hbXvok6rJ2PfLUU/view?usp=drivesdk) | `212EC21F77C2A55E61F372257127FCAB644467C1E2997D035E19C295099CB817` |
| `lt3d_lf_evaluation_summary.tar.gz` | [download](https://drive.google.com/file/d/1Eq27aE-nJlVLYtpO5Mo0bJUip_7O6KRW/view?usp=drivesdk) | `052C5742823C5AE13549F9577640F3E841D1E6ECD36D471F5D7F8EA1BAE43E51` |
| `lt3d_lf_qualitative_comparison_20260709.mp4` | [download](https://drive.google.com/file/d/1AMkLc06Lc34tMeUulnpMAUZGz_stlVzD/view?usp=drivesdk) | `2184296F8AF1A8612B1F95496601263D383AAA06A7ABE20815AD7D122B9236A1` |

Reconstruct the final prediction JSON with:

```bash
cat lt3d_lf_final_3d_predictions.json.gz.part* > lt3d_lf_final_3d_predictions.json.gz
sha256sum lt3d_lf_final_3d_predictions.json.gz
```

Expected SHA256:

```text
A5960E27FF29597F647C2E4468446EB6A5C9B51D872AC4284998EE1654340BE3
```

## Split Summary

| Split | Classes | Mean LCA0 AP |
|---|---|---:|
| Many | car, truck, adult, traffic_cone, barrier | 80.74 |
| Medium | trailer, bus, construction_vehicle, bicycle, motorcycle, construction_worker, pushable_pullable | 63.96 |
| Few | emergency_vehicle, child, police_officer, stroller, personal_mobility, debris | 29.21 |

## Per-Class LCA0 AP

| Class | GT count | LCA0 AP |
|---|---:|---:|
| car | 53043 | 88.51 |
| truck | 9260 | 68.13 |
| trailer | 2331 | 49.38 |
| bus | 2062 | 79.22 |
| construction_vehicle | 1513 | 40.02 |
| bicycle | 1824 | 77.26 |
| motorcycle | 1855 | 80.69 |
| emergency_vehicle | 77 | 28.00 |
| adult | 22304 | 89.22 |
| child | 159 | 32.09 |
| police_officer | 72 | 16.07 |
| construction_worker | 965 | 63.94 |
| stroller | 77 | 45.40 |
| personal_mobility | 15 | 42.29 |
| pushable_pullable | 2723 | 57.23 |
| debris | 305 | 11.39 |
| traffic_cone | 10570 | 82.25 |
| barrier | 15903 | 75.58 |

## Detailed Stratified Metrics

The full per-class distance/sparsity table is available in:

- [`per_class_stratified_lca0_current_best.csv`](per_class_stratified_lca0_current_best.csv)
- [`per_class_lca0_current_best.md`](per_class_lca0_current_best.md)
- [`many_medium_few_summary.md`](many_medium_few_summary.md)
