# Benchmark Results

This file is auto-generated from `benchmarks/metrics.json`.

## Inference Runs

| Date (UTC) | Script | Phase | Mode | TPU | Token Count | Batch Size | Runs | Avg Total s | Avg First Token s | Avg Post-First s | Avg Tokens/s | Avg Decode Tokens/s | Avg Pre-First Tokens/s | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | warmup | With Image | v6e | 128 | 1 | 1 | 67.8052 | 56.0170 | 11.7882 | 1.8878 | 10.7735 | 0.0179 | image_used=True |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | warmup | Text-Only | v6e | 128 | 1 | 1 | 33.1004 | 21.1222 | 11.9782 | 3.8670 | 10.6026 | 0.0473 | image_used=False |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | warmup | With Image | v6e | 256 | 1 | 1 | 70.3795 | 55.9641 | 14.4153 | 3.6374 | 17.6895 | 0.0179 | image_used=True |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | warmup | Text-Only | v6e | 256 | 1 | 1 | 34.5281 | 20.8636 | 13.6645 | 7.4142 | 18.6615 | 0.0479 | image_used=False |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | benchmark | With Image | v6e | 128 | 1 | 10 | 2.0259 | 0.2513 | 1.7746 | 63.2512 | 71.6600 | 3.9796 | image_used=True |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | benchmark | Text-Only | v6e | 128 | 1 | 10 | 2.0210 | 0.1927 | 1.8282 | 63.3400 | 69.4694 | 5.1913 | image_used=False |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | benchmark | With Image | v6e | 256 | 1 | 10 | 3.9644 | 0.2676 | 3.6968 | 64.6055 | 69.0163 | 3.7372 | image_used=True |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | benchmark | Text-Only | v6e | 256 | 1 | 10 | 3.9137 | 0.1979 | 3.7158 | 65.4166 | 68.6315 | 5.0548 | image_used=False |
| 2025-10-04 16:26:45 | run_gemma_inference_jax.py | final | Final Output | v6e | 128 | 1 | 1 | 71.2260 | 59.3130 | 11.9130 | 1.7971 | 10.6606 | 0.0169 | image_used=True |
| 2025-10-04 16:35:11 | run_gemma_inference_jax.py | warmup | With Image | v6e | 512 | 1 | 1 | 70.2293 | 55.5338 | 14.6955 | 4.9979 | 23.8168 | 0.0180 | image_used=True |
| 2025-10-04 16:35:11 | run_gemma_inference_jax.py | warmup | Text-Only | v6e | 512 | 1 | 1 | 37.1834 | 20.1356 | 17.0478 | 13.7696 | 29.9745 | 0.0497 | image_used=False |
| 2025-10-04 16:35:11 | run_gemma_inference_jax.py | benchmark | With Image | v6e | 512 | 1 | 10 | 5.2053 | 0.2578 | 4.9475 | 67.4416 | 70.7548 | 3.8792 | image_used=True |
| 2025-10-04 16:35:11 | run_gemma_inference_jax.py | benchmark | Text-Only | v6e | 512 | 1 | 10 | 7.2531 | 0.1849 | 7.0682 | 70.5944 | 72.2998 | 5.4109 | image_used=False |
| 2025-10-04 16:35:11 | run_gemma_inference_jax.py | final | Final Output | v6e | 128 | 1 | 1 | 76.9055 | 64.6255 | 12.2800 | 1.6644 | 10.3420 | 0.0155 | image_used=True |
| 2025-10-04 16:45:56 | run_gemma_inference_jax.py | warmup | With Image | v6e | 1024 | 1 | 1 | 70.3651 | 55.3114 | 15.0537 | 4.9883 | 23.2501 | 0.0181 | image_used=True |
| 2025-10-04 16:45:56 | run_gemma_inference_jax.py | warmup | Text-Only | v6e | 1024 | 1 | 1 | 46.9640 | 21.5344 | 25.4296 | 21.8039 | 40.2287 | 0.0464 | image_used=False |
| 2025-10-04 16:45:56 | run_gemma_inference_jax.py | benchmark | With Image | v6e | 1024 | 1 | 10 | 5.2705 | 0.2646 | 5.0059 | 66.6220 | 69.9445 | 3.7813 | image_used=True |
| 2025-10-04 16:45:56 | run_gemma_inference_jax.py | benchmark | Text-Only | v6e | 1024 | 1 | 10 | 14.7445 | 0.1948 | 14.5497 | 69.4644 | 70.3256 | 5.1431 | image_used=False |
| 2025-10-04 16:45:56 | run_gemma_inference_jax.py | final | Final Output | v6e | 128 | 1 | 1 | 77.6212 | 65.1420 | 12.4792 | 1.6490 | 10.1770 | 0.0154 | image_used=True |

