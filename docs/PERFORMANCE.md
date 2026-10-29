# Inference performance baseline

Run the full-report benchmark from the repository root:

```bash
python scripts/benchmark_inference.py --samples 101 501 2001 --repeats 2
```

The input is an analytic quaternion trajectory with timestamps and body-frame rates.
`compact` searches two hypotheses; `default` searches 48. Each timing is the median
of two calls measured with `perf_counter`. The peak is a separate call measured
with `tracemalloc`, so it covers traced Python and NumPy allocations during inference,
not process RSS or data preparation. A benchmark result is not an accuracy result.

These local measurements used Windows 11, CPython 3.14.6, and NumPy 2.5.2. They
compare commit `e802310` with the subsequent change that drops large per-candidate
diagnostic arrays after scoring. Numbers are approximate; compare revisions on the
same machine and interpreter before treating a difference as a regression.

| Samples | Search | Peak before (kB) | Peak after (kB) | Median time before | Median time after |
|--------:|:-------|-----------------:|----------------:|-------------------:|------------------:|
| 101 | compact | 64 | 75 | 0.014 s | 0.022 s |
| 101 | default | 609 | 382 | 0.311 s | 0.331 s |
| 501 | compact | 137 | 151 | 0.059 s | 0.092 s |
| 501 | default | 1,036 | 537 | 1.429 s | 1.469 s |
| 2,001 | compact | 438 | 457 | 0.234 s | 0.351 s |
| 2,001 | default | 2,731 | 1,133 | 5.497 s | 5.707 s |

The 48-hypothesis, 2,001-sample case used about 58% less traced peak memory, with
about 4% more elapsed time. A two-hypothesis search is slower because overall
diagnostics now recomputes details for its selected candidate; the absolute cost at
2,001 samples is about 0.12 seconds. This is a measured tradeoff, not a long-log
performance guarantee. Larger logs, Euler searches, other NumPy/SciPy versions,
process RSS, and cold-start behavior still need separate measurements before a
supported scale claim.
