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
performance guarantee. Larger logs, other NumPy/SciPy versions, and representative
process-memory measurements still need more work before a supported scale claim.

## Euler search and isolated process measurements

The benchmark also accepts `euler-compact` (two hypotheses) and `euler-default`
(576 hypotheses). The default Euler search has 48 distinct attitude decodings;
the inference loop now retains only the current decoding while it evaluates its
rate hypotheses. A regression test checks that earlier decoded rotations are no
longer held live. This does not change candidate ranking or eliminate the residual
vectors retained for evidence comparisons.

The following before/after measurements used the same Windows 11, CPython 3.14.6,
NumPy 2.5.2, and SciPy 1.18.0 environment, with one timing call and a separate `tracemalloc`
call per case. The before run was at commit `4d48b56`; the after run used the
working-tree cache change. Single timings are noisy and are not a speed claim.

| Samples | Hypotheses | Traced peak before | Traced peak after | Time before | Time after |
|--------:|-----------:|-------------------:|------------------:|------------:|-----------:|
| 101 | 576 | 4,202,856 B | 4,092,887 B | 4.28 s | 4.56 s |
| 501 | 576 | 4,712,841 B | 4,309,386 B | 20.09 s | 21.31 s |

For a fresh-process measurement, run:

```bash
python scripts/benchmark_inference.py --samples 101 501 --searches euler-default --repeats 1 --process
```

`--process` adds the first inference call's time inside a new process, the
end-to-end subprocess wall time (including interpreter startup, imports, data
creation, and that inference), and lifetime peak resident memory. On Windows it
requires the development extra (`pip install -e ".[dev]"`) for `psutil`, and the
memory number is [peak working set](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters);
on Linux it is [`ru_maxrss`](https://www.man7.org/linux/man-pages/man2/getrusage.2.html).
The benchmark reports `null` for peak process memory on other platforms. Peak
process memory includes interpreter and benchmark-harness overhead, whereas the
`tracemalloc` figure covers traced allocations during one inference call. Neither
measurement is an accuracy result.

The isolated Windows run after the cache change measured 93,872,128 B peak
working set at 101 samples and 99,418,112 B at 501 samples for the 576-hypothesis
Euler search. Cold inference inside those processes took 4.20 s and 19.02 s;
end-to-end process times were 5.21 s and 20.01 s. There is no pre-change process
memory measurement, so these numbers are a current baseline, not evidence of an
RSS reduction.

At 10,001 samples on the same host, one-call measurements show how search breadth
changes cost. These are baselines, not performance guarantees or release limits.

| Search | Representation | Hypotheses | Inference time | Peak working set |
|:-------|:---------------|-----------:|---------------:|-----------------:|
| compact | quaternion | 2 | 1.74 s | 93,429,760 B |
| euler-compact | Euler | 2 | 1.92 s | 93,646,848 B |
| default | quaternion | 48 | 28.03 s | 105,869,312 B |

The 576-hypothesis Euler search has only been measured here up to 501 samples;
larger broad-search logs still need profiling before making a scale promise.
