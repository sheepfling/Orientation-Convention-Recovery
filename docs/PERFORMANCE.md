# Inference performance baseline

Run the full-report benchmark from the repository root:

```bash
python scripts/benchmark_inference.py --samples 101 501 2001 --repeats 2
```

By default, the input is an analytic quaternion trajectory with timestamps and
body-frame rates. `--motion-profile sampled-rate-nonanalytic --motion-seed 730201`
selects a reproducible body-rate signal whose attitudes are independently integrated
numerically. `compact` searches two hypotheses; `default` searches 48. Each timing
is the median of the requested calls measured with `perf_counter`. The peak is a
separate call measured with `tracemalloc`, so it covers traced Python and NumPy
allocations during inference, not process RSS or data preparation. A benchmark
result is not an accuracy result. JSON results record the input profile/seed and
the timing sample count, minimum, median, and maximum to make run-to-run spread
visible.

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
performance guarantee. The current post-cache 10,001-sample Euler process baseline
is recorded below; broader motion distributions, other runtimes, and independent
host measurements remain open before a supported scale claim.

## Euler search and isolated process measurements

The benchmark also accepts `euler-compact` (two hypotheses) and `euler-default`
(576 hypotheses). The default Euler search has 48 distinct attitude decodings;
the inference loop now retains only the current decoding while it evaluates its
rate hypotheses. A regression test checks that earlier decoded rotations are no
longer held live. This does not change candidate ranking. A later change also
replaced retained three-component residual vectors with per-interval squared
errors for evidence comparisons; selected residual vectors are recomputed as needed.

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

## Shared prefix composition

Profiling the 101-sample, 576-hypothesis Euler search showed that computing each
window size rebuilt the same increment-prefix stream. The inference path now
builds one prefix stream per candidate and reuses it for all window sizes. The
prefix accumulator uses normalized scalar-last quaternion products instead of
constructing a SciPy `Rotation` object for every interval. Direct-product tests
cover body/world composition, several window sizes, and a 257-interval stream
with large rotations.

One-call timing comparisons on the same Windows/Python/NumPy/SciPy environment
are below. The baseline range is from two single runs at commit `654e5a6`; the
shared-prefix and scalar-prefix stages each had one run. They are diagnostic,
not stable throughput promises.

| Samples | Hypotheses | Prior timing | Shared prefix | Scalar prefix |
|--------:|-----------:|-------------:|--------------:|--------------:|
| 101 | 576 | 4.14–4.56 s | 2.03 s | 1.28 s |
| 501 | 576 | 19.60–21.31 s | 8.71 s | 4.67 s |

The final 2,001-sample, 576-hypothesis Euler run took 16.72 s for inference and
18.26 s end to end in a fresh process, with a 122,363,904 B peak Windows working
set. Traced peak allocation was 4,559,734 B. These are single-run figures; a
10,001-sample broad Euler search and repeated cross-version measurements remain
open before claiming a supported scale.

The 10,001-sample, 48-hypothesis quaternion case took 3.38 s after this change,
versus 28.03 s in the preceding baseline run. Peak working set was approximately
unchanged (106,242,048 B versus 105,869,312 B). This speed comparison also comes
from one run per version on one host, so it is a lead for further measurement,
not a portable speedup guarantee.

## Compact residual evidence

Ordinary scoring previously retained a three-component rotation vector for every
interval of every hypothesis, even though the cross-candidate fixed-noise evidence
calculation uses only its squared magnitude. It now retains one squared-error
value per interval and recomputes full vectors for selected noise and interval
diagnostics. The vector-based evidence function remains available and delegates
to the same scalar-energy calculation. Tests compare the two formulations and
check that ordinary candidates do not retain vectors.

Fresh-process measurements for the 576-hypothesis Euler search on the same Windows
host are below. The before run used commit `5f5b772`; the after run used the compact
working tree. Each is one run, so timings are not a robust speed comparison.

| Samples | Peak working set before | Peak working set after | Inference before | Inference after |
|--------:|------------------------:|-----------------------:|-----------------:|----------------:|
| 2,001 | 122,363,904 B | 101,896,192 B | 16.72 s | 15.83 s |
| 10,001 | 244,006,912 B | 147,255,296 B | 82.00 s | 79.45 s |

The 10,001-sample case used about 40% less peak working-set memory in this
measurement. The result is still an analytic synthetic workload on one interpreter;
cross-version and independently sourced workload measurements remain open.

A current post-observed-cache baseline was measured on the same local Windows
11 / CPython 3.14.6 / NumPy 2.5.2 / SciPy 1.18.0 host using the seeded
sampled-rate trajectory and 576-hypothesis Euler search. Three fresh cold-worker
runs at 10,001 samples measured 33.53–33.64 s inference time (33.57 s median) and
158,244,864–158,908,416 B peak working set (158,453,760 B median). To repeat with
the public process benchmark (which also runs an in-process timing and traced
allocation pass):

```bash
python scripts/benchmark_inference.py \
  --samples 10001 --searches euler-default --repeats 1 --process \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201
```

This is a current one-host baseline, not a cache-versus-uncached RSS comparison
or a supported maximum-size claim. Long-run memory on the other supported
interpreters and on an independent host remains unmeasured.

## Euler cross-version check

On 2026-10-30, the 576-hypothesis Euler search was measured on the same local
Windows host with the three available interpreters. The 3.11 runtime reports
Windows 10 while 3.12 and 3.14 report Windows 11; all reported build 26300. Run
the following under each interpreter, changing the output path so reports are
preserved separately:

```bash
python scripts/benchmark_inference.py \
  --samples 101 501 2001 --searches euler-default --repeats 2 --process \
  --output euler-cross-version.json
```

Each cell below is `median inference seconds / tracemalloc peak MB / fresh-process
peak working-set MB`; MB uses 1,000,000 bytes. The first timing is the median of
two in-process calls, traced memory is a separate inference call, and the process
peak comes from a fresh cold-inference subprocess.

| Python | NumPy | SciPy | 101 samples | 501 samples | 2,001 samples |
|:-------|:------|:------|:-----------|:-----------|:--------------|
| 3.11.15 | 2.4.6 | 1.17.1 | 1.20 / 3.93 / 90.58 | 4.66 / 5.99 / 92.61 | 16.97 / 13.29 / 99.30 |
| 3.12.13 | 2.5.4 | 1.18.1 | 1.21 / 3.85 / 91.85 | 4.63 / 5.91 / 93.75 | 16.84 / 13.21 / 101.48 |
| 3.14.6 | 2.5.2 | 1.18.0 | 1.22 / 3.92 / 94.56 | 4.61 / 5.98 / 96.40 | 17.33 / 13.29 / 104.42 |

The in-process medians differed by less than 3% at each size across these runtime
stacks. This is a small, same-host diagnostic, not a portable performance claim;
Python, NumPy, and SciPy versions all vary together, and the trajectory is still
the analytic generator. Both cross-version tables in this section are pre-cache
baselines; the cache comparison follows below. The sampled-rate synthetic profile
described in [`VALIDATION.md`](VALIDATION.md) was then measured with the same
Euler search:

```bash
python scripts/benchmark_inference.py \
  --samples 101 501 2001 --searches euler-default --repeats 1 --process \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --output sampled-rate-euler-cross-version.json
```

Each cell below is `single in-process timing / traced peak MB / fresh-process
peak working-set MB`. These are single timing samples (not repeated medians), on
the same local Windows host. Python, NumPy, and SciPy versions vary together.

| Python | NumPy | SciPy | 101 samples | 501 samples | 2,001 samples |
|:-------|:------|:------|:-----------|:-----------|:--------------|
| 3.11.15 | 2.4.6 | 1.17.1 | 1.19 / 3.99 / 90.73 | 4.67 / 5.99 / 91.55 | 16.89 / 13.17 / 99.02 |
| 3.12.13 | 2.5.4 | 1.18.1 | 1.22 / 3.91 / 92.25 | 4.45 / 5.90 / 93.32 | 18.18 / 13.09 / 101.12 |
| 3.14.6 | 2.5.2 | 1.18.0 | 1.20 / 3.99 / 94.63 | 5.65 / 6.00 / 96.69 | 16.57 / 13.17 / 103.98 |

For the 2,001-sample cases, the separate cold inference measurements were 19.82,
16.24, and 16.74 seconds, respectively. The timing values are close to the
analytic workload at this size; this only demonstrates similar cost for this
seeded synthetic rate family, not runtime on independent systems or real logs.
One timing per version is insufficient to claim a cross-version performance
trend. A separate five-repeat, 501-sample probe showed why: on Python 3.11 one
seed's median shifted from 9.08 seconds to 4.69 on an immediate rerun, while a
profile comparison showed nearly identical call counts for that and a faster
seed. Treat these local results as host-sensitive diagnostics, not evidence of
seed-dependent cost. External logs remain an open validation and performance
gate.

## Reusing Euler heuristic diagnostics

The Euler range/branch diagnostic depends on the input stream, Euler order, and
angle unit—not the rate convention or Euler mode. The inference loop now caches
that result per order/unit pair. For the default Euler search, a cProfile run on
Python 3.11 reduced calls to `euler_heuristic_diagnostics` from 578 to 14 (12
unique order/unit pairs plus detailed-report rescoring); its cumulative time in
that profile fell from 2.938 s to 0.069 s. Tests check that mode/rate variants
share the cached result and that inference makes one calculation per key.

The benchmark can compare cached and uncached paths on the same trajectory,
alternating their order and verifying that the resulting reports are identical:

```bash
python scripts/benchmark_inference.py \
  --samples 101 501 --searches euler-default --repeats 3 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --compare-euler-cache --output euler-cache-paired.json

python scripts/benchmark_inference.py \
  --samples 2001 --searches euler-default --repeats 1 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --compare-euler-cache --output euler-cache-paired-2001.json
```

Paired results on the same Windows/Python 3.11.15, NumPy 2.4.6, and SciPy 1.17.1
host are below. The 101- and 501-sample cells summarize three alternating runs
per mode; the 2,001-sample cell is one run per mode. Peak memory is from a
separate traced run.

| Samples | Uncached median (range) | Cached median (range) | Speedup | Traced peak MB uncached → cached |
|--------:|:------------------------|:---------------------|--------:|:---------------------------------|
| 101 | 1.324 s (1.207–1.343) | 0.879 s (0.804–0.881) | 1.51× | 3.99 → 3.95 |
| 501 | 4.658 s (4.609–4.683) | 2.653 s (2.613–2.716) | 1.76× | 5.99 → 5.95 |
| 2,001 | 19.435 s (n=1) | 11.495 s (n=1) | 1.69× | 13.17 → 13.13 |

These are local synthetic-workload measurements, not general speed or scale
claims. Interleaving reduces timing drift between modes, but this host has shown
substantial load-sensitive variation; repeat on a controlled host before using
the ratios as release guidance.

## Reusing observed interval and window rotations

Observed rotations depend on the decoded attitude series and rate frame, not on
the candidate rate order, units, or sign. During inference, interval and
window-composed observed rotations are reused for each attitude/frame/window
key. This covers both ordinary scoring and the repeated period-grid/optimizer
evaluations used when timestamps are missing; final detailed scoring gets its
own scoped cache. Direct scoring calls without a supplied cache retain the
uncached path. Tests compare both rate frames against direct rotations and count
one observed window calculation per key during the missing-time search, apart
from the separate detailed rescore.

The benchmark provides an interleaved comparison that verifies reports remain
identical and measures traced peak allocation:

```bash
python scripts/benchmark_inference.py \
  --samples 101 501 --searches euler-default --repeats 2 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --compare-observed-cache --output observed-increment-cache-paired.json

python scripts/benchmark_inference.py \
  --samples 31 --searches euler-compact --repeats 3 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 913 \
  --compare-observed-cache --missing-timestamps \
  --output observed-increment-cache-missing-time.json
```

On the local Windows 11 / Python 3.14.6 / NumPy 2.5.2 / SciPy 1.18.0 host, both
paired reports were identical. Each mode ran twice per size; traced peak memory
was measured in a separate inference call.

| Samples | Uncached median (range) | Cached median (range) | Speedup | Traced peak MB uncached → cached |
|--------:|:------------------------|:---------------------|--------:|:---------------------------------|
| 101 | 0.813 s (0.777–0.848) | 0.654 s (0.650–0.658) | 1.24× | 3.94 → 3.96 |
| 501 | 2.523 s (2.503–2.543) | 2.126 s (2.114–2.138) | 1.19× | 5.96 → 6.00 |

This shows a modest benefit on this seeded, timestamped synthetic Euler workload,
not independent accuracy or a general throughput guarantee. The missing-time
31-sample, two-hypothesis Euler-compact run (three repeats per mode) measured
0.0462 s uncached and 0.0356 s cached, a 1.30× local speedup with identical
reports; traced peak was 0.061 MB and 0.073 MB, respectively. This tiny case is
useful as a regression probe, not a scaling result. `tracemalloc` does not capture
every native allocation, so a longer-trajectory process comparison is also
available below.

The paired fresh-process comparison toggles only the observed-increment cache,
interleaves the modes, and verifies each complete report by fingerprint. To
repeat the 10,001-sample case:

```bash
python scripts/benchmark_inference.py \
  --samples 10001 --searches euler-default --repeats 2 --process \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --compare-observed-cache --output observed-cache-process.json
```

Matched runs on the same local Windows host covered the three supported
interpreters. The cache mode order was uncached/cached, then cached/uncached;
all four report fingerprints matched within each interpreter. Each cell gives
median inference seconds (range), the ratio of uncached to cached medians, and
median peak working set. MB uses 1,000,000 bytes.

| Python | NumPy / SciPy | Uncached inference | Cached inference | Ratio | Peak working set uncached → cached |
|:-------|:-------------|:------------------|:-----------------|------:|:----------------------------------|
| 3.11.15 | 2.4.6 / 1.17.1 | 44.22 s (42.90–45.55) | 47.63 s (35.75–59.52) | 0.93× | 146.16 → 150.06 MB (+2.7%) |
| 3.12.13 | 2.5.4 / 1.18.1 | 40.52 s (39.76–41.28) | 32.50 s (32.49–32.52) | 1.25× | 151.37 → 153.77 MB (+1.6%) |
| 3.14.6 | 2.5.2 / 1.18.0 | 41.42 s (41.32–41.51) | 33.22 s (32.68–33.75) | 1.25× | 155.04 → 158.44 MB (+2.2%) |

The 3.11 timing is inconclusive: its cached runs varied widely and reversed
relative to the uncached mode between pairs. The 3.12 and 3.14 measurements
showed similar timing ratios, but two repeats on one host are not enough for a
portable performance claim. Across these runs, median peak working-set memory
rose by about 1.6–2.7% with the cache enabled. The RSS is peak Windows working
set sampled after inference and before report serialization for fingerprinting;
process wall time additionally includes startup, imports, and synthetic input
creation. This is a local time/memory tradeoff measurement only, not an accuracy
result or supported maximum-size guarantee. Other cadences and independent hosts
still need matched measurements.

### Three-repeat 25 ms long-recording follow-up

A current-worktree rerun used three interleaved fresh-process pairs at 10,001
samples, 25 ms cadence, the 576-hypothesis Euler-default search, and the seeded
`sampled-rate-nonanalytic` motion (seed 730201). All six report fingerprints
matched exactly. The cache was faster in each paired repeat, but timing spread
remains substantial:

| Cache mode | Median inference | Range across three runs | Median peak working set |
|:-----------|----------------:|:-----------------------|------------------------:|
| Uncached | 44.98 s | 36.46–46.86 s | 168,452,096 B |
| Cached | 32.23 s | 27.20–35.20 s | 171,098,112 B |

The median speedup was 1.40× and median peak working set increased by 1.6% with
caching. This corroborates the local time/memory tradeoff for one synthetic
trajectory on one Python 3.14.6 host; it does not establish a scale limit or a
portable speedup. The ignored JSON result is
`.validation-corpus-performance-python-matrix-20261030/observed-cache-process-314-25ms-rep3-current.json`
(SHA-256 `7cf5fb413de4d2b901a0e88dfe90b332d7477674eaae159ba79b8c8d59d9ace3`).
Reproduce with:

```bash
mkdir -p .validation-corpus-performance-python-matrix-20261030
python scripts/benchmark_inference.py \
  --samples 10001 --searches euler-default --repeats 3 --process \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --compare-observed-cache \
  --output .validation-corpus-performance-python-matrix-20261030/observed-cache-process-314-25ms-rep3-current.json
```

### Additional cadence follow-up

The benchmark accepts `--sample-period-s` (default `0.025`) and records the value
in both in-process JSON and fresh-process worker results. Paired 50 ms and 100 ms
runs used the same 10,001-sample, 576-hypothesis workload shape, motion profile,
and seed. The command below reproduces the 50 ms case; use `0.1` for the 100 ms
case:

```bash
python scripts/benchmark_inference.py \
  --samples 10001 --searches euler-default --repeats 2 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --sample-period-s 0.05 --compare-observed-cache --process
```

On Windows 11 / CPython 3.14.6 / NumPy 2.5.2 / SciPy 1.18.0, all four fresh
process reports had identical fingerprints within each cadence. Results are
medians of two runs per cache mode:

| Period | Uncached inference | Cached inference | Ratio | Peak working set uncached → cached |
|-------:|:-------------------|:-----------------|------:|:----------------------------------|
| 50 ms | 43.45 s (43.15–43.76) | 35.55 s (35.20–35.90) | 1.22× | 153.83 → 156.63 MB (+1.8%) |
| 100 ms | 44.46 s (44.09–44.84) | 35.26 s (34.99–35.52) | 1.26× | 152.86 → 156.92 MB (+2.7%) |

These are two-repeat, single-host diagnostics, not portable performance claims.
Holding sample count fixed means the 50 ms and 100 ms trajectories span 500 and
1,000 seconds, respectively, rather than 250 seconds at 25 ms; the measurements
do not isolate a causal cadence effect.

Machine-readable outputs are retained locally under the Git-ignored
`.validation-corpus-performance-*` directories and are not part of the source
distribution.

## Reusing predicted rate integrations

Euler hypotheses vary attitude decoding much more often than they vary rate
decoding. The inference loop now caches each predicted interval integration by
rate order, unit, sign, frame, and integration substep count. It reuses the
result only when decoded rate values and the resolved interval durations compare
exactly; a mismatch recomputes and replaces the entry. This keeps timestamped and
configured-period paths reusable while remaining safe when the sample period is
inferred. It stores snapshots of those inputs, so in-place changes cannot return
stale predictions; retained input memory scales with the active rate/frame keys.
The cache is scoped to one
trajectory and one inference call. A regression test compares complete reports
with the cache bypassed and confirms that a 576-hypothesis Euler search performs
far fewer integrations.

A current one-repeat run on Windows / CPython 3.11.15 / NumPy 2.4.6 / SciPy
1.17.1 used the sampled-rate synthetic profile, seed 730201, 25 ms cadence, and
the 576-hypothesis Euler-default search. The standard benchmark reports timed
inference plus a separate `tracemalloc` pass:

| Samples | Inference time | Traced peak |
|--------:|---------------:|------------:|
| 501 | 1.636 s | 6.32 MB |
| 2,001 | 6.172 s | 14.45 MB |

These are single-run, single-host figures and should not be compared as a
controlled speedup against older rows above. They confirm the cache's current
cost profile, not a supported maximum-size or external-data performance claim.

A separate cold-process probe of the same current implementation at 10,001
samples completed inference in 29.78 s, with 31.02 s end-to-end process time and
160,129,024 B peak Windows working set. It used the same 576-hypothesis search,
seed, profile, cadence, and Python 3.11 dependency stack. This is one run without
a `tracemalloc` pass; the 10,001-sample working set is a useful current baseline,
not a repeatable SLA or a supported maximum-size claim.
