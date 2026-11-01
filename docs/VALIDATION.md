# Independent validation corpus

Synthetic trajectories test implementation invariants, but they cannot establish
recovery accuracy on logs produced by other systems. Keep a separate corpus of
orientation and angular-rate CSV files whose conventions were recorded independently
of this package's predictions. A source specification, reference implementation, or
original logger configuration should establish each label.

## Optional TUM-VI real-data preparation

The optional `scripts.prepare_tumvi_corpus` adapter prepares a bounded segment from
an extracted TUM-VI room sequence. The [official dataset page](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
documents motion-capture ground truth, calibrated IMU data, and full-trajectory
ground truth for the room sequences; it licenses the data under CC BY 4.0. Follow
the dataset's attribution terms. The adapter does not download or redistribute
the source files, and no TUM-VI data is checked into this repository.

On the official dataset page, use the **Euroc/DSO 512x512 export** (not the ROS bag
download), then extract a room sequence such as `dataset-room1_512_16` and run:

```bash
python -m scripts.prepare_tumvi_corpus \
  /path/to/dataset-room1_512_16 \
  --outdir .validation-corpus-tumvi-room1 \
  --start-s 10 --duration-s 10
python -m scripts.validate_corpus \
  .validation-corpus-tumvi-room1/manifest.json \
  --output .validation-corpus-tumvi-room1/assessment.json
```

The input directory must contain the CSV export files `mav0/imu0/data.csv` and
`mav0/mocap0/data.csv`. The adapter selects gyro timestamps in the common
coverage and SLERPs motion-capture orientations onto those times, rejecting any
interpolation bracket wider than 25 ms by default. Choose a segment with useful
three-axis rotation; a short or weakly excited excerpt can be ambiguous even
when its convention labels are known. Output directories are never overwritten.
The parser requires the documented EuRoC-style `timestamp [ns]`, `w_RS_S_*`
gyro, and `q_RS_*` quaternion columns with explicit nanosecond, radians-per-second,
and unitless declarations; it rejects generic column aliases and incompatible or
missing units rather than inferring physical conventions from values.
The manifest records the source-file SHA-256 digests, the dataset citation and
license, the labeled convention basis, the requested and selected timestamp ranges,
the timing caveat, and each transformation. Controlled noise, latency, jitter, and
dropout and timestamp-clock-rate-error fields that are not measured are recorded as
`null`, not as zero; the
validator's residual diagnostics are estimates from the observed trajectory and
remain separate from those source-condition fields. These provenance-rich imports
use manifest schema version 3; the resulting assessment uses schema version 7.
The assessment carries that provenance forward, and reports unknown dropped-row
counts as `null` rather than zero.

The expected labels for this calibrated export are body-to-world quaternion
orientation in `qx,qy,qz,qw` order and `x,y,z` angular velocity in the IMU/body
frame in radians per second. Here, body-to-world means the calibrated `T_WI`
transform maps IMU/body coordinates into the motion-capture world frame. This is
real measured motion with an independent
motion-capture pose source, but it is not a blind test of raw sensor-axis or
time-offset recovery: TUM-VI's upstream alignment estimates time offsets by
matching gyro measurements with rotational velocity differentiated from the
ground-truth orientations. It also evaluates one external acquisition system,
not the roadmap's multi-system evidence gate. The official EuRoC/DSO export has
not been run locally.

A one-off local ingestion smoke test instead used the room1 CSVs from a pinned
[third-party mirror commit](https://github.com/BahMoh/TUMVI-IMU-only-Dataset-room1-to-room6/commit/f5538aff424124096727abfa8fa9cb9f0ca034cf).
Its README says the files were extracted from an original TUM-VI bag and converted
with an external implementation; that conversion was not independently audited.
The flat CSVs were normalized to the adapter's explicit EuRoC-style input with
decimal epoch-second-to-nanosecond conversion, `angular_velocity.x/y/z` mapped to
body `p/q/r` in rad/s, and `transform.rotation.x/y/z/w` mapped to the labeled
body-to-world quaternion. These labels follow the documented calibrated TUM-VI
convention, but the re-export itself is not independent confirmation of that
mapping. The pinned mirror files had SHA-256 values
`6a5b3f5e526677a419d5f9d066c7f52ec1edd6efab0d69631e532bcb0b829f88` (IMU) and
`959bd51462d7f3ef6b1bac712ece7e6d6ab46f73679b22d8801aa38e1ed51a36` (poses).
The mirror provides no separate license statement; its data are not committed or
redistributed here.

A scan found 26 motion-capture gaps wider than the adapter's 25 ms limit, so only
gap-free windows were prepared. The 0–20 s case is nested within the 0–40 s case;
all excerpts come from this one sequence and are not independent trials.

| Window from common coverage start | Samples | Outcome |
| --- | ---: | --- |
| 0–20 s | 3,987 | Abstained with two evidence contenders; data winner and operational class matched the label. |
| 0–40 s | 7,975 | Correct recommendation with a unique evidence contender. |
| 82–98 s | 3,190 | Correct recommendation with a unique evidence contender. |
| 122–140.5 s | 3,688 | Correct recommendation with a unique evidence contender; weak three-axis excitation warning. |

All four excerpts had the labeled class as data winner and operational class; the
three recommendations matched the candidate label, and the remaining case
abstained. This is useful only as a parser, resampling, provenance, and decision
pipeline smoke test on one calibrated, third-party re-export. It is not an accuracy
estimate, a blind test of raw-axis or timing recovery, or evidence for the
multi-system gate. The official TUM-VI dataset is licensed CC BY 4.0; follow its
attribution terms when obtaining or using the upstream data.

The repeatable `scripts.validate_sparse_tumvi` runner exercises the sparse API on
separately timestamped streams. It selects the native mocap quaternion nearest
each target pose time without pose interpolation, and preserves the native gyro
timestamps, adding only samples that bracket the selected anchor range. Its JSON
report records input hashes, unknown measurement conditions, and each cadence's
recovery outcome. If inputs were converted from another export, pass a matching
`--lineage-manifest`; the runner verifies that it fingerprints those exact inputs
and carries the upstream provenance separately from transformations performed by
the sparse run.

For the local normalized room1 mirror export, the command was:

```bash
python -m scripts.validate_sparse_tumvi \
  /path/to/dataset-room1_512_16 \
  --duration-s 40 \
  --pose-cadences-s 0.2 0.4 0.8 1.6 3.2 \
  --lineage-manifest /path/to/prepared-40s/manifest.json \
  --output .validation-corpus/sparse-tumvi-room1-40s.json
```

The output path must be new; the runner refuses to overwrite a report. Omit
`--lineage-manifest` for an unmodified official export, or supply a manifest that
records the exact direct-input hashes when using converted files.

| Nominal pose cadence | Orientation anchors | Dense rate samples used | Actual anchor span | Labeled data rank | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.2 s | 201 | 7,977 | 40.0 s | 1 | Unique operational recommendation matched the mapped label. |
| 0.4 s | 101 | 7,977 | 40.0 s | 1 | Unique operational recommendation matched the mapped label. |
| 0.8 s | 51 | 7,977 | 40.0 s | 1 | Unique operational recommendation matched the mapped label. |
| 1.6 s | 26 | 7,977 | 40.0 s | 1 | Unique operational recommendation matched the mapped label. |
| 3.2 s | 13 | 7,658 | 38.4 s | 1 | Unique operational recommendation matched the mapped label. |

This confirms that the sparse-stream API and repeatable runner accept distinct pose
and rate timestamps and recover the documented mapping on this excerpt, down to 13
poses over 38.4 s. It remains a compatibility smoke test on one sequence, not
independent accuracy evidence: the mirror conversion was not audited, all windows
come from one acquisition, and TUM-VI's upstream gyro/pose alignment precludes a
blind timing or raw-axis test. No field-noise or timing conditions are inferred from
this result.

By default, the independent generator writes four cases from closed-form ZYX
rotations and analytic body/world angular velocities. It does not import the
inference package or its synthetic helper. These cases exercise quaternion storage
variants, direction and rate-frame changes, rate-channel reordering, quaternion sign
flips, irregular timestamps with dropped samples, Euler/degrees inputs, and seeded
orientation/rate measurement noise. The noise case records its seed and standard
deviations in the manifest and assessment. Tests cross-check analytic angular rates
against symmetric finite differences evaluated with SciPy's rotation implementation,
without using the package's quaternion-conversion helpers as the oracle. They remain
**synthetic**, so passing them is not real-world validation.

The regular CI test suite also runs a noiseless end-to-end recovery gate over the
full 192-value quaternion/rate convention matrix, plus the generator's three fixed
controls (195 cases total). The independent generator writes each case at 41 samples;
the corpus validator must recover every labeled class with no evidence, data-winner,
or operational misses. Run just this gate with
`python -m pytest -q tests/test_independent_recovery.py`, or include it in the full
`python scripts/ci.py` pipeline. This deterministic regression check is not a noise
robustness rate, confidence calibration, or evidence of accuracy on real systems.

A companion clean Euler recovery gate runs the entire 192-value matrix (12 axis
orders, intrinsic/extrinsic modes, four angle units, and both attitude directions)
on one independently generated 101-sample motion. It requires all 192 labeled classes
to be correctly recommended with no evidence, winner, or operational misses. The
Euler gate is also noiseless and single-motion; neither gate measures robustness or
calibration.

With the checked-in generator and current inference settings, the noisy case is
ambiguous but retains the known class; the other three are uniquely resolved.
This is a deterministic regression fixture, not a Monte Carlo robustness rate or
confidence calibration.

For a paired noise sensitivity study, the generator offers low, moderate, and high
per-axis standard deviations: orientation/rate `(0.0015 rad, 0.012 rad/s)`,
`(0.003 rad, 0.025 rad/s)`, and `(0.006 rad, 0.05 rad/s)`. It uses the same seed
set across all levels and motion profiles; increase `--noise-replicates` for a
larger study. `--noise-sample-counts 161 401` compares 4-second and 10-second
recordings at 25 ms spacing. Separate seeded streams for orientation/rate noise,
timestamp jitter, and dropout are reused across acquisition-factor levels. Each
shorter paired trial is an exact prefix of the longer one, with configured duration
endpoints protected from dropout; the assessment groups results by sample count and
period.

Acquisition stress factors can be varied independently with
`--timestamp-jitter-amplitudes-s`, `--dropout-probabilities`,
`--orientation-latencies-s`, and `--rate-latencies-s`. Supplying several values to a
flag creates a factorial sweep, so case counts multiply across the selected motion,
noise, duration, and acquisition conditions. Timestamp jitter is uniform in the
requested symmetric amplitude and must be less than half the 25 ms sample period.
Dropout removes a whole row (orientation and rate together), is seeded per sample,
and the manifest records both its requested probability and realized count; scheduled
duration endpoints are retained to preserve paired duration comparisons. A positive
channel latency means that channel's value at row
time `t` was generated from the analytic signal at `t - latency`. Thus a mismatch
between orientation and rate latency is an intentional synchronization error, not
an inferred convention. The assessment groups by requested acquisition condition,
not each seed's realized dropout count, and reports observed/dropped sample-count
ranges and means for those replicates.

Timestamp-clock scale error can be varied with
`--timestamp-clock-rate-errors-ppm`. Positive ppm means the logged clock reports
elapsed time faster than physical time; negative ppm means it runs slower. The
generator keeps the physical orientations and SI angular rates fixed and scales the
recorded timestamps by `1 + ppm / 1,000,000`, so this isolates a shared timebase
calibration error from channel delay and timestamp jitter. Its known value is recorded
in validation manifest schema version 4 and assessment condition summaries. For
imported data without an independently measured clock calibration, record `null`;
do not assume a zero error. These generated cases are synthetic sensitivity tests,
not empirical evidence about a logger's clock accuracy.

### Shared timestamp-clock scale pilot

A clean pilot crossed one analytic three-axis motion with `-50,000`, `-10,000`,
`-1,000`, `0`, `1,000`, `10,000`, and `50,000` ppm at both 4 seconds (161 samples)
and 10 seconds (401 samples). All 14 cases retained the true convention as the sole
data/evidence winner, with no operational or evidence misses. The package recommended
at 0 and ±1,000 ppm (0.1%); it abstained at ±10,000 ppm (1%) and ±50,000 ppm (5%)
at both durations. Those abstentions were driven by measurable, strongly correlated
residuals and the existing evidence-sufficiency guard. No false recommendations
occurred.
The same 14 cases are now asserted end-to-end in
`tests/test_analytic_corpus.py::test_timestamp_clock_scale_errors_are_applied_and_grouped`,
including the sole-winner and recommend/abstain outcomes.

This is a narrow, noiseless synthetic sensitivity result for one motion and one truth
convention. It shows that this guard can withhold evidence when a shared timebase scale
is wrong; it does not identify clock error as the cause, establish a general ppm
threshold, or calibrate confidence. Reproduce the pilot with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-clock-scale-duration-pilot \
  --noise-profile none --noise-sample-counts 161 401 \
  --timestamp-clock-rate-errors-ppm -50000 -10000 -1000 0 1000 10000 50000
python -m scripts.validate_corpus \
  .validation-corpus-clock-scale-duration-pilot/manifest.json \
  --workers 4 \
  --output .validation-corpus-clock-scale-duration-pilot/assessment.json
```

### Paired noisy timestamp-clock-scale screen across motion profiles

A broader paired synthetic cohort crossed all seven generator motion profiles with
three noise tiers, three noise replicates (seeds `20261040`--`20261042`), 4- and
10-second durations (161 and 401 samples at 25 ms), and the same seven clock-scale
values as the pilot. The three sampled-rate profiles each used motion seeds
`730301`--`730303`; the analytic profiles used their deterministic default paths.
Noise was independent Gaussian orientation/rate noise at the configured low, moderate,
and high levels. There was no timestamp jitter, row dropout, or channel latency. All
noisy cases used one generated truth convention: body-to-world scalar-last quaternions
and positive body-frame `pqr` rates in radians per second. The generator scaled
recorded timestamps while holding the physical trajectory and SI rates fixed.

The cohort contained 1,638 noisy cases plus three fixed clean controls. Across the
noisy cases there were 522 correct recommendations and 1,116 abstentions; every
recommendation was correct and truth remained in the evidence set in every case.
There were no evidence misses or false recommendations. In 245 cases the true class
was not the raw data winner (and therefore not the operational class), so the policy
abstained rather than recommend it. The recommendation counts below exclude the three
controls and are paired over the same 234 trajectory/noise/duration combinations at
each clock scale.

| Timestamp clock-scale error | Correct recommendations / 234 | Evidence contender sets changed vs. 0 ppm | Decision statuses changed vs. 0 ppm |
|---:|---:|---:|---:|
| −50,000 ppm (−5%) | 68 | 55 | 9 (recommendation → abstention) |
| −10,000 ppm (−1%) | 75 | 15 | 2 (recommendation → abstention) |
| −1,000 ppm (−0.1%) | 77 | 0 | 0 |
| 0 ppm | 77 | — | — |
| +1,000 ppm (+0.1%) | 77 | 0 | 0 |
| +10,000 ppm (+1%) | 77 | 14 | 0 |
| +50,000 ppm (+5%) | 71 | 43 | 6 (recommendation → abstention) |

Across the paired comparisons, neither the raw data-winner set nor operational class
changed with clock scale. Larger scale errors did change some evidence contender sets,
and 17 baseline recommendations became abstentions; no abstention became a
recommendation. This is useful evidence that the current decision policy remains
conservative on this generated cohort, while also showing that evidence and coverage
are not perfectly invariant to timebase scale. No thresholds were changed.

This remains a synthetic sensitivity test, not a logger-accuracy estimate or a
clock-calibration method. It uses one truth convention, a finite set of generator
motions and seeds, and the package's own synthetic noise model; it does not measure
real clock behavior, field false-recommendation risk, or confidence calibration.
Reproduce it with:

```bash
set -e

root=.validation-corpus-clock-scale-all-motions-repro
python -m scripts.generate_validation_corpus \
  --outdir "$root" --motion-profile all --motion-seeds 730301 730302 730303 \
  --noise-profile all --noise-replicates 3 --noise-seed-start 20261040 \
  --noise-sample-counts 161 401 \
  --timestamp-clock-rate-errors-ppm -50000 -10000 -1000 0 1000 10000 50000
python -m scripts.validate_corpus "$root/manifest.json" \
  --workers 4 --progress --checkpoint "$root/assessment.checkpoint.json" \
  --checkpoint-every 100 --output "$root/assessment.json"
```

Recorded manifest and assessment SHA-256 values were
`09d061c7af49f4eb2050afc5889d5d306e561d4620b5aa53d8f5bf49535c2663` and
`f5468416e42b941ffc96398e4cdc7745551f58776093add6fab8fbc058af905d`, respectively.
The run used Windows 11, Python 3.14.6, NumPy 2.5.2, and SciPy 1.18.0. These labels
are generator-derived and are not independent logger evidence.

`--truth-rate-units radians_per_second degrees_per_second` adds paired noisy
cases written in both rate units from the same physical trajectory and seeded
noise realization. The manifest retains physical noise levels in rad/s and
enables both unit hypotheses for those cases. This is useful for checking unit
conversion symmetry, but the paired files are not independent Monte Carlo
replicates. The default remains rad/s only.

For a motion that is not built from closed-form attitude angles, select
`sampled-rate-nonanalytic`. It evaluates a seeded random Fourier mixture at the
sample times and generates quaternions independently of the inference package's
integration path. The standalone sampler's default path exponentiates average
endpoint rates. A separate DOP853 integration checks its convergence: for seed
`730201` over four seconds, the maximum attitude error was `7.46e-4`, `1.87e-4`,
and `4.66e-5 rad` at 50, 25, and 12.5 ms sample spacing, respectively. The test
repeats this check for three motion seeds and verifies second-order convergence.

Corpus generation uses a finer fixed 2.5 ms trapezoidal integration grid anchored
at `t = 0`, then evaluates requested samples with a partial endpoint step. The
path is independent of the selected channel-delay and timestamp-jitter values, so
orientation and rate samples share one absolute-time trajectory. A DOP853 test
checks this path across three seeds, jittered query times, and times before and
after the reference; maximum attitude error stays below `1e-5 rad` for that
tested four-second range. Channel latency is supported: a positive delay samples
that channel at `t - latency`, including pre-zero motion at the start of a log.
This tests generator accuracy, not recovery accuracy. Set `--motion-seed` to
reproduce the physical motion; noise and acquisition perturbations retain their
separate seeds. Noise, timestamp jitter, and row dropout remain available. The
result is still synthetic, not an external-system accuracy measurement.

The `sampled-rate-piecewise-linear` profile is a second non-Euler rate family. It
draws independent three-axis rate values at fixed 0.5 s knots and linearly
interpolates between them, then uses the same independent fixed-grid quaternion
integrator. Its fixed knot grid supports times from -60 s through 3,600 s and rejects
queries outside that range rather than silently extrapolating. The distinct rate
process is cross-checked against DOP853 and is available to the corpus and benchmark
CLIs; `--motion-seeds` applies to either sampled-rate family.

## Persistent residuals under sensor-frame mismatch

An out-of-model stress case uses 151 samples at 20 ms spacing from a seeded three-axis
Fourier rate signal. The attitudes are integrated from body-frame `pqr` rates, then the
reported rate vectors are rotated by 90 degrees about normalized axis `(0.3, -0.5, 0.8)`
without applying the same transform to the attitudes. This simulates an unmodeled fixed
sensor-to-body rotation; it is not one of the discrete convention changes the package
claims to identify.

Before the persistence guard, seed `703` produced a unique but incorrect
`world/rqp/+1` recommendation for data labeled `body/pqr/+1`. The winning residual had
lag-one correlation `0.9989` and estimated per-axis scale `0.00257 rad`; the i.i.d.
likelihood reported sufficient evidence despite that extreme dependence. Evidence is
now withheld when absolute residual lag-one correlation is at least `0.95` and the
estimated scale exceeds the configured `residual_floor_rad`. Near-exact numerical
residuals below that floor are exempt because their correlation estimate is not a
meaningful measurement-noise statistic. The checked-in regressions include the same 90-degree
defect on the independent sampled-rate generator and a second near-discrete axis-permutation
fixture. In the latter, an effectively disabled guard makes a false `body/rpq/+1`
recommendation for `body/pqr/+1` truth (`rho=0.999453`, residual scale `3.56e-5 rad`); the
default guard withholds evidence sufficiency and abstains. This is an executable false-
recommendation regression, though not the exact historical `world/rqp/+1` fixture above.

```bash
python -m pytest -q tests/test_sensor_mismatch.py
```

The current behavior is abstention, not recovery of the mounting rotation. Neither fixture
validates arbitrary axis misalignment, bias, scale error, or a general correlated-noise
model; milder or differently structured model mismatch remains an open false-recommendation
risk.

### Fresh-seed sensor-mismatch screen

The repeatable sensor-mismatch harness applies 21 fixed defects (mount rotations, per-axis
biases, scales, polarities, and clipping) plus a clean control to independently integrated
Fourier-rate motions. A ten-seed holdout (704--713), with 151 samples at 20 ms, produced 220
cases. Under the default persistence guard (`|lag-1 correlation| >= 0.95` and residual scale
above `1e-5 rad`), it made 17 correct recommendations, abstained on 203 cases, and made no
incorrect recommendations. Truth remained in the evidence contender set in 206/220 cases;
the guard fired in 196 cases. These are synthetic generator-derived labels, not a field
accuracy estimate.

An exploratory paired run with the guard effectively disabled produced 32 recommendations,
all correct in this same holdout, and 188 abstentions. Thus this cohort shows a coverage cost
from the guard but no false recommendation prevented by it. We retain the conservative guard
because a separate earlier synthetic fixture did produce a false recommendation under
extreme residual persistence; neither synthetic family is enough to calibrate its cutoff.
The harness records the exact guard cutoff and residual floor in its JSON output. The cutoff
override below is a validation-only sensitivity control, not a library configuration option.

Reproduce the default-policy screen with:

```bash
python -m scripts.validate_sensor_mismatch \
  --motion-seeds 704 705 706 707 708 709 710 711 712 713

# Paired sensitivity run with the guard disabled for this sweep only.
python -m scripts.validate_sensor_mismatch \
  --motion-seeds 704 705 706 707 708 709 710 711 712 713 \
  --persistence-guard-correlation-limit 1.1
```

The script emits a complete per-case JSON report to standard output. Do not treat its
recommendation or abstention counts as estimates for real sensors: all labels, perturbations,
and motions are generated, and only one motion family and cadence are represented.

### Mixed timing/calibration across rate-motion families

The harness also has an opt-in trio of mixed conditions combining channel latency with
misalignment/bias, misalignment/scale, or polarity/clipping. Positive latency samples that
channel at `t - latency`; orientation and rate queries use the same seeded physical motion
but their independently shifted query times. A fresh three-seed pilot (714--716) ran the
full 25-condition suite at 20 ms sampling on Fourier-mixture, piecewise-linear, and
maneuver-burst rate families:

| Motion profile | Cases | Correct recommendations | Abstentions | Wrong recommendations | Truth in evidence | Guard triggered |
|---|---:|---:|---:|---:|---:|---:|
| Fourier mixture | 75 | 7 | 68 | 0 | 68 | 68 |
| Piecewise-linear | 75 | 6 | 69 | 0 | 74 | 67 |
| Maneuver bursts | 75 | 6 | 69 | 0 | 72 | 66 |

All 27 mixed timing/calibration cases abstained for insufficient evidence, retained truth
in the evidence set, and triggered the persistence guard. This is useful behavior under
these defects, but not recovery of the actual mounting or timing parameters. The pilot had
no added measurement noise, jitter, or dropout, used only three fresh seeds per family, and
does not establish field accuracy or a false-recommendation rate.

Reproduce all three profiles with:

```bash
profiles=(
  sampled-rate-nonanalytic
  sampled-rate-piecewise-linear
  sampled-rate-maneuver-bursts
)
for profile in "${profiles[@]}"; do
  python -m scripts.validate_sensor_mismatch \
    --motion-seeds 714 715 716 --motion-profile "$profile" \
    --include-mixed-timing-calibration
done
```

### Paired three-tier sensor-mismatch screen with jitter and dropout

The 25-condition sensor-mismatch suite was crossed with three acquisition tiers.
They use orientation/rate standard deviations of `(0.0015 rad, 0.012 rad/s)`
(low), `(0.003 rad, 0.025 rad/s)` (moderate), and `(0.006 rad, 0.05 rad/s)`
(high). Across all tiers, the orientation and rate errors are OU processes with
correlation times `0.05 s` and `0.1 s`; uniform timestamp jitter is `±0.002 s`,
and 10% of whole rows are dropped. The quaternion error is left-multiplied
(world-frame rotation-vector error); rate noise is added after synthetic sensor
calibration fields and before clipping. Both channels share timestamps and the
retained-row mask. OU transitions use each actual jittered inter-sample interval.
The same motion/noise seeds, jitter, dropout, and defect conditions are paired
across tiers; only the noise standard deviations change.

Each tier contains three motion families × three motion seeds (714--716) × three
noise seeds (20314101--20314103) × 25 conditions: 675 trials per tier, 2,025
total. Each trial began with 151 nominal samples and retained 131--141 rows.
Results by tier and family:

| Noise tier | Motion profile | Cases | Correct recommendations | Abstentions | Wrong recommendations | Truth was data winner | Truth in evidence |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | Fourier mixture | 225 | 33 | 192 | 0 | 135 | 204 |
| Low | Piecewise-linear | 225 | 34 | 191 | 0 | 122 | 222 |
| Low | Maneuver bursts | 225 | 0 | 225 | 0 | 86 | 220 |
| Low total | — | 675 | 67 | 608 | 0 | 343 | 646 |
| Moderate | Fourier mixture | 225 | 26 | 199 | 0 | 134 | 204 |
| Moderate | Piecewise-linear | 225 | 11 | 214 | 0 | 122 | 222 |
| Moderate | Maneuver bursts | 225 | 0 | 225 | 0 | 82 | 225 |
| Moderate total | — | 675 | 37 | 638 | 0 | 338 | 651 |
| High | Fourier mixture | 225 | 16 | 209 | 0 | 133 | 208 |
| High | Piecewise-linear | 225 | 0 | 225 | 0 | 123 | 223 |
| High | Maneuver bursts | 225 | 0 | 225 | 0 | 76 | 225 |
| High total | — | 675 | 16 | 659 | 0 | 332 | 656 |
| All tiers | — | 2,025 | 120 | 1,905 | 0 | 1,013 | 1,953 |

The persistence guard did not trigger in any tier. Among the 243 mixed
timing/calibration trials (81 per tier), all abstained and truth stayed in the
evidence set. Evidence was sufficient but had multiple compatible contenders in
209 cases; 34 abstained for insufficient evidence. Truth was the data winner in
30/81 trials at each tier. Overall, stronger noise coincided with lower
recommendation coverage in this small paired cohort, while truth-in-evidence
counts rose slightly; that non-monotonic diagnostic behavior cautions against
interpreting these counts as a calibrated noise response. The 2,025 cases are not
independent replicates: they reuse just three motion seeds and three noise seeds
across conditions. No wrong recommendations here is not an estimated false-
recommendation rate, and these made-up noise parameters are not hardware
measurements or field validation.

Reproduce the screen and save each full per-case JSON report under an ignored
output directory with:

```bash
set -e

outdir=.validation-corpus/sensor-mismatch-ou-jitter-dropout
mkdir -p "$outdir"
for tier in low moderate high; do
  for profile in \
    sampled-rate-nonanalytic \
    sampled-rate-piecewise-linear \
    sampled-rate-maneuver-bursts; do
    python -m scripts.validate_sensor_mismatch \
      --motion-seeds 714 715 716 \
      --noise-seeds 20314101 20314102 20314103 \
      --motion-profile "$profile" \
      --acquisition-profile "$tier-ou-jitter-dropout" \
      --include-mixed-timing-calibration \
      > "$outdir/$profile-$tier.json"
  done
done
```

Manifest and assessment SHA-256 values are:

```text
profile cadence manifest assessment
fourier 20 Hz 08b6bfc8073159b8ee675d4df6153424653a30e156a679e3f9bb1e5ce1210af7 286cdb3c91689eeef5d2cf25480bf0c922f5164f778ad12cf52ea4cebf4bae69
fourier 40 Hz 27f72139a4f9b1a076eb6bca3b30421adb4b28986391324c1ac4fb6dcb8d543f 53ff0ab8eb9590969045c11b168bbf0c1647411ee9c346698e0aa1e16b808cc0
fourier 100 Hz 196acbc5e0c0900e63db57b404fb701e1caadfccf9cfdfe7252e785baf86891b 162970c7edefe463bbf594e2ba9ee7c44e8bc94000513c0996370b979f8e985e
piecewise 20 Hz d044a905f1e019753f29026e21ca827d19f6c26e24ab29a97048507d32667e41 bd039fd1a003dba6d4312644c45f64ffd1786f5a37f737f8310ea90964f99cd1
piecewise 40 Hz 1a6ea4f2ff521ff81333198a52b9323c9f7ae381b64efe69eb11828453a7d52b 5b21793ab857c4e8fa0b97845069953af9f67ac9b8a4b2370d93e2906c3a6c45
piecewise 100 Hz e97be4210c104b8580a8d1ebc1dedc30934572f6293b07b8824fdb3a014f1951 3b78c31cc9c55ec629577f148c7e124499a47bc4f8941a8207378a83bad2c1a6
bursts 20 Hz 57a8607215fcba0da2141bfb6a889bbea026452fe7d5c941126cca9e1ac092e4 1f6c3b00b1ee41703465bf4055ef00bc1840ea6f0d6ee7833c970d00f81f53e5
bursts 40 Hz 061336f2c420330b79e6918f9b85cb0c4d22ab685dcfbdd9d14291f9d230edec 9c6e4ccaefa0ee219aa1c01d31eb342d0402374a0baf5ff6c138c912be19a528
bursts 100 Hz 61b21ef6c5463cc0bee88d4691bb70a40dbf1ff08673d465162e69d513537fe0 ba338e182ef6bee518e155febc465dd4168638fca48485b73e5f00533479dac8
```

The channel-isolation manifests/assessments were:

```text
clean            aac2b81e4447f9cdba264b4671f784f22ad5210b2517df57df8a857ebc687add 45ae1b7e5049e7c0bc3a37f50a4358853e955ef71b67cf9dc69ea97494d60d52
orientation-only 077d73e1d46ddba6e0f157102c2aec1ba264b376cec81ff3b4d7d64bfeaf53b0 c047f83b682471cc9fc0ea8bddfb7e12c90c72f17e70fb99ce00e7c614958239
rate-only        81ee713aae0b0c6ad28e60cb4ba8fb50f710c7aed256101ce4a578802786c3c3 11898156273aef8ab05e354e949b0053ccb2c133396a5340b1c4530b2f658ac4
```

Generator SHA-256: `ff497728be95164114b4d0a35494da54adf34a20fc3848d1c89761290525b306`
OU helper SHA-256: `a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`
Validator SHA-256: `a9e8ed8d7086c20176d2bbd24be99891e55f753e0ca13c7ab3b4b304d4c27752`
Package plus validator implementation SHA-256:
`76c065ead2c5c7c011dfd268355f6d10ea164df6037c450685e582edddfabd3d`

The assessments were produced on Windows 11 with Python 3.14.6, NumPy 2.5.2,
SciPy 1.18.0, and package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261101`.

### Fresh-seed replication with noise-free acquisition controls

To check seed sensitivity, a second paired batch used five fresh motion seeds
(744701--744705) and five fresh noise seeds (20414101--20414105). Each noisy tier
crossed all 25 conditions with the three motion families, giving 625 cases per
family and 1,875 per tier. A paired noise-free acquisition control used the same
five motion seeds and one nominal noise seed (20414100), giving 125 cases per family.
The three noise tiers reuse the same noise paths, jitter, dropout, and motion
seeds; only the configured standard deviations change. Each complete JSON report
is saved in the ignored `.validation-corpus/sensor-mismatch-fresh-seed-replication`
directory. The reports record package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261101`,
Python 3.14.6, NumPy 2.5.2, and SciPy 1.18.0. All runs used 151 samples at 20 ms,
2.5 ms attitude-integration steps, and the default persistence guard (`|rho| >= 0.95`,
residual floor `1e-5 rad`).

| Acquisition tier | Cases | Correct recommendations | Abstentions | Wrong recommendations | Truth among data-winning classes | Truth in evidence | Persistence guard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noise-free acquisition | 375 | 27 | 348 | 0 | 227 | 354 | 328 |
| Low | 1,875 | 198 | 1,677 | 0 | 1,140 | 1,772 | 1 |
| Moderate | 1,875 | 149 | 1,726 | 0 | 1,128 | 1,792 | 0 |
| High | 1,875 | 72 | 1,803 | 0 | 1,106 | 1,825 | 0 |

Correct recommendations depended strongly on motion family:

| Motion profile | Noise-free (125) | Low (625) | Moderate (625) | High (625) |
|---|---:|---:|---:|---:|
| Fourier mixture | 11 | 120 | 107 | 54 |
| Piecewise-linear | 9 | 78 | 42 | 18 |
| Maneuver bursts | 7 | 0 | 0 | 0 |

The no-mismatch condition separates acquisition noise from sensor defects. The
noise-free control correctly recommended all 15/15 trajectories. Under the paired
low/moderate/high acquisition tiers, it recommended 49/75, 37/75, and 15/75
trajectories, respectively, with no wrong recommendations. Truth remained in the
evidence set in all 225 noisy no-mismatch trials; at high noise, truth was outside
the data-winning set in two trials but remained an evidence contender.

Every recommendation in this seed batch was correct, but the data-winning set
contained the labeled truth in only 3,374/5,625 noisy cases; the evidence
contender set retained truth in 5,389/5,625. Abstention reasons were dominated by
multiple evidence contenders at low/moderate noise (1,602/1,556 cases), shifting
toward insufficient evidence at high noise (424 cases). The persistence guard
fired only once across the 5,625 noisy cases, versus 328/375 noise-free controls.
This is descriptive of the selected synthetic cohort, not a false-recommendation
rate or evidence that the guard is calibrated under measured sensor noise.

All cases where truth left the evidence set (103 low, 83 moderate, 50 high) had
best normalized RMSE above `0.5`, which triggers the package's existing generic
large-residual warning. The new focused regression verifies that a representative
out-of-model axis-polarity case emits that warning alongside the multiple-hypothesis
warning and abstains. Four additional low-tier cases had truth outside the
data-winning set while normalized RMSE was at or below `0.5`; they remained in the
evidence set. Thus the residual warning exposed every evidence miss in this batch,
but it is not a complete detector for candidate-ranking weakness or a sensor
calibration diagnosis.

In the pinned high-noise fixture (motion seed 744701, noise seed 20414101), the
truth `body/pqr/+1` class ranked 16th by data-only fit, outside the configured
near-tie margin of 2.0; five other classes remained evidence contenders. The best
fit was `body/qpr/+1` with normalized RMSE `0.702`. Evidence itself was sufficient,
but the package abstained because several candidates remained, while warning about
both ambiguity and poor fit. The arbitrary per-axis polarity transform is outside
the convention hypothesis space. A different tie-break rule would not constitute
recovery here; estimating the transform would require extending and validating the
model explicitly.

Across mixed timing/calibration conditions, all 225 cases per tier abstained. The
Truth was in the data-winning set in 103/225, 102/225, and 103/225 low/moderate/high
trials; truth stayed in evidence in 205/225, 205/225, and 211/225. There were
209/197/176 cases with sufficient evidence but multiple contenders, and no guard
triggers. Fresh seeds therefore did not reproduce the earlier three-seed pilot's
81/81 truth retention in mixed cases. The severe `axis-polarity-04` and
`axis-rotation-150deg` defects had no truth-containing data winner in any of the
75 trials per tier, although truth remained among the evidence contenders in
47--64/75; all such cases abstained.

The cases within a tier are not independent Bernoulli trials: the same five
noise paths and motion paths are paired across defect conditions and noise
levels. This five-seed study still cannot calibrate rates, explain the ranking
misses, or support threshold changes. The previous three-seed results remain
above as an exploratory pilot rather than being silently replaced.

Reproduce the clean controls and all paired noisy tiers with:

```bash
set -e

outdir=.validation-corpus/sensor-mismatch-fresh-seed-replication
mkdir -p "$outdir"
profiles=(
  sampled-rate-nonanalytic
  sampled-rate-piecewise-linear
  sampled-rate-maneuver-bursts
)
for tier in clean low moderate high; do
  if [ "$tier" = clean ]; then
    acquisition=clean
    noise_seeds=(20414100)
  else
    acquisition="$tier-ou-jitter-dropout"
    noise_seeds=(20414101 20414102 20414103 20414104 20414105)
  fi
  for profile in "${profiles[@]}"; do
    python -m scripts.validate_sensor_mismatch \
      --motion-seeds 744701 744702 744703 744704 744705 \
      --noise-seeds "${noise_seeds[@]}" \
      --motion-profile "$profile" \
      --acquisition-profile "$acquisition" \
      --include-mixed-timing-calibration \
      > "$outdir/$profile-$tier.json"
  done
done
```

## Paired sampled-rate channel-latency pilot

On 2026-10-30, two independently seeded rate motions were evaluated at 161 samples
and 25 ms spacing, with eight paired noise seeds per low/moderate/high noise level.
Orientation and rate delays were independently set to either 0 or 100 ms, producing
192 noisy trajectories across the four delay pairs. Reproduce the cohort with a
fresh output directory:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-sampled-rate-latency-100ms-pilot \
  --noise-profile all --noise-replicates 8 --noise-seed-start 20802030 \
  --noise-sample-counts 161 --motion-profile sampled-rate-nonanalytic \
  --motion-seeds 2947 2948 \
  --orientation-latencies-s 0 0.1 --rate-latencies-s 0 0.1
python -m scripts.validate_corpus \
  .validation-corpus-sampled-rate-latency-100ms-pilot/manifest.json \
  --output .validation-corpus-sampled-rate-latency-100ms-pilot/assessment.json
```

All 40 recommendations were correct; the other 152 cases abstained with truth
retained in the evidence contenders. There were no data-winner, evidence, or
operational misses. At low noise, matched delays (0/0 and 100/100 ms) each yielded
16/16 recommendations; mismatched delays yielded 8/16 for orientation/rate
0/100 ms and 0/16 for 100/0 ms. All moderate- and high-noise cases abstained in
this seed block. This is a timing-path regression study over two synthetic motions,
not evidence of real logger delay-detection accuracy or confidence calibration.

Use `--motion-seeds 1947 1948 1949` to repeat the same acquisition/noise seeds
across multiple independently generated sampled-rate motions. The manifest
records each `motion_seed`, uses distinct CSV names, and keeps each motion seed
in its own assessment condition summary. This enables paired robustness checks
across physical trajectories; it does not make the synthetic cohort a substitute
for external logs.

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-sampled-rate \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --noise-profile low --noise-replicates 4 --noise-sample-counts 161 \
  --timestamp-jitter-amplitudes-s 0 0.002 --dropout-probabilities 0 0.1
python scripts/validate_corpus.py \
  .validation-corpus-sampled-rate/manifest.json \
  --output .validation-corpus-sampled-rate/assessment.json
```

## Paired motion-seed stress run

On 2026-10-30, the sampled-rate generator was exercised with three independently
seeded trajectories and 20 paired noise seeds (`20261030`–`20261049`) at each of
the low, moderate, and high noise levels. Each noisy log had 161 samples at
25 ms spacing. The same noise/acquisition seeds were reused across motion seeds;
condition summaries kept each physical trajectory separate. Reproduce the
180-case noisy cohort with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-motion-seed-sweep \
  --noise-profile all --noise-replicates 20 --noise-seed-start 20261030 \
  --motion-profile sampled-rate-nonanalytic \
  --motion-seeds 730201 1947 1948 --noise-sample-counts 161
python -m scripts.validate_corpus \
  .validation-corpus-motion-seed-sweep/manifest.json \
  --output .validation-corpus-motion-seed-sweep/assessment.json
```

| Motion seed | Low recommendations / 20 | Moderate recommendations / 20 | High recommendations / 20 |
|---:|---:|---:|---:|
| 1947 | 20 | 0 | 0 |
| 1948 | 20 | 20 | 0 |
| 730201 | 20 | 20 | 0 |

The low, moderate, and high orientation/rate noise standard deviations are
`(0.0015 rad, 0.012 rad/s)`, `(0.003 rad, 0.025 rad/s)`, and
`(0.006 rad, 0.05 rad/s)`. Across all 180 cases, the data winner matched truth
and truth remained in the evidence contenders; all 100 recommendations were
correct, while the other 80 cases abstained without an evidence miss. Moderate
noise exposed motion-seed sensitivity in coverage (0–100% by trajectory); high
noise led all three motions to abstain. This is a small, single-convention,
synthetic stress run—not a field accuracy rate, probability calibration, or
evidence about external loggers.

For example, this compares baseline and perturbed acquisition conditions with paired
seeds at two durations:

```bash
python scripts/generate_validation_corpus.py \
  --noise-profile low --noise-replicates 30 --noise-sample-counts 161 401 \
  --timestamp-jitter-amplitudes-s 0 0.002 \
  --dropout-probabilities 0 0.1 \
  --orientation-latencies-s 0 0.01 --rate-latencies-s 0 0.025
```

This produces 32 conditions per seed (two durations × two jitter levels × two
dropout levels × two orientation delays × two rate delays), or 960 noisy cases for
30 seeds. Start with fewer factors or replicates when exploring because the sweep
grows multiplicatively. These remain controlled synthetic stress tests: they can
expose sensitivity and implementation defects, but they do not model every logger
or substitute for independently labeled real trajectories.

```bash
python scripts/generate_validation_corpus.py \
  --noise-profile all --motion-profile all --noise-replicates 30 \
  --noise-sample-counts 161 401
python scripts/validate_corpus.py .validation-corpus/manifest.json
```

For larger assessments, add `--progress` to report completed cases, throughput,
and an estimated remaining time to stderr every 100 cases. Assessment JSON on
stdout or in `--output` remains unchanged. Independent cases can also be evaluated
in separate processes with `--workers N`; the default is one worker, and completed
results remain in manifest order. Checkpoint writes stay in the parent process, so
parallel runs can still be resumed with a different worker count. Each worker
loads its own numerical runtime, so increase `N` conservatively when memory is
limited.

The motion profiles are rich three-axis, low-excitation, single-axis, near-gimbal,
and sampled-rate-nonanalytic trajectories. The assessment groups replicates by
representation, motion profile, and measurement conditions; it reports evidence,
data-winner, and operational miss rates, plus indeterminate-confidence and
evidence-insufficiency rates. It also
records the per-case noise estimate, residual lag-one autocorrelation, evidence
prefix/likelihood diagnostics, and inference warnings; condition summaries include
the mean residual lag-one autocorrelation. Python/platform/NumPy/SciPy/package
versions are recorded as well. The typed `report.decision`
recommends a class only when evidence is sufficient and one data-only contender remains;
otherwise it abstains with a reason. The report still preserves its deterministic
`operational_class_id` as a best candidate for diagnostics, not as confirmation. These
paired synthetic trials are not accuracy estimates for real loggers or independent
recording sessions. The assessment records each decision and reports recommendation
coverage, accuracy among recommendations, and wrong-recommendation counts separately
from the always-present operational class metrics.

## Observed synthetic snapshot

On 2026-10-30, commit `3e9173c` was evaluated with 30 paired seeds for each of four
motion profiles and three noise levels (360 noisy cases total). Each trajectory had
161 samples at 25 ms spacing. All 360 true classes remained in the evidence contender
set, but 171 operational classes differed from truth; 280 cases were evidence-
insufficient, all 360 confidence labels were indeterminate, and no incorrect result
was labeled evidence-sufficient. The table gives operational misses / evidence-
insufficient cases; both counts are out of 30 per cell.

| Motion profile | Low noise | Moderate noise | High noise |
|---|---:|---:|---:|
| Low excitation | 11 / 30 | 16 / 30 | 24 / 30 |
| Near gimbal | 2 / 30 | 14 / 30 | 23 / 30 |
| Rich three-axis | 0 / 0 | 0 / 0 | 0 / 10 |
| Single axis | 27 / 30 | 27 / 30 | 27 / 30 |

Noise levels are orientation/rate standard deviations of `(0.0015 rad, 0.012 rad/s)`,
`(0.003 rad, 0.025 rad/s)`, and `(0.006 rad, 0.05 rad/s)`, respectively. The run
used Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev16+g3e9173cb8`. This is a reproducible point-in-time synthetic
diagnostic, not a real-world success rate, confidence calibration, or release gate.
It demonstrates that an operational class can be wrong even while truth remains a
plausible contender; consumers should inspect evidence sufficiency, confidence, and
the contender set rather than treating the operational class as confirmed.

## Duration-aware decision snapshot

The evidence-gated decision was evaluated on commit `c54c51d` with 30 paired seeds,
four motion profiles, three noise levels, and 4-second/10-second recordings: 720 noisy
cases plus the three original control cases. In the rich-three-axis, low-noise
condition, the decision abstained on all 30 four-second trials and recommended the
correct class on all 30 ten-second trials:

| Duration | Recommendations | Correct recommendations | Abstentions |
|---|---:|---:|---:|
| 4 seconds | 0 / 30 | — | 30 / 30 |
| 10 seconds | 30 / 30 | 30 / 30 | 0 / 30 |

Across all 720 noisy cases, 30 were recommended and 690 abstained; none of the 30
recommendations was wrong (4.17% noisy-case coverage, 100% accuracy among those
recommendations). The three control cases were also recommended correctly. All other
noisy motion/noise/duration combinations abstained. This demonstrates useful sensitivity
to recording duration for one informative synthetic condition, but the overall coverage
is intentionally low and does not establish real-world precision or calibration.

The run used Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev18+gc54c51d8d`. The counts are synthetic diagnostic results only; do
not interpret the 30/30 outcome as a statistical guarantee.

## Acquisition-impairment snapshot

Commit `6400de0` was evaluated on rich-three-axis trajectories at 4 seconds
(161 scheduled samples, 25 ms spacing), using 30 paired noise seeds. The first
acquisition sweep crossed timestamp jitter (0 or ±5 ms), whole-row dropout
probability (0 or 0.35), orientation delay (0 or 50 ms), and rate delay (0 or 50 ms):
16 conditions per seed, or 480 noisy cases plus three controls. The table combines
the two jitter levels; each row therefore contains 60 cases. The run used Python
3.14.6, NumPy 2.5.2, SciPy 1.18.0, and package version
`0.0.1a0.post1.dev20+g6400de09c`.

| Dropout p | Orient delay | Rate delay | Trials | Recommended | Abstain | Wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 ms | 0 ms | 60 | 0 | 60 | 0 |
| 0 | 0 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0 | 50 ms | 0 ms | 60 | 0 | 60 | 0 |
| 0 | 50 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0.35 | 0 ms | 0 ms | 60 | 26 | 34 | 0 |
| 0.35 | 0 ms | 50 ms | 60 | 0 | 60 | 0 |
| 0.35 | 50 ms | 0 ms | 60 | 2 | 58 | 0 |
| 0.35 | 50 ms | 50 ms | 60 | 20 | 40 | 0 |

Across the 480 noisy trials, 48 recommendations were correct and 432 cases abstained;
there were no evidence misses or operational-class misses. All noisy cases met the
current evidence-sufficiency test, but every case's separate heuristic confidence
label remained `indeterminate`. In 240 paired comparisons of no dropout versus 35%
dropout, 48 contender sets changed from two classes to one, always retaining truth;
the other 192 remained two-class sets. The ±5 ms jitter changed no contender set or
decision in the 240 paired jitter comparisons. At 35% requested dropout, the generator
retained an average of 105.8 of 161 samples because duration endpoints are protected.

A plausible generator-specific explanation is that longer gaps let the integrated
orientation change stand out from independent per-sample measurement noise. That is
not evidence that real dropout improves inference: this is one analytic motion and
white-noise model, and the inference diagnostics warn that adjacent residuals are
correlated while the evidence calculation uses an approximate i.i.d. likelihood.
More importantly, the apparent benefit disappears at higher noise:

| Noise profile | Dropout probability | Trials | Recommendations | Abstentions |
|---|---:|---:|---:|---:|
| Low | 0 | 60 | 0 | 60 |
| Low | 0.35 | 60 | 26 | 34 |
| Moderate | 0 | 60 | 0 | 60 |
| Moderate | 0.35 | 60 | 0 | 60 |
| High | 0 | 60 | 0 | 60 |
| High | 0.35 | 60 | 0 | 60 |

This companion 4-second run crossed all three noise profiles with the same two jitter
and dropout levels over 30 seeds (360 noisy cases plus controls). The 26 recommendations
were all correct and occurred only at low noise with 35% dropout. All moderate- and
high-noise cases abstained; the 29 evidence-insufficient cases were all high-noise.
These paired synthetic results help expose sensitivity, but do not establish
real-world accuracy, logger robustness, or confidence calibration.

## Temporally correlated measurement-noise stress

The independent generator also supports stationary Gaussian AR(1) perturbations
for the three orientation rotation-vector components and three angular-rate
components. `--orientation-noise-ar1-coefficients` and
`--rate-noise-ar1-coefficients` accept one or more sample-lag coefficients
`rho` in `(-1, 1)`. A coefficient of zero is white noise and preserves the
generator's previous seeded sequence exactly. Positive values model slowly
varying measurement error; negative values model alternating error. Each
coefficient is the correlation between adjacent scheduled samples, not a
continuous-time constant. Marginal standard deviations remain those of the
selected noise profile. The manifest and assessment retain both coefficients,
and summaries keep different correlation conditions separate.

Noise is generated on the scheduled sample grid before timestamp jitter and row
dropout are applied. Thus, after a row is removed, adjacent observed rows may
span multiple scheduled lags. Seeds and underlying innovation draws are paired
across coefficient levels, as well as across the existing profile and duration
factors. For example, this compares white and positively correlated orientation
and rate noise over 30 paired 4-second trials per combination:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-ar1-30seed \
  --noise-profile all --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python scripts/validate_corpus.py \
  .validation-corpus-ar1-30seed/manifest.json \
  --output .validation-corpus-ar1-30seed/assessment.json
```

The factorial sweep produces 12 correlation/noise-profile conditions, or 360
noisy cases. AR(1) is a controlled stress model, not a claim that every sensor's
errors follow this process. The evaluator's likelihood still has approximate
i.i.d. assumptions, so these runs are specifically useful for checking whether
evidence sufficiency or recommendations become misleading under temporal
dependence. Report any such result as a synthetic warning; do not tune the
inference thresholds solely to fit this generator.

### Observed AR(1) snapshot

On 2026-10-30, 30 seeds were evaluated for each noise profile and each paired
orientation/rate coefficient combination at 161 samples (4 seconds): 360 noisy
trials plus three control cases. `rho=0.8` was compared with white noise
`rho=0` independently on each channel. All 360 true classes remained in their
evidence contender sets; there were no operational misses, evidence misses, or
incorrect evidence-sufficient cases. The recommendation counts below are out of
30 per condition; all recommendations were correct.

| Noise profile | `(rho_orientation, rho_rate) = (0, 0)` | `(0, 0.8)` | `(0.8, 0)` | `(0.8, 0.8)` |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 27 | 14 |
| Moderate | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |

Across the 360 noisy trials, the evaluator recommended 41 and abstained on 319;
all 360 heuristic confidence labels remained indeterminate. The 32
evidence-insufficient cases occurred in high-noise conditions. Correlated
orientation noise therefore changed decision coverage substantially at low
marginal noise, despite no false recommendation in these 30-seed cells. That is
a useful warning about sensitivity to dependence, not proof of calibration or
overconfidence: this finite synthetic run cannot establish the false-positive
rate, and the evidence calculation still uses an approximate i.i.d. likelihood.
The three control cases bring the complete assessment to 44 recommendations
(including those controls) and 319 abstentions. The run used Python 3.14.6,
NumPy 2.5.2, SciPy 1.18.0, and inference package version
`0.0.1a0.post1.dev20+g6400de09c`.

The generator's `rho` values describe raw per-sample measurement perturbations;
the inference diagnostic estimates lag-one correlation on interval rotation
residuals, so those numbers are not expected to match. In fact, differencing
independent orientation errors naturally induces negative adjacent interval
correlation. In the low-noise cells, the mean measured residual correlation and
number of cases carrying the correlation warning were:

| Requested `(rho_orientation, rho_rate)` | Mean interval-residual lag-1 | Warning cases |
|---|---:|---:|
| `(0, 0)` | -0.476 | 30 / 30 |
| `(0, 0.8)` | -0.463 | 30 / 30 |
| `(0.8, 0)` | -0.084 | 0 / 30 |
| `(0.8, 0.8)` | -0.042 | 0 / 30 |

For an ideal scalar AR(1) orientation perturbation, first differences alone
have lag-one correlation `-(1-rho)/2` (approximately -0.5 at `rho=0` and -0.1
at `rho=0.8`). The observed estimates are close to that pattern, with small
effects from rate noise and the full rotation residual calculation. This makes
the warning interpretable as a property of interval residuals, not a direct
sensor-noise-correlation estimate; it also reinforces that the evidence model's
independence assumption needs a more suitable residual covariance model before
its likelihood ratio can be treated as calibrated.

### 200-seed low-noise follow-up

To reduce Monte Carlo uncertainty in the only noise profile that produced
recommendations in the preceding 30-seed sweep, 200 paired seeds were run for
each of the same four correlation conditions at 4 seconds (800 noisy cases plus
three controls). Truth remained in
every evidence contender set; there were no evidence misses, operational
misses, or insufficient-evidence cases. The table reports recommendations out
of 200, the mean fixed-noise log-likelihood ratio, and the mean interval-residual
lag-1 estimate:

| Requested `(rho_orientation, rho_rate)` | Recommendations | Mean log LR | Mean interval-residual lag-1 |
|---|---:|---:|---:|
| `(0, 0)` | 0 / 200 | 88.93 | -0.466 |
| `(0, 0.8)` | 0 / 200 | 88.35 | -0.452 |
| `(0.8, 0)` | 196 / 200 | 425.50 | -0.066 |
| `(0.8, 0.8)` | 89 / 200 | 411.47 | -0.018 |

All 285 recommendations were correct, and the remaining 515 noisy cases
abstained. With zero observed errors, the one-sided 95% exact-binomial upper
bounds on the conditional error rate are 1.52% for the 196 recommendations and
3.31% for the 89 recommendations. These bounds apply only to these
generator-defined conditions; they are not a real-world precision guarantee.
The large shift in log LR and recommendation coverage under correlated
orientation noise is now reproducible at 200 seeds, so dependence-aware evidence
is a substantive next correctness task—not a threshold-tuning exercise.
The assessment is saved locally at
`.validation-corpus-ar1-low-200seed/assessment.json` (Git-ignored) and can be
regenerated with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-ar1-low-200seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 200 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python scripts/validate_corpus.py \
  .validation-corpus-ar1-low-200seed/manifest.json \
  --output .validation-corpus-ar1-low-200seed/assessment.json
```

### First-order colored-noise likelihood diagnostic

The package's likelihood currently treats interval residuals as i.i.d., while
sampled orientation noise is differenced and rate noise is integrated across
intervals. To quantify how that assumption changes the evidence scale, the
`scripts.compare_correlated_evidence` diagnostic compares the package score with
a first-order temporal-covariance reference. It is an offline synthetic-data
analysis only; it does not change inference or recommendations.

For uniform synchronized sampling, small inter-sample rotations, and independent
isotropic AR(1) measurement noise, write the interval error per axis as
`r[i] = (e[i+1] - e[i]) - (dt/2) * (w[i+1] + w[i])`, where `e` is attitude
measurement error and `w` is rate measurement error. For lag `k`, its temporal
covariance is:

```text
C[0] = 2*sigma_e^2*(1-rho_e) + (dt^2/2)*sigma_w^2*(1+rho_w)
C[k] = -sigma_e^2*(1-rho_e)^2*rho_e^(k-1)
       + (dt^2/4)*sigma_w^2*(1+rho_w)^2*rho_w^(k-1),  k > 0
```

The reference uses this shared Toeplitz covariance across the three axes and
compares the best competitor's whitened residual sum of squares with the
data-winner's. It uses known synthetic noise parameters from the manifest—not
estimated parameters—and checks the formulas against Monte Carlo interval-noise
draws. The assumptions do not cover timestamp jitter, dropout, channel latency,
anisotropic noise, or large rotations; eligible cases are rejected if they have
unsupported acquisition effects. This is not a full nonlinear sensor likelihood.

Reproduce the bounded comparison on the existing 200-seed low-noise corpus with:

```bash
python -m scripts.compare_correlated_evidence \
  .validation-corpus-ar1-low-200seed/manifest.json \
  --output .validation-corpus-ar1-low-200seed/colored-evidence-50seed.json \
  --max-cases-per-condition 50
```

On 2026-10-30, the first 50 seeds in each of the four `(rho_orientation,
rho_rate)` conditions (200 trajectories total) gave:

| `(rho_orientation, rho_rate)` | Package i.i.d. mean log LR | Known-noise i.i.d. reference mean log LR | Colored-reference mean log LR | Covariance condition number | Recommendations / 50 |
|---|---:|---:|---:|---:|---:|
| `(0, 0)` | 87.60 | 83.66 | 3896.80 | 99.06 | 0 |
| `(0, 0.8)` | 86.67 | 82.65 | 473.85 | 18.49 | 0 |
| `(0.8, 0)` | 422.29 | 402.42 | 3400.35 | 10.69 | 46 |
| `(0.8, 0.8)` | 408.13 | 385.86 | 462.99 | 1.23 | 21 |

The top-ranked data class matched truth in all 200 cases, and all 67 package
recommendations were correct; the other 133 cases abstained. The colored score is
not uniformly a discount: in this synthetic set it is roughly 1.2–46 times the
package i.i.d. score. Temporal covariance changes which residual patterns are
penalized, and the large values merit investigation rather than a claim of
improved calibration. This experiment uses manifest-specified noise parameters
and a linearized shared covariance, scores only the complete recording, and does
not test prefix stability or calibrate a recommendation threshold. Do not use
these colored scores as production evidence or tune the current decision
thresholds against this generator.

### Full-pipeline residual covariance check

The companion `scripts.validate_correlated_residual_model` check scores the
known convention and six one-field counterfactuals (attitude direction,
quaternion storage order, rate frame, rate order, rate sign, and rate unit),
through CSV loading, quaternion decoding, rate integration, and the scorer's
rotation-log residuals. It uses known noise parameters from the manifest. For a
rate-unit counterfactual, it scales the rate-noise standard deviation by the
candidate/truth unit-conversion ratio and includes each model's Gaussian
log-determinant term; the other counterfactuals currently share the truth
covariance as a first-order approximation. Positive pairwise values favor truth
under those stated reference models; they are not candidate-specific likelihoods
for every transformation or calibrated package likelihoods. The report also
retains empirical 3x3 covariance blocks by lag and their paired differences from
the truth blocks, exposing transformations that change the residual noise
structure. These differences are estimated in-sample and are descriptive, not
held-out model validation. Use a fresh output filename for each run because the
CLI does not overwrite reports.
Reproduce the 200-seed-per-condition run with:

```bash
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-ar1-low-200seed/manifest.json \
  --output .validation-corpus-ar1-low-200seed/residual-counterfactuals-200seed-rerun.json \
  --max-cases-per-condition 200
```

On 2026-10-30, all 800 trajectories (200 in each correlation condition) had
161 samples at 25 ms spacing, the rich-three-axis motion, and low measurement
noise. Covariance errors are normalized by the predicted lag-zero variance
`C[0]`; this is more stable than a relative ratio at lags where the predicted
covariance is near zero.

| `(rho_orientation, rho_rate)` | Observed / predicted lag-0 variance | Observed / predicted lag-1 covariance | Maximum absolute lag error / `C[0]` | Centered whitened sum of squares / dof | Mean residual RMS / `sqrt(C[0])` |
|---|---:|---:|---:|---:|---:|
| `(0, 0)` | 1.0054 | 1.0024 | 0.00544 | 1.00925 | 0.12265 |
| `(0, 0.8)` | 1.0056 | 1.0012 | 0.00558 | 1.00871 | 0.12262 |
| `(0.8, 0)` | 1.0078 | 0.9855 | 0.00776 | 1.01108 | 0.12641 |
| `(0.8, 0.8)` | 1.0082 | 0.8252 | 0.00824 | 1.00877 | 0.12822 |

The full-pipeline residual noise covariance is close to the first-order model in
this low-noise condition: lag-0 variance error is at most 0.83%, the largest
absolute error across lags 0, 1, 2, 4, 8, and 16 is 0.83% of `C[0]`, and the
zero-lag cross-axis covariance error is below 1.05% in every condition. The
apparently larger relative lag-1 discrepancy at `(0.8, 0.8)` corresponds to an
absolute error of only 0.31% of `C[0]`, because those two noise sources nearly
cancel at lag one. Covariance is estimated after centering each interval across
replicates; the replicate-mean residual is reported separately and includes
finite-sample variability. The validator also reports uncentered whitened
energy so that the mean-residual contribution is not hidden.

In the original direction-only run, the alternative's centered covariance also
remained close to the truth model: its maximum absolute lag error was 0.58%,
0.61%, 0.78%, and 1.01% of `C[0]` across the four conditions, respectively;
centered whitened energy was 1.009–1.011 per degree of freedom. Its
replicate-mean residual was 9.29–20.44 times `sqrt(C[0])`, so the separation is
primarily a nonzero convention mismatch rather than a changed noise scale. Each
of the 800 paired comparisons ranked truth above this alternative under both
truth-referenced scores. Mean score differences were:

| `(rho_orientation, rho_rate)` | Truth-model i.i.d. mean score difference | Truth-model colored mean score difference | Truth wins |
|---|---:|---:|---:|
| `(0, 0)` | 6949.63 | 317316.78 | 200 / 200 |
| `(0, 0.8)` | 6897.66 | 39961.80 | 200 / 200 |
| `(0.8, 0)` | 33423.45 | 258887.63 | 200 / 200 |
| `(0.8, 0.8)` | 32209.36 | 38536.26 | 200 / 200 |

These very large values are for one deliberately wrong direction-only
counterfactual under the manifest-specified noise parameters; they are not posterior
odds, calibrated confidence, or the package's multi-candidate recommendation
score.

#### Candidate covariance-block comparison

For the 200-seed rich-three-axis, low-noise run above, the report compares each
counterfactual's empirical 3x3 lag blocks against the truth's blocks, normalized
by truth `C[0]`. The table reports the maximum Frobenius difference over lags
0, 1, 2, 4, 8, and 16 and the four correlation conditions. These are paired,
in-sample estimates from the same noise replicates:

| Counterfactual | Maximum block difference / truth `C[0]` |
|---|---:|
| Attitude direction | 1.99% |
| Quaternion storage order | 5.72% |
| Rate frame | 1.70% |
| Rate order reversal | 1.38% |
| Rate sign | 1.46% |
| Rate unit, adjusted | 14.52% |

The rate-unit change is expected to alter the covariance scale. The quaternion
storage-order result shows that treating the truth covariance as an oracle for
every alternative is not justified, even in this controlled low-noise cohort.
In the broader 30-seed, 48-cell sweep, per-cell maximum differences reached
12.5% for quaternion order, 12.4% for direction, and 12.1% for rate frame; the
smaller replicate count makes those estimates noisier. None of these in-sample
comparisons calibrates likelihood magnitudes.

#### Even-seed covariance fit, odd-seed prediction

The validator can fit each candidate's empirical 3x3 lag-covariance blocks on
even noise seeds and compare them with held-out odd-seed blocks. This is a
covariance-transfer diagnostic, not a likelihood score. Reproduce it on the
same low-noise corpus with a new report path:

```bash
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-ar1-low-200seed/manifest.json \
  --output .validation-corpus-ar1-low-200seed/heldout-covariance-fit-rerun.json \
  --max-cases-per-condition 200
```

On 2026-10-30, the fit used 100 even seeds per each of four AR(1) conditions
(400 cases total) and evaluated on 100 disjoint odd seeds per condition (400
held-out cases). The table shows mean held-out root-mean-square block error
across lags 0, 1, 2, 4, 8, and 16, normalized by the truth model's `C[0]` and
averaged over the four conditions. “Fitted blocks” are estimated from the even
seed residuals. The scalar reference uses the manifest's known noise parameters
(with the rate-unit scale adjustment where applicable), so it is a privileged
synthetic baseline rather than a deployable estimate.

| Candidate residuals | Fitted blocks | Known scalar reference |
|---|---:|---:|
| Truth | 4.32% | 3.22% |
| Attitude direction flip | 3.92% | 3.07% |
| Quaternion storage-order flip | 3.79% | 3.11% |
| Rate-frame flip | 4.01% | 2.98% |
| Rate-order reversal | 4.23% | 3.31% |
| Rate-sign flip | 4.25% | 3.34% |
| Rate-unit flip, adjusted | 4.20% | 3.25% |

The fitted empirical blocks did not improve held-out error in any of the 28
candidate/condition cells. In this low-noise synthetic cohort, estimating full
blocks added sampling variance without beating the known scalar reference.
This does not rule out a better candidate-specific analytic model or a
data-driven fit under different noise levels and motion; it does mean that
adding empirical covariance fitting to package inference is not supported by
this experiment. Broader transfer tests and real labeled logs remain open.

#### Multi-regime follow-up

The check was also run on all four motion profiles, all three noise levels, and
the four `(rho_orientation, rho_rate)` combinations above, with 30 seeds per
cell: 1,440 trajectories across 48 condition cells. Regenerate that stress set
and its residual-covariance report with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-ar1-full-residual-30seed \
  --noise-profile all --motion-profile all --noise-replicates 30 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-ar1-full-residual-30seed/manifest.json \
  --output .validation-corpus-ar1-full-residual-30seed/candidate-aware-residuals-30seed-rerun.json \
  --max-cases-per-condition 30
```

Across the 48 truth cells, the largest absolute lag-covariance error was 2.13%
of `C[0]`; centered whitened sum of squares ranged from 1.0065 to 1.0136 per
degree of freedom, and zero-lag cross-axis covariance error ranged from 1.35%
to 3.64%. For the direction-only counterfactual, the corresponding maximum
lag error was 2.28%, centered whitened energy ranged from 1.0077 to 1.0174, and
cross-axis error ranged from 1.86% to 3.23%. Its uncentered whitened energy
ranged from 1.084 per degree of freedom in low-excitation/high-noise cases to
1,323 in highly informative low-noise cases. These 30-seed cells include
low-excitation, single-axis, and near-gimbal motion and moderate/high noise, but
the estimates still have finite Monte Carlo uncertainty and are not a
likelihood-ratio calibration study.

The original direction-only pairwise comparison on those 1,440 trajectories
ranked truth first in 1,432 cases under the truth-model i.i.d. score and 1,436
under the truth-model colored score. All eight i.i.d. misses and all four
colored misses were in low-excitation, high-noise cells; at
`(rho_orientation, rho_rate) = (0.8, 0.8)`, both methods ranked truth first in
27 / 30 trials.

#### Development-set counterfactual results

The corrected validator compares six alternatives, changing exactly one
convention field at a time. These development results use the 1,440 trajectories
from the original seed block `20261030`–`20261059` (four motion profiles, three
noise levels, four AR(1) combinations, 30 seeds per cell). “Truth wins” means
its Gaussian residual score was higher under the stated covariance model.
These are not posterior odds or estimates of package recommendation accuracy.

| One-field alternative | Truth wins, i.i.d. / 1,440 | Truth wins, colored / 1,440 |
|---|---:|---:|
| Attitude direction | 1,432 | 1,436 |
| Quaternion storage order | 1,426 | 1,433 |
| Rate frame | 1,059 | 1,051 |
| Rate order reversal | 1,384 | 1,409 |
| Rate sign | 1,432 | 1,436 |
| Rate unit | 1,260 | 1,440 |

All rate-frame alternative wins occurred in low-excitation or single-axis
conditions, where the two scores are often close to zero. The i.i.d. rate-unit
comparison still favored the alternative in 180 cases, all in low-excitation
moderate/high-noise cells; the candidate-adjusted colored score ranked truth
first in all 1,440. The colored reference reduced misses for several
alternatives but slightly increased them for rate-frame flips, so it is not a
uniformly better decision rule. The 30-seed cells do not calibrate confidence
or field error rates.

Except for rate unit, candidate covariance uses the truth model. For the
rate-unit alternative, the rate-noise standard deviation is scaled by
`f(unit_candidate) / f(unit_truth)`, where `f(radians/s) = 1` and
`f(degrees/s) = pi/180`. The residual covariance table compares each candidate
to that specified model:

| Candidate | Maximum absolute lag error / `C[0]` | Centered whitened energy / dof | Maximum cross-axis covariance error |
|---|---:|---:|---:|
| Truth | 2.12% | 1.0065–1.0136 | 3.63% |
| Attitude direction flip | 2.28% | 1.0077–1.0174 | 3.23% |
| Quaternion storage-order flip | 2.10% | 1.0048–1.0131 | 3.07% |
| Rate-frame flip | 2.01% | 1.0064–1.0099 | 2.88% |
| Rate-order reversal | 2.26% | 1.0103–1.0178 | 3.89% |
| Rate-sign flip | 2.28% | 1.0046–1.0170 | 4.05% |
| Rate-unit flip, adjusted | 2.13% | 1.0118–1.0148 | 3.88% |

The earlier shared-covariance rate-unit diagnostic had whitened energy below
one; the unit adjustment brings its second-order residual statistics in line
with the model on this development set. This is a targeted covariance model for
the rate-unit transformation, not a complete candidate-specific model for all
six alternatives.

#### Independent-seed follow-up

Before applying the unit adjustment to a new seed block, the generator was
extended with `--noise-seed-start`. The independent synthetic cohort used seeds
`20301030`–`20301059`, did not overlap the development block, and repeated the
same 48 cells (1,440 trajectories). It uses the same analytic generator and
noise assumptions, so this is independent Monte Carlo validation—not
independent real-world evidence.

| Alternative | Development truth wins, i.i.d. | Held-out truth wins, i.i.d. | Development truth wins, colored | Held-out truth wins, colored |
|---|---:|---:|---:|---:|
| Attitude direction | 1,432 / 1,440 | 1,431 / 1,440 | 1,436 / 1,440 | 1,439 / 1,440 |
| Quaternion storage order | 1,426 / 1,440 | 1,419 / 1,440 | 1,433 / 1,440 | 1,430 / 1,440 |
| Rate frame | 1,059 / 1,440 | 1,058 / 1,440 | 1,051 / 1,440 | 1,072 / 1,440 |
| Rate order reversal | 1,384 / 1,440 | 1,354 / 1,440 | 1,409 / 1,440 | 1,398 / 1,440 |
| Rate sign | 1,432 / 1,440 | 1,431 / 1,440 | 1,436 / 1,440 | 1,439 / 1,440 |
| Rate unit, adjusted | 1,260 / 1,440 | 1,253 / 1,440 | 1,440 / 1,440 | 1,440 / 1,440 |

The truth and rate-unit candidate covariance checks were similar between
development and held-out seeds:

| Candidate model | Development max lag error / `C[0]` | Held-out max lag error / `C[0]` | Development centered energy / dof | Held-out centered energy / dof |
|---|---:|---:|---:|---:|
| Truth | 2.12% | 1.89% | 1.0065–1.0136 | 1.0083–1.0168 |
| Rate-unit adjusted | 2.13% | 1.80% | 1.0118–1.0148 | 1.0098–1.0120 |

Rate-frame flips remained difficult in low-excitation and single-axis motion:
the alternative won 381/389 development trials and 382/368 held-out trials
under the i.i.d./colored comparisons. This is a motion observability limitation
in these synthetic cases, not evidence that one score is universally superior.
The adjusted colored rate-unit score ranked truth in every held-out case, but
its log-likelihood differences can be extremely large because of the covariance
normalizer and residual mean. Do not interpret their magnitude as calibrated
confidence or use them to tune production thresholds.

#### Bidirectional rate-unit check

A separate paired cohort used seeds `20401030`–`20401059`, low noise, rich
three-axis motion, and 161 samples at 25 ms. For each seed the generator wrote
the same physical trajectory and noise realization once in rad/s and once in
deg/s. The rate-unit counterfactual ranked truth first in all 30 cases in each
direction under both the i.i.d. and colored reference scores. Full package inference
also put truth first in all 60 noisy cases, but all 60 remained ambiguous and the
decision policy abstained; this supports correct ranking, not decisive or calibrated
recommendations. The paired unit files are not 60 independent draws.

Reproduce this check with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-rate-unit-paired-30seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-seed-start 20401030 --noise-sample-counts 161 \
  --truth-rate-units radians_per_second degrees_per_second
python -m scripts.validate_corpus \
  .validation-corpus-rate-unit-paired-30seed/manifest.json \
  --output .validation-corpus-rate-unit-paired-30seed/assessment.json
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-rate-unit-paired-30seed/manifest.json \
  --output .validation-corpus-rate-unit-paired-30seed/residual-model.json \
  --max-cases-per-condition 30
```

Reproduce the independent seed block and scoring run with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-ar1-candidate-heldout-30seed \
  --noise-profile all --motion-profile all --noise-replicates 30 \
  --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8 \
  --noise-seed-start 20301030
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-ar1-candidate-heldout-30seed/manifest.json \
  --output \
  .validation-corpus-ar1-candidate-heldout-30seed/candidate-aware-residuals-triangular.json \
  --max-cases-per-condition 30
```

The generator and report writer refuse to overwrite existing outputs. For a
repeat run, choose a new `--outdir`, a non-overlapping `--noise-seed-start`, and
a new report path.

For a split within a manifest, `--seed-partition even` and `--seed-partition odd`
select disjoint seed parities; the JSON report records the partition, case count,
and selected seed values. The independent-seed run above reserved its whole new
seed block and used `--seed-partition all`.

#### Broad 60-seed motion/noise follow-up

A larger cohort used the non-overlapping seeds `20501030`–`20501089`, four
motion profiles, three noise levels, four AR(1) coefficient pairs, and 161
samples at 25 ms. This makes 48 condition cells with 60 cases each. The noisy
cases all use the same labeled quaternion/rate convention; this sweep varies
motion and noise, not the truth convention. It has no timestamp jitter,
dropouts, or channel latency.

Reproduce the corpus and both assessments with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-ar1-full-covariance-holdout-60seed \
  --noise-profile all --motion-profile all --noise-replicates 60 \
  --noise-seed-start 20501030 --noise-sample-counts 161 \
  --orientation-noise-ar1-coefficients 0 0.8 \
  --rate-noise-ar1-coefficients 0 0.8
python -m scripts.validate_corpus \
  .validation-corpus-ar1-full-covariance-holdout-60seed/manifest.json \
  --output .validation-corpus-ar1-full-covariance-holdout-60seed/assessment.json
python -m scripts.validate_correlated_residual_model \
  .validation-corpus-ar1-full-covariance-holdout-60seed/manifest.json \
  --output .validation-corpus-ar1-full-covariance-holdout-60seed/covariance-holdout.json \
  --max-cases-per-condition 60
```

The package assessment covered 2,880 noisy cases (the report also includes
three noiseless demonstration cases). Truth remained in the evidence contenders
for all 2,880; 92 cases produced a correct recommendation, and the remaining
2,788 abstained. All 92 recommendations came from low-noise rich-three-axis
motion. There were no wrong recommendations, but the recommendation coverage
was only 3.19%. The data-winning class differed from truth in 1,496 cases; those
cases remained ambiguous rather than becoming recommendations. This supports
the current conservative abstention behavior on this synthetic sweep, not a
claim of calibrated real-world accuracy.

The covariance report fit candidate 3x3 lag blocks on 30 even seeds and
evaluated on 30 disjoint odd seeds in each condition. Mean held-out RMS block
error is normalized by truth `C[0]` and averaged over the six diagnostic lags.
The scalar reference uses the manifest's known noise parameters, so it is a
privileged synthetic baseline:

| Noise level | Candidate/condition comparisons | Fitted blocks | Known scalar reference | Fitted blocks better |
|---|---:|---:|---:|---:|
| Low | 112 | 6.75% | 5.24% | 0 / 112 |
| Moderate | 112 | 6.74% | 5.23% | 0 / 112 |
| High | 112 | 6.74% | 5.23% | 0 / 112 |
| All | 336 | 6.74% | 5.23% | 0 / 336 |

Empirical block fitting did not beat the known scalar reference in any
candidate/condition comparison across either this broad sweep or the earlier
200-seed low-noise cohort. That argues against adding this empirical fit to
package inference under the tested assumptions. The result does not rule out a
better analytic candidate covariance or other sampling/noise regimes.

#### Full 192-convention encoding sweep

The generator's `--truth-convention-sweep` option varies all quaternion/rate
fields: attitude direction, quaternion storage order, rate frame, all six rate
orders, rate sign, and rate unit. It encodes these directly from the analytic
motion rather than calling the inference implementation. A zero-noise
round-trip test independently decodes all 192 encodings with SciPy.

Reproduce the four-seed low-noise inference run with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-truth-convention-sweep-low-4seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 4 \
  --noise-seed-start 20601030 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-truth-convention-sweep-low-4seed/manifest.json \
  --output .validation-corpus-truth-convention-sweep-low-4seed/assessment.json
```

The noisy cohort has 768 trajectories (192 raw conventions x four seeds),
161 samples at 25 ms, rich-three-axis motion, and the low-noise profile. Both
rate signs and units are included in the configured search. The four seeds are
paired across convention encodings, so these are not 768 independent Monte
Carlo draws. The assessment report is schema version 3; its condition
summaries include `truth_class_id` and `motion_seed` so different truth
equivalence classes and sampled-rate trajectories are not pooled together.

The raw data-winning class matched the labeled truth class in all 768 cases,
and truth remained an evidence contender in every case. Exactly two classes
remained within the evidence margin in each case; the alternative differed
only in rate frame. The policy abstained in all 768 rather than making an
unsupported frame recommendation. The 192 raw encodings form 96 structural
equivalence classes, with eight paired cases per class. This is evidence that
the generator covers the convention transformations and that ranking is
consistent on this cohort; it is not evidence that rate frame is identifiable
from this motion or that the policy is calibrated.

The same one-seed convention sweep was also run across all four motion
profiles. Reproduce it with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-truth-convention-sweep-all-motion-1seed \
  --noise-profile low --motion-profile all --noise-replicates 1 \
  --noise-seed-start 20602030 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-truth-convention-sweep-all-motion-1seed/manifest.json \
  --output .validation-corpus-truth-convention-sweep-all-motion-1seed/assessment.json
```

All 768 cases remained ambiguous and were abstained on; evidence misses were
zero. The raw data-winner and within-margin class counts varied sharply with
motion:

| Motion profile | Cases | Data-winner misses | Within-margin classes per case | Recommendations |
|---|---:|---:|---:|---:|
| Rich three-axis | 192 | 0 | 2 | 0 |
| Low excitation | 192 | 96 | 48 or 96 | 0 |
| Near gimbal | 192 | 96 | 8 | 0 |
| Single axis | 192 | 192 | 8 | 0 |

In this single paired seed, low-excitation and near-gimbal data-winner misses
were split evenly across body- and world-frame rate labels (48 misses per
frame); single-axis data missed for both frames. Rich three-axis data had no
misses. The frame split corrects an earlier misreading of the labels: these
misses are not confined to world-frame truth.

#### Four-seed follow-up

Three additional independent noise seeds were run across all four motion
profiles. Reproduce this block with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-truth-convention-sweep-all-motion-seeds-20602031-33 \
  --noise-profile low --motion-profile all --noise-replicates 3 \
  --noise-seed-start 20602031 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-truth-convention-sweep-all-motion-seeds-20602031-33/manifest.json \
  --output .validation-corpus-truth-convention-sweep-all-motion-seeds-20602031-33/assessment.json
```

That assessment contains 2,304 noisy trajectories plus three analytic demo
cases. The table below combines only noisy cases from this run and the prior
seed 20602030: four paired seeds, 192 convention encodings per profile/seed,
and 3,072 noisy cases total.

| Motion profile | Noisy cases | Data/operational misses | Evidence-contender class counts | Noisy recommendations |
|---|---:|---:|---:|---:|
| Rich three-axis | 768 | 0 | 2 in 768 cases | 0 |
| Low excitation | 768 | 288 | 48 in 384; 96 in 384 | 0 |
| Near gimbal | 768 | 216 | 8 in 768 cases | 0 |
| Single axis | 768 | 768 | 8 in 768 cases | 0 |

Truth remained an evidence contender in all 3,072 noisy cases, and the policy
abstained on all of them. Data-winner and operational misses totaled 1,272
(41.4%); the equal totals here do not mean those classes were identified as
wrong with high confidence. Misses were highly seed-sensitive in the two
intermediate-excitation profiles:

| Motion profile | Seed 20602030 | Seed 20602031 | Seed 20602032 | Seed 20602033 |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/192 | 0/192 | 0/192 | 0/192 |
| Low excitation | 96/192 | 96/192 | 96/192 | 0/192 |
| Near gimbal | 96/192 | 0/192 | 0/192 | 120/192 |
| Single axis | 192/192 | 192/192 | 192/192 | 192/192 |

Across the four seeds, low-excitation misses were evenly split by body/world
rate frame (144 each); near-gimbal misses affected both frames (120 body, 96
world). These synthetic results show stable ambiguity for rich motion,
unresolvable alternatives for single-axis motion, and substantial realization
sensitivity between them. Four paired seeds are not an estimate of production
error rates or evidence that the policy is calibrated.

#### One-seed convention sweep across noise levels

The full 192-convention sweep was also run for low, moderate, and high noise
across all four motion profiles. Reproduce this paired-seed cohort with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-truth-convention-sweep-all-noise-all-motion-1seed-20701030 \
  --noise-profile all --motion-profile all --noise-replicates 1 \
  --noise-seed-start 20701030 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-truth-convention-sweep-all-noise-all-motion-1seed-20701030/manifest.json \
  --output \
  .validation-corpus-truth-convention-sweep-all-noise-all-motion-1seed-20701030/assessment.json \
  --progress
```

The assessment contains 2,304 noisy cases plus three analytic demo cases.
Each motion/noise condition has 192 convention encodings from the same seed.
Data-winner misses out of 192 noisy cases were:

| Motion profile | Low noise | Moderate noise | High noise |
|---|---:|---:|---:|
| Rich three-axis | 0 | 0 | 0 |
| Low excitation | 0 | 0 | 96 |
| Near gimbal | 0 | 24 | 96 |
| Single axis | 96 | 96 | 96 |

Truth remained in the evidence contender set for all 2,304 cases; all were
ambiguous and abstained, with no noisy recommendations. The 504 raw/operational
winner misses were not evidence misses or confident recommendations. Evidence
was marked insufficient in 1,728 cases: every low-excitation, near-gimbal, and
single-axis case. Rich three-axis cases were evidence-sufficient but remained
ambiguous. This is one paired seed, not a miss-rate estimate; the four-seed
low-noise run above shows that data-winner outcomes can vary substantially by
seed even with the same motion and convention matrix.

#### Four-seed high-noise convention sweep

Three more high-noise seeds were evaluated across all motion profiles. The
following command reproduces the added seeds; the combined table also includes
high-noise seed 20701030 from the preceding all-noise cohort:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-truth-convention-high-noise-all-motion-seeds-20701031-33 \
  --noise-profile high --motion-profile all --noise-replicates 3 \
  --noise-seed-start 20701031 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-truth-convention-high-noise-all-motion-seeds-20701031-33/manifest.json \
  --output \
  .validation-corpus-truth-convention-high-noise-all-motion-seeds-20701031-33/assessment.json \
  --progress
```

Combining four paired seeds gives 768 noisy cases per motion profile:

| Motion profile | Data/operational misses | Evidence-insufficient | Evidence misses | Noisy recommendations |
|---|---:|---:|---:|---:|
| Rich three-axis | 0 | 192 | 0 | 0 |
| Low excitation | 576 | 768 | 0 | 0 |
| Near gimbal | 480 | 768 | 0 | 0 |
| Single axis | 576 | 768 | 0 | 0 |

Truth remained in the evidence contender set in all 3,072 cases, and the
policy abstained on all of them. The 1,632 raw/operational winner misses
(53.1%) were not evidence misses or incorrect recommendations. Of the 576
evidence-sufficient cases, all remained ambiguous; evidence sufficiency did
not resolve the convention. Miss counts by seed (out of 192 per profile) were:

| Motion profile | Seed 20701030 | Seed 20701031 | Seed 20701032 | Seed 20701033 |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/192 | 0/192 | 0/192 | 0/192 |
| Low excitation | 96/192 | 192/192 | 96/192 | 192/192 |
| Near gimbal | 96/192 | 192/192 | 0/192 | 192/192 |
| Single axis | 96/192 | 192/192 | 96/192 | 192/192 |

Across all four seeds, misses were evenly split between body- and world-frame
rate labels within each affected profile. These are four paired synthetic
seeds, not 3,072 independent Monte Carlo draws or a production error-rate
estimate.

These synthetic results do not calibrate likelihood magnitudes, package
thresholds, or real-world sensor behavior. The covariance cohorts still use
one truth convention and synchronized fixed-rate data. Acquisition-artifact
sweeps and independently labeled real recordings remain open validation work.

#### Four-seed moderate-noise convention sweep

The same four seeds were evaluated at moderate noise for all motion profiles.
The three additional seeds can be regenerated and validated with:

```bash
corpus_dir=.validation-corpus-truth-convention-moderate-noise-all-motion-seeds-20701031-33
python -m scripts.generate_validation_corpus \
  --outdir "$corpus_dir" \
  --noise-profile moderate --motion-profile all --noise-replicates 3 \
  --noise-seed-start 20701031 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  "$corpus_dir/manifest.json" \
  --checkpoint "$corpus_dir/assessment.checkpoint.json" \
  --checkpoint-every 100 --progress \
  --output "$corpus_dir/assessment.json"
```

Across the four paired seeds, all 3,072 cases retained truth in the evidence
contender set, received no recommendation, and were abstained on. Data-winner
misses and evidence insufficiency were:

| Motion profile | Data/operational misses | Evidence-insufficient | Evidence misses | Noisy recommendations |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/768 | 0/768 | 0 | 0 |
| Low excitation | 192/768 | 768/768 | 0 | 0 |
| Near gimbal | 264/768 | 768/768 | 0 | 0 |
| Single axis | 576/768 | 768/768 | 0 | 0 |

Misses by seed (out of 192 convention encodings per motion profile) were:

| Motion profile | Seed 20701030 | Seed 20701031 | Seed 20701032 | Seed 20701033 |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/192 | 0/192 | 0/192 | 0/192 |
| Low excitation | 0/192 | 96/192 | 0/192 | 96/192 |
| Near gimbal | 24/192 | 192/192 | 0/192 | 48/192 |
| Single axis | 96/192 | 192/192 | 96/192 | 192/192 |

As with the low- and high-noise cohorts, these are paired synthetic seeds,
not independent samples from a deployment population.

#### Four-seed low-noise convention sweep

The low-noise cohort uses the same four paired seeds and motion profiles. The
three additional seeds can be regenerated and validated with:

```bash
corpus_dir=.validation-corpus-truth-convention-low-noise-all-motion-seeds-20701031-33
python -m scripts.generate_validation_corpus \
  --outdir "$corpus_dir" \
  --noise-profile low --motion-profile all --noise-replicates 3 \
  --noise-seed-start 20701031 --noise-sample-counts 161 \
  --truth-convention-sweep
python -m scripts.validate_corpus \
  "$corpus_dir/manifest.json" \
  --checkpoint "$corpus_dir/assessment.checkpoint.json" \
  --checkpoint-every 100 --progress \
  --output "$corpus_dir/assessment.json"
```

Across four seeds, all 3,072 noisy cases retained truth in the evidence
contender set, received no recommendation, and were abstained on. Data-winner
misses and evidence insufficiency were:

| Motion profile | Data/operational misses | Evidence-insufficient | Evidence misses | Noisy recommendations |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/768 | 0/768 | 0 | 0 |
| Low excitation | 288/768 | 768/768 | 0 | 0 |
| Near gimbal | 120/768 | 768/768 | 0 | 0 |
| Single axis | 576/768 | 768/768 | 0 | 0 |

Misses by seed (out of 192 convention encodings per motion profile) were:

| Motion profile | Seed 20701030 | Seed 20701031 | Seed 20701032 | Seed 20701033 |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/192 | 0/192 | 0/192 | 0/192 |
| Low excitation | 0/192 | 96/192 | 96/192 | 96/192 |
| Near gimbal | 0/192 | 96/192 | 0/192 | 24/192 |
| Single axis | 96/192 | 192/192 | 96/192 | 192/192 |

#### Paired comparison across noise levels

Combining the same four seeds at low, moderate, and high noise gives 9,216
noisy cases (768 per motion/noise cell). Every case remained ambiguous and was
abstained on; truth was never excluded from the evidence contenders, and there
were no noisy recommendations. The data-winner misses were:

| Motion profile | Low noise | Moderate noise | High noise | Total |
|---|---:|---:|---:|---:|
| Rich three-axis | 0/768 | 0/768 | 0/768 | 0/2,304 |
| Low excitation | 288/768 | 192/768 | 576/768 | 1,056/2,304 |
| Near gimbal | 120/768 | 264/768 | 480/768 | 864/2,304 |
| Single axis | 576/768 | 576/768 | 576/768 | 1,728/2,304 |
| All profiles | 984/3,072 | 1,032/3,072 | 1,632/3,072 | 3,648/9,216 |

Evidence was marked insufficient in 2,304 low-noise, 2,304 moderate-noise,
and 2,496 high-noise cases; there were zero evidence misses. These raw winner
misses are not incorrect recommendations: the policy abstained on every noisy
case. The outcomes remain paired synthetic measurements, not independent draws
or production error-rate estimates.

#### Four-seed convention sweep with whole-row dropout

To isolate missing-row effects, the low-noise all-convention sweep was repeated
with a 35% dropout probability, using the same four noise seeds as its clean
baseline:

```bash
corpus_dir=.validation-corpus-truth-convention-low-noise-dropout-all-motion-seeds-20701030-33
python -m scripts.generate_validation_corpus \
  --outdir "$corpus_dir" \
  --noise-profile low --motion-profile all --noise-replicates 4 \
  --noise-seed-start 20701030 --noise-sample-counts 161 \
  --dropout-probabilities 0.35 --truth-convention-sweep
python -m scripts.validate_corpus \
  "$corpus_dir/manifest.json" \
  --checkpoint "$corpus_dir/assessment.checkpoint.json" \
  --checkpoint-every 100 --progress \
  --output "$corpus_dir/assessment.json"
```

The 3,072 noisy cases pair one-to-one with the no-dropout low-noise baseline by
exact convention, motion profile, and seed. Separate seeded random streams hold
the measurement noise and timestamps fixed while the row mask changes; a CSV
check confirmed exact timestamp and channel equality for the retained rows in a
representative pair. Dropout retained 101–111 of 161 rows (105 on average).

| Motion profile | Baseline misses | 35% dropout misses | Newly missed | Recovered | Contender sets changed | Recommendations (correct) |
|---|---:|---:|---:|---:|---:|---:|
| Rich three-axis | 0/768 | 0/768 | 0 | 0 | 576 | 576/576 |
| Low excitation | 288/768 | 384/768 | 192 | 96 | 0 | 0/0 |
| Near gimbal | 120/768 | 24/768 | 0 | 96 | 0 | 0/0 |
| Single axis | 576/768 | 768/768 | 192 | 0 | 0 | 0/0 |
| All profiles | 984/3,072 | 1,176/3,072 | 384 | 192 | 576 | 576/576 |

Truth stayed in the evidence contenders for every case, with no evidence misses
or incorrect recommendations. The 576 recommendations all occurred in rich
three-axis motion; the other motion profiles abstained on every case. However,
all noisy cases retained the heuristic `indeterminate` confidence label,
including those recommendations. Decision status and that uncalibrated label
are separate outputs and should not be conflated. The raw data-winner miss total
rose by 192 overall, despite 192 recovered cases, and the apparent recommendation
success is specific to this synthetic generator and seed block—not evidence that
dropout helps real logs or that the policy is calibrated.

The 30-seed acquisition and noise sweeps can be regenerated with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-short-duration-30seed \
  --noise-profile low --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 \
  --timestamp-jitter-amplitudes-s 0 0.005 \
  --dropout-probabilities 0 0.35 \
  --orientation-latencies-s 0 0.05 --rate-latencies-s 0 0.05
python scripts/validate_corpus.py \
  .validation-corpus-short-duration-30seed/manifest.json \
  --output .validation-corpus-short-duration-30seed/assessment.json
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-short-duration-noise-30seed \
  --noise-profile all --motion-profile rich-three-axis --noise-replicates 30 \
  --noise-sample-counts 161 --timestamp-jitter-amplitudes-s 0 0.005 \
  --dropout-probabilities 0 0.35
python scripts/validate_corpus.py \
  .validation-corpus-short-duration-noise-30seed/manifest.json \
  --output .validation-corpus-short-duration-noise-30seed/assessment.json
```

```bash
python scripts/generate_validation_corpus.py
python scripts/validate_corpus.py .validation-corpus/manifest.json \
  --output .validation-corpus/assessment.json
```

The default `.validation-corpus/` and `.validation-corpus-*` directories are ignored
by Git. Keep generated outputs under that prefix, choose a new name with `--outdir`
for each run, and the generator refuses to overwrite an existing directory.

Each manifest case records the CSV path, a source for its label, the complete known
convention, and the search configuration. Paths are relative to the manifest. The
validator checks that the labeled hypothesis is actually in the configured search;
otherwise it rejects the case instead of counting it as a success or failure.

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "demo-only-synthetic",
      "csv": "../examples/synthetic_quaternion.csv",
      "label_source": "scripts/generate_examples.py (demo only; not independent validation)",
      "representation": "quaternion",
      "orientation_columns": ["q0", "q1", "q2", "q3"],
      "rate_columns": ["w0", "w1", "w2"],
      "time_column": "time",
      "truth": {
        "attitude_direction": "world_to_body",
        "quaternion_order": "scalar_first",
        "rate_frame": "world",
        "rate_order": "rqp",
        "rate_unit": "radians_per_second"
      },
      "config": {}
    }
  ]
}
```

Run the evaluator after installing development dependencies:

```bash
python scripts/validate_corpus.py path/to/manifest.json --output assessment.json
python scripts/validate_corpus.py path/to/manifest.json --fail-on-miss
```

The result distinguishes a unique correct evidence class, ambiguity that contains
the known class, and an evidence miss (truth outside the contender classes). It
separately reports when the data-winning or operational class differs from truth.
Condition summaries expose each miss rate, evidence insufficiency, and false
evidence sufficiency by group. `--fail-on-miss` exits nonzero for either an evidence
miss or an operational miss. If a config fixes a convention dimension, the case
does **not** measure recovery of that dimension; record what was known before
running inference.

Long evaluations can be checkpointed and resumed without re-running completed
cases:

```bash
python scripts/validate_corpus.py path/to/manifest.json \
  --checkpoint path/to/assessment.checkpoint.json --checkpoint-every 100 \
  --output path/to/assessment.json
python scripts/validate_corpus.py path/to/manifest.json \
  --checkpoint path/to/assessment.checkpoint.json --resume \
  --output path/to/assessment.json
```

For faster case-level evaluation, add `--workers N`. This uses separate processes
and defaults to one worker; output ordering and checkpoint semantics do not depend
on the worker count. Parallel workers each load NumPy/SciPy, so choose a count that
fits the available memory.

Checkpoints are atomically replaced and tied to the exact manifest, source CSV
contents, inference package and validator source, and runtime environment. A
changed input, implementation, or environment is rejected; after interruption,
at most `--checkpoint-every - 1` newly evaluated cases need to be repeated. Keep
checkpoints alongside generated, Git-ignored corpus data.

Start with logs from at least two independent sources and include quiet motion,
three-axis motion, timestamp jitter, missing samples, and sensor latency when those
conditions occur naturally. Keep acquisition details, label evidence, permissions,
and source versions with the corpus. Split by source or recording session when
tuning thresholds; adjacent samples from one recording are not independent
validation cases. The packaged `common-engineering-v1` profile and confidence labels
remain uncalibrated until held-out measurements support a stronger claim.

## Sampled-rate nonanalytic pilot

The numerically integrated sampled-rate generator was assessed on four noise
seeds at 161 samples (4 seconds, 25 ms nominal spacing), with paired 0/2 ms
timestamp jitter and 0/10% whole-row dropout. All 16 sampled-rate cases produced
correct recommendations; there were no evidence misses, operational misses,
abstentions, or incorrect evidence-sufficient cases. Four of four dropout cases
per jitter condition retained between 139 and 143 rows (mean 141). Fourteen
cases received low confidence and two were indeterminate. Three analytic control
cases were included separately; do not pool them with the 16 sampled-rate cases.

Reproduce the generated cohort and assessment with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-nonanalytic-sampled-rate-four-seed-20261030 \
  --motion-profile sampled-rate-nonanalytic --motion-seed 730201 \
  --noise-profile low --noise-replicates 4 --noise-sample-counts 161 \
  --timestamp-jitter-amplitudes-s 0 0.002 \
  --dropout-probabilities 0 0.1
python scripts/validate_corpus.py \
  .validation-corpus-nonanalytic-sampled-rate-four-seed-20261030/manifest.json \
  --output .validation-corpus-nonanalytic-sampled-rate-four-seed-20261030/assessment.json
```

This was Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, package `0.0.1a0`. It tests
one known convention on one generated rate process and one contiguous four-seed
block. Perfect performance on this small, constructed cohort is a regression
result—not an estimate of production accuracy or confidence calibration. Channel
latency is excluded from this profile until its generator uses one shared
absolute-time trajectory. A limited sampled-rate convention sweep is now recorded
below; broader motion, noise, and acquisition-condition sweeps remain future work.

## Sampled-rate full convention sweep

Three seeded random-Fourier rate processes were each encoded across the complete
192-value quaternion/rate truth-convention matrix. Every case used 161 samples at
25 ms spacing, the low-noise profile, and the same noise seed. This creates 576
paired convention encodings, not 576 independent physical trajectories. The three
analytic demo controls in the manifest are excluded from the results below.

| Motion seed | Convention encodings | Correct recommendations | Ambiguous with truth retained |
|---:|---:|---:|---:|
| 730202 | 192 | 192 | 0 |
| 730203 | 192 | 0 | 192 |
| 730204 | 192 | 192 | 0 |
| Total | 576 | 384 | 192 |

The data winner and operational class matched truth in all 576 cases. There were
no evidence misses, operational misses, or incorrect recommendations. The 192
cases for seed 730203 abstained because multiple evidence contenders remained;
the other 384 produced unique, correct recommendations. The heuristic confidence
label was `indeterminate` in all 576 cases. That label is not calibrated confidence,
and the seed-to-seed coverage variation is a reminder that this constructed cohort
does not establish deployment accuracy or calibration.

Reproduce the manifest and assessment with:

```bash
python scripts/generate_validation_corpus.py \
  --outdir .validation-corpus-sampled-rate-convention-3seed \
  --motion-profile sampled-rate-nonanalytic \
  --motion-seeds 730202 730203 730204 \
  --noise-profile low --noise-replicates 1 \
  --noise-seed-start 20261030 --noise-sample-counts 161 \
  --truth-convention-sweep
python scripts/validate_corpus.py \
  .validation-corpus-sampled-rate-convention-3seed/manifest.json \
  --checkpoint .validation-corpus-sampled-rate-convention-3seed/assessment.checkpoint.json \
  --checkpoint-every 25 --progress \
  --output .validation-corpus-sampled-rate-convention-3seed/assessment.json
```

This is synthetic robustness evidence from three rate processes generated by one
model, not independent-source validation. It does not measure field miss rates,
sensor diversity, or confidence calibration. Independently labeled logs from
multiple real acquisition systems remain an evidence gate.

## Euler representation convention sweep

The native Euler generator covers 12 supported sequences (Tait--Bryan and proper
Euler), both intrinsic/extrinsic modes, all four supported angle units, and both
attitude directions. Each of three seeded random-Fourier Euler paths yields 192
convention cases, for 576 paired cases total. The generator constructs elementary
axis-rotation matrices directly and obtains body rates from analytic matrix
derivatives; it imports neither the inference package nor SciPy rotation routines.
The rate label is intentionally fixed to body-frame `pqr`, radians/second, positive
sign. Each trajectory has 161 samples at 25 ms spacing, with shared seeded channel
noise across its convention encodings.

The data winner and operational class matched truth in 573/576 cases. The other
three cases still retained truth in the evidence contenders, so none produced an
incorrect recommendation. Overall, 366 cases received correct recommendations;
210 abstained with truth among the contenders, including 51 marked
evidence-insufficient. All 576 heuristic confidence labels were indeterminate.

| Motion seed | Correct recommendations | Ambiguous with truth retained | Data-winner misses | Evidence insufficient |
|---:|---:|---:|---:|---:|
| 730301 | 96 | 96 | 3 | 45 |
| 730302 | 96 | 96 | 0 | 6 |
| 730303 | 174 | 18 | 0 | 0 |
| Total | 366 | 210 | 3 | 51 |

| Angle unit | Cases | Correct recommendations | Ambiguous with truth retained | Evidence insufficient |
|---|---:|---:|---:|---:|
| Radians | 144 | 144 | 0 | 0 |
| Degrees | 144 | 36 | 108 | 51 |
| Turns | 144 | 144 | 0 | 0 |
| Grads | 144 | 42 | 102 | 0 |

The degrees/grads ambiguity and seed-dependent coverage are specific to these
motions, noise level, fixed rate convention, and inference profile; they are not
general claims about unit identifiability. Tests compare the direct axis products
against SciPy's rotations and check analytic rates against finite-difference
relative rotations. SciPy is used only as a test oracle, not by the corpus generator.

Regenerate and validate the cohort with:

```bash
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-native-rich-3seed
python scripts/validate_corpus.py \
  .validation-corpus-euler-native-rich-3seed/manifest.json \
  --checkpoint .validation-corpus-euler-native-rich-3seed/assessment.checkpoint.json \
  --checkpoint-every 25 --progress \
  --output .validation-corpus-euler-native-rich-3seed/assessment.json
```

The assessment ran on Windows/Python 3.11.9 with NumPy 2.4.6 and SciPy 1.17.1.
These are generated regression results from one independently implemented motion
model, not held-out accuracy or confidence calibration; independently labeled
recordings from multiple systems remain required.

### Fixed-duration cadence sensitivity

The same native Euler sweep was repeated at 10, 40, and 100 Hz while holding
physical duration at four seconds and using the same three motion seeds, noise
seed, inference implementation, and runtime. The sample counts were 41, 161,
and 401. Each cadence covered all 192 convention encodings per seed (576 cases).
The per-sample noise standard deviations were held at 0.0015 rad for Euler
coordinates and 0.012 rad/s for rates; channels were synchronous, with no jitter,
dropout, or latency.

| Cadence | Samples | Correct recommendations | Truth-retaining abstentions | Data-winner / operational misses | Evidence-insufficient | Evidence misses / wrong recommendations | Confidence labels |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 Hz | 41 | 507 (88.0%) | 69 | 0 / 0 | 1 | 0 / 0 | 145 low, 431 indeterminate |
| 40 Hz | 161 | 366 (63.5%) | 210 | 3 / 3 | 51 | 0 / 0 | 576 indeterminate |
| 100 Hz | 401 | 132 (22.9%) | 444 | 54 / 54 | 132 | 0 / 0 | 576 indeterminate |

Every recommendation was correct and truth stayed in the evidence contenders in
all three cohorts. At 100 Hz, all 54 data-winner misses were degree-unit cases;
they remained in the evidence set and were abstained on rather than recommended
incorrectly. Recommendation coverage also varied by motion seed:

| Motion seed | 10 Hz recommendations | 40 Hz recommendations | 100 Hz recommendations |
|---:|---:|---:|---:|
| 730301 | 123 | 96 | 0 |
| 730302 | 192 | 96 | 36 |
| 730303 | 192 | 174 | 96 |

This is a generator-specific sensitivity result, not evidence that a lower sample
rate is generally better. The study fixes noise *per sample*, not continuous-time
noise density, so changing cadence also changes the observation count and the
effective noise process per unit time. The 10 Hz cohort's `low` labels and the
other cohorts' `indeterminate` labels are heuristic outputs, not calibrated
confidence. The paired OU follow-up below addresses the noise-model gap, but uses
the same motion seeds and is not held-out validation.

All three assessments used Windows/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`, and implementation
SHA-256 `266aaaf3f08aa96ec818dec09de3fea5b98e6dfccd7a014f3563c24ebb4bae7a`.
Regenerate each four-second cohort with the default motion seeds and noise
settings:

```bash
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-10hz-3seed --sample-count 41 --sample-period-s 0.1
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-40hz-3seed --sample-count 161 --sample-period-s 0.025
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-100hz-3seed --sample-count 401 --sample-period-s 0.01
for cadence in 10hz 40hz 100hz; do
  python scripts/validate_corpus.py \
    ".validation-corpus-euler-${cadence}-3seed/manifest.json" \
    --checkpoint ".validation-corpus-euler-${cadence}-3seed/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress \
    --output ".validation-corpus-euler-${cadence}-3seed/assessment.json"
done
```

### Matched cross-family noise ladder and covariance audit

To determine whether the burst-family abstentions were specific to that motion
shape, the same four-second, 25 ms, single-truth-convention design was run on three
sampled-rate families: Fourier, piecewise-linear, and maneuver bursts. Each used
motion seeds 732501--732503 and noise seeds 20311601--20311620 at each noise level,
for 180 noisy cases per family. The paired noise standard deviations were
`(0.0015 rad, 0.012 rad/s)`, `(0.003 rad, 0.025 rad/s)`, and
`(0.006 rad, 0.05 rad/s)` for low, moderate, and high noise respectively.

| Motion family | Low recommendations / 60 | Moderate / 60 | High / 60 | Total / 180 | Colored-reference / package-i.i.d. mean LLR ratio* |
|---|---:|---:|---:|---:|---:|
| Fourier | 60 | 28 | 20 | 108 | 38.1--41.0x |
| Piecewise-linear | 60 | 25 | 0 | 85 | 33.5--35.9x |
| Maneuver bursts | 0 | 0 | 0 | 0 | 34.6--36.9x |

Counts are correct recommendations; all remaining cases abstained. Across all 540
cases there were no wrong data winners, evidence-set misses, operational-class
misses, or incorrect recommendations. This is a strong family dependence in
coverage under this one truth convention and cadence, not a pooled accuracy
estimate. In one representative low-noise trial, the burst record accumulated
about 1.08 rad of total rotation versus 2.21--2.59 rad for the other families and
remained inside the configured near-tie margin. That is consistent with weaker
separation for this particular burst record, but is not a causal result or a basis
for changing the margin.

The starred score ratios compare the mean full-recording known-noise colored
reference LLR against the package's i.i.d. LLR in each family/noise cell. The
reference uses the synthetic manifest's known noise parameters, the labeled truth's
scalar covariance for every ranked class, and a first-order small-angle covariance
approximation. It does not model candidate-specific transformed covariance, prefix
stability, or package decisions. These scores are not calibrated likelihood ratios
and must not be used to tune thresholds or replace current inference.

The separate residual-covariance audit scored the truth and six one-field
counterfactuals on the same 540 cases. Truth residuals whitened by the stated
covariance had centered sum-of-squares per degree of freedom close to one. A
training/holdout split by noise-seed parity (30 cases per side and noise cell)
compared empirical covariance blocks fitted on even seeds with the analytic scalar
reference on odd seeds:

| Motion family | Centered whitened SS / DOF | Empirical covariance fit: held-out RMS block error | Scalar reference: held-out RMS block error |
|---|---:|---:|---:|
| Fourier | 0.985--0.986 | 11.0% | 10.0% |
| Piecewise-linear | 0.977 | 9.8% | 10.6% |
| Maneuver bursts | 0.973 | 12.7% | 12.2% |

Errors are normalized by the truth model's zero-lag variance. The empirical fit is
slightly better for piecewise-linear motion but slightly worse for Fourier and
bursts, so this small synthetic holdout does not show a consistent benefit from
learning covariance blocks. The covariance approximation is useful for diagnosing
the residual dependence, not yet a validated production evidence model.

Reproduce the matched corpora and both offline diagnostics with fresh output
directories (the generated corpora/reports are ignored local artifacts; choose a
new suffix if rerunning):

```bash
set -e

for spec in \
  "fourier sampled-rate-nonanalytic" \
  "piecewise sampled-rate-piecewise-linear" \
  "bursts sampled-rate-maneuver-bursts"
do
  read -r family motion_profile <<< "$spec"
  outdir=".validation-corpus-cross-family-${family}-732501-503-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile all --noise-replicates 20 \
    --noise-seed-start 20311601 --motion-profile "$motion_profile" \
    --motion-seeds 732501 732502 732503 --noise-sample-counts 161 \
    --sample-period-s 0.025
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 50 --output "$outdir/assessment.json"
  python -m scripts.compare_correlated_evidence "$outdir/manifest.json" \
    --max-cases-per-condition 60 --output "$outdir/colored-evidence.json"
  python -m scripts.validate_correlated_residual_model "$outdir/manifest.json" \
    --max-cases-per-condition 60 --output "$outdir/residual-model.json"
done
```

The assessments ran on Windows 11/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
package `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`. The package-plus-validator
implementation SHA-256 was
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.
The SHA-256 values for `scripts/generate_validation_corpus.py`,
`scripts/validation_noise.py`, `scripts/validate_corpus.py`,
`scripts/compare_correlated_evidence.py`, `scripts/correlated_evidence.py`, and
`scripts/validate_correlated_residual_model.py` were, respectively:

```text
ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa
a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f
d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d
4f9ee7b121fe0e9d4e93eddfeffbd38ee1b84c9428976e30b9740494bd5402f9
1e0599aec9e7e6f00604f55289ad33733c9b802b44037bfe24fae6b9b7b0d5fb
2df21bfe990622f67de3d819fcfc90349b1b9b652b0ecf837c394bd4f47de842
```

### Matched full-convention sweep across sampled-rate families

A follow-up expanded the paired family comparison to every one of the 192
quaternion/rate truth conventions. It reused motion seeds 732501--732503, one
noise seed (20311701) for each convention/motion/noise cell, the same 161-sample
four-second records, and all three noise levels. This produced 1,728 noisy cases
per motion family. Each cell has one noise realization, so this is broad
convention coverage but not a replicated accuracy estimate.

| Motion family | Low recommendations / 576 | Moderate / 576 | High / 576 | Total / 1,728 |
|---|---:|---:|---:|---:|
| Fourier | 576 | 336 | 192 | 1,104 |
| Piecewise-linear | 576 | 192 | 0 | 768 |
| Maneuver bursts | 0 | 0 | 0 | 0 |

Across all 5,184 noisy cases, the true class remained in the data/evidence and
operational winners; there were no incorrect recommendations. Every
non-recommendation was an abstention. For the burst family, low- and moderate-noise
cases all had multiple evidence contenders; at high noise, 480/576 were
evidence-insufficient and the remaining 96 still had multiple contenders. Fourier
and piecewise abstentions were all due to multiple contenders, not insufficient
evidence. The family-dependent coverage gap therefore persists across the full
truth matrix, not just the base convention used in the paired noise ladder.

Coverage also varies sharply by motion seed. Each value below is correct
recommendations out of 576 convention/noise cases for that seed:

| Motion family | Seed 732501 | Seed 732502 | Seed 732503 |
|---|---:|---:|---:|
| Fourier | 336 | 576 | 192 |
| Piecewise-linear | 384 | 192 | 192 |
| Maneuver bursts | 0 | 0 | 0 |

Recommendation counts were balanced across quaternion storage, attitude
direction, rate order, units, and sign. One exploratory exception was Fourier
rate-frame coverage: body-frame truth received 528/864 recommendations versus
576/864 for world-frame truth. The difference came entirely from moderate noise
(144/288 body versus 192/288 world); low- and high-noise frame counts were equal.
Because each cell has only one noise realization and motion-seed effects are large,
this is a replication target, not evidence of a systematic frame bias.

These results reinforce conservative abstention and seed-sensitive coverage; they
do not support changing the near-tie margin, evidence thresholds, or confidence
labels. They remain synthetic and use one cadence.

Reproduce with fresh output directories (these corpora and assessments are ignored
local artifacts; choose a new suffix when rerunning):

```bash
set -e

for spec in \
  "fourier sampled-rate-nonanalytic" \
  "piecewise sampled-rate-piecewise-linear" \
  "bursts sampled-rate-maneuver-bursts"
do
  read -r family motion_profile <<< "$spec"
  outdir=".validation-corpus-cross-family-multiconvention-${family}-732501-503-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile all --noise-replicates 1 \
    --noise-seed-start 20311701 --motion-profile "$motion_profile" \
    --motion-seeds 732501 732502 732503 --noise-sample-counts 161 \
    --sample-period-s 0.025 --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
done
```

The assessment runtime and package implementation SHA-256 match the preceding
cross-family study. Manifest and assessment SHA-256 values are:

```text
fourier manifest     12a914f4f4b80cad6e3bfac3ba4bd7ab817807d5aea46f88ae13968379adb0e4
fourier assessment   9c26ce8635a9ebbef819a5df5c836518b378806db67deb6fc16cc50e057f1d6c
piecewise manifest   4b596e69e077389004e7c6b47c4a37fb9ee8e0e265e36001932ad85b8f20092e
piecewise assessment c2c50b64c9792897f65b875bbf82ca21d8f6fc678279edca415248464840c747
bursts manifest      da5c259db1331e38794c5247f8078c8518649d60bf91344bd169b65c7abfeb9f
bursts assessment    5de6e00673e12d3208d50bc029a28b08ad9ed07de9d7fb8032e0ea9bbe676a06
```

### Independent-noise replication of the Fourier rate-frame comparison

The exploratory body/world rate-frame gap above was checked with five new
moderate-noise seeds (`20311711`--`20311715`) for each of the same three Fourier
motion seeds and all 192 truth conventions. Each motion seed was assessed in its
own 960-case corpus. The comparison below also includes the original matched
moderate-noise cell at seed `20311701`, for six independent noise seeds total.
Each frame has 96 non-frame convention combinations per motion/noise pair, giving
1,728 paired convention/motion/noise combinations (3,456 individual cases) across
the six seeds.

| Motion seed | Body-frame recommendations / 576 | World-frame recommendations / 576 | Both recommend | World-only | Neither |
|---:|---:|---:|---:|---:|---:|
| 732501 | 432 | 576 | 432 | 144 | 0 |
| 732502 | 576 | 576 | 576 | 0 | 0 |
| 732503 | 0 | 0 | 0 | 0 | 576 |
| Total | 1,008 / 1,728 | 1,152 / 1,728 | 1,008 | 144 | 576 |

There were no body-only recommendations. All 144 world-only pairs occurred on
motion seed 732501 and in three of the six noise seeds (`20311701`, `20311714`,
and `20311715`). In each discordant pair, the world-frame case had unique sufficient
evidence, while the body-frame case abstained with multiple evidence contenders.
Both alternatives retained the correct data/evidence/operational winner; there
were no false recommendations. The other two motion seeds had no frame difference:
seed 732502 recommended every paired case, while seed 732503 abstained on every
paired case.

The aggregate coverage gap is now 8.3 percentage points (58.3% body versus 66.7%
world), smaller than the 16.7-point gap in the single-seed observation. Its complete
concentration in one motion seed, alongside the much larger seed-to-seed coverage
spread, points to a motion-specific ambiguity in this cohort rather than evidence
of a general frame preference. It remains a synthetic, one-family, one-cadence
result; more motion seeds and other rate families are needed before ruling out a
systematic effect. No inference threshold or behavior was changed.

The five new corpora use the same four-second, 25 ms cadence and moderate noise
(`0.003 rad`, `0.025 rad/s`). Reproduce them with fresh output directories; seed
`20311701` is reproduced by the full-convention command in the preceding section:

```bash
set -e

for motion_seed in 732501 732502 732503; do
  outdir=".validation-corpus-fourier-rateframe-${motion_seed}-5noise-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile moderate --noise-replicates 5 \
    --noise-seed-start 20311711 --motion-profile sampled-rate-nonanalytic \
    --motion-seed "$motion_seed" --noise-sample-counts 161 \
    --sample-period-s 0.025 --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
done
```

The three manifests and assessments were ignored local artifacts. Their SHA-256
values are:

```text
seed 732501 manifest   7ddff909977d837cc66802248964ff8be514783fa815b71e161bcd05b8546593
seed 732501 assessment e6b896e4e3cfa942141b84100a036b35005bd60fcb5d0024f0d25046188f4e16
seed 732502 manifest   b62c9756747e4186bbab9d1af9d59a20f8b59f098102c07939ba2a80b67a3251
seed 732502 assessment 5a4d965f0be297d71c62e4c8c3a0c9fbd8b41394e450d9ab5e72e7c996027346
seed 732503 manifest   81a54a383400a9a2ed2d3e9c1aa18f22906a2b0621a8ca41b226866e65c88b5f
seed 732503 assessment 70fbf7565a44f6604dab5fcf35ac96cbb33096989899ba5db8ddd69442633cfc
```

### Cadence sensitivity with paired time-domain noise

The Euler generator can also add stationary Ornstein--Uhlenbeck (OU) measurement
noise with a fixed real-time covariance:

\[
\operatorname{Cov}(n(t), n(t+\Delta t)) = \sigma^2 e^{-|\Delta t|/\tau}.
\]

Here the orientation-coordinate marginal standard deviation remained 0.0015 rad
with a 50 ms correlation time; the rate marginal standard deviation remained
0.012 rad/s with a 100 ms correlation time. The OU paths use a shared 1 ms
endpoint grid; intermediate half-millisecond values are drawn from the exact OU
bridge conditioned on neighboring endpoints. Integer-millisecond values retain
the legacy seeded process, so the 10, 40, and 100 Hz cohorts share values at
common timestamps. The manifest records each channel's time constant and its
cadence-specific discrete coefficient `rho = exp(-sample_period_s / tau)`. OU
mode accepts sample periods that are multiples of 0.5 ms, including 12.5 ms for
80 Hz.

The three cadences again cover the complete 192-convention matrix for each of the
same three motion seeds, for 1,728 cases total. All recommendations were correct;
truth remained in the evidence contenders in every case. The dense-cadence raw
winner misses resulted in abstention, not an incorrect recommendation.

| Cadence | Samples | Correct recommendations | Truth-retaining abstentions | Data-winner / operational misses | Evidence-insufficient | Evidence misses / wrong recommendations | Confidence labels |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 Hz | 41 | 504 (87.5%) | 72 | 0 / 0 | 5 | 0 / 0 | 121 low, 455 indeterminate |
| 40 Hz | 161 | 382 (66.3%) | 194 | 0 / 0 | 35 | 0 / 0 | 18 low, 558 indeterminate |
| 100 Hz | 401 | 238 (41.3%) | 338 | 16 / 16 | 43 | 0 / 0 | 576 indeterminate |

All 16 data-winner misses at 100 Hz were degree-unit cases; truth stayed in the
evidence set and the decision abstained. Compared with the fixed-per-sample
white-noise cohort above, the OU model increased 100 Hz recommendation coverage
from 132/576 to 238/576, while data-winner misses fell from 54 to 16 and
evidence-insufficient cases fell from 132 to 43. This is a change in a controlled
synthetic noise assumption, not evidence that the OU process describes a real
sensor or that its coverage is calibrated. The confidence labels remain heuristic.

Coverage also varies by motion seed:

| Motion seed | 10 Hz recommendations | 40 Hz recommendations | 100 Hz recommendations |
|---:|---:|---:|---:|
| 730301 | 120 | 96 | 34 |
| 730302 | 192 | 96 | 96 |
| 730303 | 192 | 190 | 108 |

These runs used Windows/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, package version
`0.0.1a0.post1.dev25+gc2594b4a8.d20261031`, and validator/inference implementation
SHA-256 `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.
The independent generator's SHA-256 was
`60460a9ae659213d29a1603dc1555d1135ecd6ef457940c41914bb50454a3b5b`.
Regenerate the paired four-second cohorts with the default motion/noise seeds:

```bash
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-10hz-3seed --sample-count 41 \
  --sample-period-s 0.1 --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-40hz-3seed --sample-count 161 \
  --sample-period-s 0.025 --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-100hz-3seed --sample-count 401 \
  --sample-period-s 0.01 --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1
for cadence in 10hz 40hz 100hz; do
  python scripts/validate_corpus.py \
    ".validation-corpus-euler-ou-${cadence}-3seed/manifest.json" \
    --checkpoint ".validation-corpus-euler-ou-${cadence}-3seed/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress \
    --output ".validation-corpus-euler-ou-${cadence}-3seed/assessment.json"
done
```

The OU time-domain path is tested for identical seeded values at timestamps shared
by all three cadences, and the assessment schema now preserves the continuous-time
constants alongside the discrete coefficients.

### Fresh motion/noise seed block

To measure seed sensitivity without changing the motion or noise model, the same
OU experiment was repeated with motion seeds 730304--730306 and noise seed
20301032. It again covered the complete 192-convention matrix at each cadence,
giving 1,152 cases per cadence and 3,456 cases overall across the original and
fresh seed blocks.

| Cadence | Cases | Correct recommendations | Truth-retaining abstentions | Data-winner / operational misses | Evidence-insufficient | Evidence misses / wrong recommendations |
|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 1,152 | 1,050 (91.1%) | 102 | 0 / 0 | 5 | 0 / 0 |
| 40 Hz | 1,152 | 744 (64.6%) | 408 | 0 / 0 | 45 | 0 / 0 |
| 100 Hz | 1,152 | 466 (40.5%) | 686 | 21 / 21 | 57 | 0 / 0 |

All 21 data-winner misses at 100 Hz were degree-unit cases; truth remained in the
evidence set and the decision abstained. Recommendation counts by individual
motion seed show considerable variation even under this paired noise model:

| Motion seed | 10 Hz recommendations | 40 Hz recommendations | 100 Hz recommendations |
|---:|---:|---:|---:|
| 730301 | 120 | 96 | 34 |
| 730302 | 192 | 96 | 96 |
| 730303 | 192 | 190 | 108 |
| 730304 | 192 | 144 | 96 |
| 730305 | 191 | 122 | 96 |
| 730306 | 163 | 96 | 36 |

The fresh block supports the same broad cadence/seed sensitivity pattern; it does
not validate the Fourier motion family against real trajectories or another motion
model. These six seeds and two noise seeds are synthetic regression cases only.
Do not interpret the zero evidence-miss and zero wrong-recommendation counts as
field rates or a calibration guarantee.

Reproduce the fresh seed block with:

```bash
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-fresh-10hz --motion-seeds 730304 730305 730306 \
  --noise-seed 20301032 --sample-count 41 --sample-period-s 0.1 \
  --orientation-noise-correlation-time-s 0.05 --rate-noise-correlation-time-s 0.1
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-fresh-40hz --motion-seeds 730304 730305 730306 \
  --noise-seed 20301032 --sample-count 161 --sample-period-s 0.025 \
  --orientation-noise-correlation-time-s 0.05 --rate-noise-correlation-time-s 0.1
python scripts/generate_euler_validation_corpus.py \
  --outdir .validation-corpus-euler-ou-fresh-100hz --motion-seeds 730304 730305 730306 \
  --noise-seed 20301032 --sample-count 401 --sample-period-s 0.01 \
  --orientation-noise-correlation-time-s 0.05 --rate-noise-correlation-time-s 0.1
for cadence in 10hz 40hz 100hz; do
  python scripts/validate_corpus.py \
    ".validation-corpus-euler-ou-fresh-${cadence}/manifest.json" \
    --checkpoint ".validation-corpus-euler-ou-fresh-${cadence}/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress \
    --output ".validation-corpus-euler-ou-fresh-${cadence}/assessment.json"
done
```

### OU correlation-time sensitivity screen

The preceding cadence study used orientation/rate OU correlation times of
50/100 ms. This follow-up compared that reference with a shorter 12.5/25 ms pair
and a longer 200/400 ms pair. Marginal noise standard deviations remained
0.0015 rad and 0.012 rad/s; the shared 1 ms OU grid, four-second duration, and
10/40/100 Hz cadences were unchanged. Thus the experiment varies the *joint*
orientation/rate correlation scale, not either channel independently.

For a balanced comparison, the table below uses the same motion seed (730301),
noise seed (20301031), and all 192 convention encodings in each cell. The
50/100 ms reference values are extracted from the earlier six-seed study. Every
recommendation in these cells was correct; there were no evidence misses or
incorrect recommendations. Data-winner and operational misses are shown
separately from abstentions because a raw ranking miss did not imply a false
recommendation.

| Orientation/rate τ | Cadence | Cases | Correct recommendations | Truth-retaining abstentions | Data-winner / operational misses | Evidence-insufficient |
|---:|---:|---:|---:|---:|---:|---:|
| 50 / 100 ms (reference) | 10 Hz | 192 | 120 | 72 | 0 / 0 | 5 |
| 50 / 100 ms (reference) | 40 Hz | 192 | 96 | 96 | 0 / 0 | 35 |
| 50 / 100 ms (reference) | 100 Hz | 192 | 34 | 158 | 16 / 16 | 32 |
| 12.5 / 25 ms (short) | 10 Hz | 192 | 120 | 72 | 0 / 0 | 7 |
| 12.5 / 25 ms (short) | 40 Hz | 192 | 96 | 96 | 2 / 2 | 46 |
| 12.5 / 25 ms (short) | 100 Hz | 192 | 0 | 192 | 47 / 47 | 13 |
| 200 / 400 ms (long) | 10 Hz | 192 | 117 | 75 | 0 / 0 | 0 |
| 200 / 400 ms (long) | 40 Hz | 192 | 96 | 96 | 0 / 0 | 3 |
| 200 / 400 ms (long) | 100 Hz | 192 | 96 | 96 | 0 / 0 | 12 |

The 100 Hz matched-seed coverage changed from 34/192 under the reference
correlation times to 0/192 under the short pair and 96/192 under the long pair.
The short-pair 100 Hz cohort had 47 raw winner/operational misses, but the true
class remained in the evidence set and the policy abstained. This is a useful
stress result: cadence outcomes are sensitive to the assumed temporal covariance.
It is not a threshold-tuning target or evidence that any of these time constants
describes a deployed sensor.

The short-pair sweep was then expanded to motion seeds 730302 and 730303 at all
three cadences. Across its three-seed cohorts, correct recommendations were
504/576 at 10 Hz, 368/576 at 40 Hz, and 168/576 at 100 Hz. Data-winner / operational
misses were 0/0, 2/2, and 47/47; evidence-insufficient cases were 7, 48, and 61.
There were no evidence misses or incorrect recommendations. The 100 Hz outcomes
varied sharply by motion seed:

| Motion seed | 10 Hz recommendations | 40 Hz recommendations | 100 Hz recommendations | 100 Hz winner / operational misses | 100 Hz evidence-insufficient |
|---:|---:|---:|---:|---:|---:|
| 730301 | 120 | 96 | 0 | 47 / 47 | 13 |
| 730302 | 192 | 96 | 72 | 0 / 0 | 48 |
| 730303 | 192 | 176 | 96 | 0 / 0 | 0 |

Thus every short-pair 100 Hz winner miss came from seed 730301; the other two
seeds had 72 and 96 correct recommendations without raw winner misses. The long
pair still uses only seed 730301, so comparisons against it are not a balanced
multi-seed estimate. The extension was kept to three seeds because 100 Hz
inference took about one case per second on this host.

These are synthetic cases from one Euler-Fourier motion family. Even the matched
192-case cells share one physical motion seed and are convention encodings of the
same trajectory, not independent recordings. The results measure sensitivity and
regression behavior; they are not field accuracy, a population rate, or confidence
calibration. Later sections add two non-Euler sampled-rate families; independent
external logs remain an open validation gate.

The run used Windows/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, package version
`0.0.1a0.post1.dev25+gc2594b4a8.d20261031`, and inference implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`. The
independent generator SHA-256 was
`60460a9ae659213d29a1603dc1555d1135ecd6ef457940c41914bb50454a3b5b`.
Reproduce the balanced single-motion-seed comparison with:

```bash
for profile in fast slow; do
  if [ "$profile" = fast ]; then
    orientation_tau=0.0125
    rate_tau=0.025
  else
    orientation_tau=0.2
    rate_tau=0.4
  fi
  for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
    read -r name count period <<< "$cadence"
    outdir=".validation-corpus-euler-ou-tau-${profile}-single-${name}"
    python scripts/generate_euler_validation_corpus.py \
      --outdir "$outdir" --motion-seeds 730301 --noise-seed 20301031 \
      --sample-count "$count" --sample-period-s "$period" \
      --orientation-noise-correlation-time-s "$orientation_tau" \
      --rate-noise-correlation-time-s "$rate_tau"
    python scripts/validate_corpus.py "$outdir/manifest.json" \
      --checkpoint "$outdir/assessment.checkpoint.json" \
      --checkpoint-every 100 --progress \
      --output "$outdir/assessment.json"
  done
done
```

The three-seed short-pair cohorts use the same generator settings with
`--motion-seeds 730301 730302 730303`, output directories ending in
`fast-seed-a-10hz`, `fast-seed-a-40hz`, and `fast-seed-a-100hz`, and cadence
settings `(41, 0.1)`, `(161, 0.025)`, and `(401, 0.01)` for sample count and period.

### Sampled-rate OU cadence and noise-level screen

The generic corpus generator now uses the shared OU sampler in
`scripts/validation_noise.py` for its numerically integrated
`sampled-rate-nonanalytic` motion profile. The profile is a seeded Fourier mixture
of body rates integrated on an independent fixed grid; it does not parameterize
attitude with Euler angles. `--sample-period-s` sets the observation cadence, and
the per-channel `--*-noise-correlation-time-s` options derive the sample-lag
coefficient `rho = exp(-sample_period_s / tau)`. OU sample periods must be
multiples of 0.5 ms; intermediate samples use a conditional OU bridge, leaving
seeded integer-millisecond values unchanged. Explicit non-default AR(1)
coefficients for a channel cannot be combined with that channel's OU time constant.

At low noise (0.0015 rad orientation and 0.012 rad/s rate), one sampled-rate
motion seed (730201) and noise seed (20301031) were assessed across all 192
quaternion/rate convention encodings at each four-second cadence. The OU time
constants were varied together, matching the Euler screen above:

| Orientation/rate tau | 10 Hz | 40 Hz | 100 Hz |
|---:|---:|---:|---:|
| 12.5 / 25 ms | 192/192 recommendations | 192/192 recommendations | 192/192 recommendations |
| 50 / 100 ms | 192/192 recommendations | 192/192 recommendations | 192/192 recommendations |
| 200 / 400 ms | 192/192 recommendations | 192/192 recommendations | 192/192 recommendations |

There were no abstentions, raw winner misses, evidence misses, or incorrect
recommendations in these nine cells. At the reference 50/100 ms time constants,
the same motion/noise seeds were also assessed at moderate noise (0.003 rad,
0.025 rad/s) and high noise (0.006 rad, 0.05 rad/s):

| Noise profile | 10 Hz | 40 Hz | 100 Hz |
|---|---:|---:|---:|
| Moderate | 192/192 recommendations | 192/192 recommendations | 192/192 recommendations |
| High | 192/192 recommendations | 192/192 recommendations | 0/192 recommendations; 192 abstentions |

Every recommendation remained correct. The high-noise 100 Hz cell abstained
because multiple evidence contenders remained; truth was retained and there were
no raw winner misses. This abstention behavior repeated for motion seeds 730202
and 730203: across seeds 730201--730203, all 576 high-noise 100 Hz convention
cases abstained for the same reason, with zero winner misses, evidence misses, or
incorrect recommendations. These three motion seeds share one noise seed, so this
does not measure variability over independent noise realizations.

The low-noise time-constant matrix has one motion seed and one noise seed per
cell. Its 192 convention encodings reuse the same physical trajectory and seeded
noise path; they are not 192 independent recordings. The result shows that this
sampled-rate family is easy for the current policy under these low/moderate
conditions and that high-noise 100 Hz data trigger conservative abstention. It is
not an accuracy estimate, false-positive guarantee, or calibration result.

The assessment used Windows/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, package
version `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`, and inference implementation
SHA-256 `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.
The generic generator SHA-256 was
`946edaca1790e859d64679e7035cad1bd6c5e6158d3803e24304cc96dbebefb4`; the shared
noise module SHA-256 was
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`.
Reproduce the low-noise time-constant matrix with:

```bash
for profile in short reference long; do
  case "$profile" in
    short) orientation_tau=0.0125; rate_tau=0.025 ;;
    reference) orientation_tau=0.05; rate_tau=0.1 ;;
    long) orientation_tau=0.2; rate_tau=0.4 ;;
  esac
  for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
    read -r name count period <<< "$cadence"
    outdir=".validation-corpus-sampled-rate-ou-cadence-730201-${profile}-${name}"
    python scripts/generate_validation_corpus.py \
      --outdir "$outdir" --noise-profile low --noise-seed-start 20301031 \
      --motion-seeds 730201 --noise-sample-counts "$count" \
      --sample-period-s "$period" --truth-convention-sweep \
      --motion-profile sampled-rate-nonanalytic \
      --orientation-noise-correlation-time-s "$orientation_tau" \
      --rate-noise-correlation-time-s "$rate_tau"
    python scripts/validate_corpus.py "$outdir/manifest.json" \
      --checkpoint "$outdir/assessment.checkpoint.json" \
      --checkpoint-every 50 --progress \
      --output "$outdir/assessment.json"
  done
done
```

The moderate/high reference-time-constant cohorts use the same loop with noise
profiles `moderate` and `high`, output directories under
`.validation-corpus-sampled-rate-ou-stress-730201-`, and both time constants set
to 0.05/0.1 seconds. The extra high-noise 100 Hz seed check used motion seeds
730202 and 730203, sample count 401, and sample period 0.01 seconds.

### Balanced sampled-rate OU seed and cadence follow-up

On 2026-10-31, a paired follow-up varied motion, noise, OU correlation time, and
observation cadence together. Each cell used three independently seeded
`sampled-rate-nonanalytic` motions (730401--730403), 20 shared noise seeds
(20311031--20311050), and the low, moderate, and high noise profiles. The full
four-second recordings were sampled at 10, 40, or 100 Hz (41, 161, or 401 samples)
and used orientation/rate OU time-constant pairs of 12.5/25 ms, 50/100 ms, or
200/400 ms. Equal noise seeds were paired across motions, noise levels, correlation
times, and cadences; the shared 1 ms process makes cadence comparisons use common
noise values at common timestamps. The correlation-time changes alter the process
itself, and the three motion seeds all come from one Fourier-rate family.

There were 60 noisy cases in each time-constant/cadence/noise cell (three motions by
20 seeds), for 1,620 noisy cases total. Each generated corpus also contains three
clean control cases, assessed separately from these results. Table entries are
correct recommendations out of 60; unlisted cases abstained.

| Orientation/rate tau | Cadence | Low noise | Moderate noise | High noise |
|---:|---:|---:|---:|---:|
| 12.5 / 25 ms | 10 Hz | 60 | 60 | 39 |
| 12.5 / 25 ms | 40 Hz | 58 | 20 | 13 |
| 12.5 / 25 ms | 100 Hz | 20 | 0 | 0 |
| 50 / 100 ms | 10 Hz | 60 | 60 | 33 |
| 50 / 100 ms | 40 Hz | 60 | 20 | 20 |
| 50 / 100 ms | 100 Hz | 20 | 20 | 0 |
| 200 / 400 ms | 10 Hz | 60 | 59 | 30 |
| 200 / 400 ms | 40 Hz | 60 | 38 | 20 |
| 200 / 400 ms | 100 Hz | 60 | 20 | 20 |

Across the three time-constant/cadence pairs, low noise yielded 458/540
recommendations, moderate noise 297/540, and high noise 175/540. Overall there were
930 recommendations and 690 abstentions. All 930 recommendations were correct;
all 690 abstentions cited `multiple_evidence_contenders`. There were no data-winner
misses, evidence misses, or incorrect recommendations. Coverage also varied sharply
by physical motion seed:

| Motion seed | Low recommendations / 180 | Moderate / 180 | High / 180 | Total / 540 |
|---:|---:|---:|---:|---:|
| 730401 | 138 | 59 | 0 | 197 |
| 730402 | 140 | 78 | 42 | 260 |
| 730403 | 180 | 160 | 133 | 473 |

This motion dependence is important: the pooled coverage is not a portable success
rate, and the three rate paths are not independent motion families. This run tests
one convention rather than the full 192-value convention matrix. The paired seeds
make within-generator sensitivity comparisons easier to interpret but do not create
independent real recordings, validate a deployment logger, establish calibration,
or justify changing thresholds. In particular, the zero-winner-miss result applies
only to this synthetic generator, convention, and tested grid.

The assessment ran on Windows 11/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and
package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`. Its inference
implementation SHA-256 was
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`; source SHA-256
values were `946edaca1790e859d64679e7035cad1bd6c5e6158d3803e24304cc96dbebefb4` for
`scripts/generate_validation_corpus.py`,
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f` for
`scripts/validation_noise.py`, and
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d` for
`scripts/validate_corpus.py`.

Reproduce the nine paired corpora with:

```bash
for profile in fast reference slow; do
  case "$profile" in
    fast) orientation_tau=0.0125; rate_tau=0.025 ;;
    reference) orientation_tau=0.05; rate_tau=0.1 ;;
    slow) orientation_tau=0.2; rate_tau=0.4 ;;
  esac
  for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
    read -r name count period <<< "$cadence"
    outdir=".validation-corpus-sampled-rate-ou-balanced-730401-403-${profile}-${name}"
    python -m scripts.generate_validation_corpus \
      --outdir "$outdir" --noise-profile all --noise-replicates 20 \
      --noise-seed-start 20311031 \
      --motion-profile sampled-rate-nonanalytic \
      --motion-seeds 730401 730402 730403 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --orientation-noise-correlation-time-s "$orientation_tau" \
      --rate-noise-correlation-time-s "$rate_tau"
    python -m scripts.validate_corpus "$outdir/manifest.json" \
      --checkpoint "$outdir/assessment.checkpoint.json" \
      --checkpoint-every 50 --progress --output "$outdir/assessment.json"
  done

done
```

The generated corpora are ignored local artifacts; only the study design, results,
and reproduction commands are tracked. Choose new output directories when rerunning
so that existing local assessments are preserved.

### Shared-path Fourier-rate cadence and rate-delay screen

A fresh synthetic screen repeated the paired cadence and 27 ms rate-delay
design with the distinct `sampled-rate-nonanalytic` Fourier-rate motion family.
The generator labels these trajectories from seeded Fourier rates and an
independent fixed-grid trapezoidal quaternion integration. Motion seeds were
742530--742532 and OU-noise seeds were 20313701--20313703. Each four-second
cohort used the full 192-convention sweep, three motion seeds, and three noise
replicates: 1,728 noisy cases plus three analytic controls per assessment.

The 25 ms and 50 ms grids contain 161 and 81 observations, respectively. Both
use orientation/rate OU time constants of 0.05/0.1 s, with no jitter or dropout.
Baseline and delayed conditions share motion/noise seeds; the only within-cadence
measurement-condition difference is `rate_latency_s` (0 versus 0.027 s).

| Noise | Cadence | Correct recommendations, baseline → rate delay | Evidence-set changes | Decision-status changes |
|---|---:|---:|---:|---:|
| Moderate | 25 ms | 1,152 → 1,152 | None | None |
| Moderate | 50 ms | 1,728 → 1,728 | None | None |
| High | 25 ms | 24 → 0 | 384 (360 expanded, 24 contracted) | 24 |
| High | 50 ms | 1,080 → 1,080 | None | None |

The 24 high-noise 25 ms recommendations all came from motion seed 742530 and
became truth-retaining abstentions under the 27 ms delay. No paired rate-delay
comparison changed the data-winner set or operational class. Every recommendation
was correct; there were no data-winner, operational, evidence, evidence-insufficiency,
or recommendation misses. At 50 ms high noise, the 1,080 recommendations were
seed-dependent: 576/576 on seed 742530, 504/576 on 742531, and 0/576 on 742532.

The 25-to-50 ms comparisons used the same OU paths. For each noise/latency pair,
the comparator verified all 139,968 common-time rows across 1,728 noisy cases;
the maximum shared timestamp/orientation/rate difference was exactly zero, and
the inference configurations and implementation hashes matched.

| Noise | Condition | Correct recommendations, 25 ms → 50 ms | Evidence-set changes (all contractions) | Decision-status changes |
|---|---|---:|---:|---:|
| Moderate | Baseline | 1,152 → 1,728 | 576 | 576 |
| Moderate | Rate delay 27 ms | 1,152 → 1,728 | 576 | 576 |
| High | Baseline | 24 → 1,080 | 1,704 | 1,056 |
| High | Rate delay 27 ms | 0 → 1,080 | 1,728 | 1,080 |

All cross-cadence changes were evidence-set contractions; none changed the data
winner or operational class. At moderate noise, all 576 added recommendations
were on seed 742532. At high noise, 25 ms had very sparse coverage while 50 ms
recommended on seeds 742530 and 742531. The larger recommendation coverage at
the coarser cadence is a sensitivity signal, not evidence that 50 ms is better:
the comparison has only three synthetic motion seeds and does not calibrate the
evidence threshold or account for all forms of temporal dependence. Do not tune
thresholds from this screen; repeat on more independent motions and evaluate the
correlated-residual evidence model before making confidence or calibration claims.

A first-order residual-covariance diagnostic was also run on the four synchronized
baseline corpora (1,728 cases each; latency and jitter are unsupported by this
validator). Across the 192 truth-convention groups, normalized residual diagnostics
were similar at both noise levels: median centered/uncentered whitened sum of
squares per degree of freedom was 0.837/0.970 at 25 ms and 0.793/0.949 at 50 ms.
Median replicate-mean residual RMS was 0.999/1.006 noise standard deviations;
median zero-lag trace error was -0.156/-0.200 of modeled variance, and median
zero-lag cross-axis block error was 0.293/0.324.

The covariance holdout used the single even noise seed (20313702; three training
motion cases per convention) and held out two odd seeds (six cases per convention).
A fitted 3×3 covariance had higher held-out RMS block error than the known scalar
OU reference in all 192 truth groups: median fitted/reference errors were
0.402/0.351 at 25 ms and 0.408/0.399 at 50 ms. This tiny holdout does not support
integrating a fitted covariance into package decisions; collect more independent
noise paths and held-out motions first.

The baseline covariance reports are reproducible with:

```bash
for noise in moderate high; do
  for cadence in 25ms 50ms; do
    outdir="$root/$noise-$cadence-baseline"
    python -m scripts.validate_correlated_residual_model "$outdir/manifest.json" \
      --seed-partition all --output "$outdir/correlated-residual-model.json"
  done
done
```

Reproduce the eight corpora, assessments, and paired reports with:

```bash
set -e
root=.validation-corpus/cross-family

for noise in moderate high; do
  for cadence in 25ms 50ms; do
    if [ "$cadence" = 25ms ]; then
      period=0.025
      count=161
    else
      period=0.05
      count=81
    fi
    for condition in baseline rate27; do
      if [ "$condition" = rate27 ]; then latency=0.027; else latency=0.0; fi
      outdir="$root/$noise-$cadence-$condition"
      python -m scripts.generate_validation_corpus \
        --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
        --noise-seed-start 20313701 --motion-profile sampled-rate-nonanalytic \
        --motion-seeds 742530 742531 742532 --noise-sample-counts "$count" \
        --sample-period-s "$period" --truth-convention-sweep \
        --orientation-latencies-s 0.0 --rate-latencies-s "$latency" \
        --orientation-noise-correlation-time-s 0.05 \
        --rate-noise-correlation-time-s 0.1
      python -m scripts.validate_corpus "$outdir/manifest.json" \
        --checkpoint "$outdir/assessment.checkpoint.json" \
        --checkpoint-every 100 --progress --output "$outdir/assessment.json"
    done
  done
done

for noise in moderate high; do
  for condition in baseline rate27; do
    baseline="$root/$noise-25ms-$condition"
    candidate="$root/$noise-50ms-$condition"
    python -m scripts.compare_cadence_assessments \
      "$baseline/assessment.json" "$candidate/assessment.json" \
      --baseline-manifest "$baseline/manifest.json" \
      --candidate-manifest "$candidate/manifest.json" \
      --output "$baseline/cadence-comparison-50ms-$condition.json"
  done
  for cadence in 25ms 50ms; do
    baseline="$root/$noise-$cadence-baseline"
    candidate="$root/$noise-$cadence-rate27"
    python -m scripts.compare_corpus_assessments \
      "$baseline/assessment.json" "$candidate/assessment.json" \
      --output "$candidate/rate-delay-comparison.json"
  done
done

for noise in moderate high; do
  for cadence in 25ms 50ms; do
    outdir="$root/$noise-$cadence-baseline"
    python -m scripts.validate_correlated_residual_model "$outdir/manifest.json" \
      --seed-partition all --output "$outdir/correlated-residual-model.json"
  done
done
```

Artifact SHA-256 values:

```text
moderate 25ms baseline manifest 02713a460410e0fd161d8fe84da4f5272d59dcea90a771318bc53b1ce63d537a
moderate 25ms baseline assessment 97472a92ee7f5f9a06002d4b92896bfffbf049a4d410debf2b39721ad8b5c81e
moderate 25ms rate27 manifest aaf6679f41bba2d8cb91d0bb2b1ebd5dfe734d14a89c16c249cf82de24622bc0
moderate 25ms rate27 assessment 2d4f44560358cb9282c88b3252b61dc7d0b0850ef7caaca028ca3ae58823f986
moderate 50ms baseline manifest 31176cabae4bc4fda6382292e2fdeeb7e9045d860903e23bd4152aa4443886ad
moderate 50ms baseline assessment f1f04647117e670b3d3dfb89695f97099b0996dbe0d66628f2a9920f299afa35
moderate 50ms rate27 manifest ca7a0c0a2afd49cbb4f375303956c7eedf84f070d3ebf20b6ccfd4bd0be53a6e
moderate 50ms rate27 assessment fa142292002226f5ccc89338b0f8fcaf4b02910d460327dbe35c6c6c2210d842
high 25ms baseline manifest 4346ba11a364c2361385d098c157cb38186ec62ceb97d069b3ca15402a578c10
high 25ms baseline assessment b7d35ed8e6fa6185ab9b47d8cf6133c42c17bdd0517e89028925175a0f62604a
high 25ms rate27 manifest 38d3bd49c5a123374d1d73900cb3178c5b3511193bfd903783e109e8294cd9f3
high 25ms rate27 assessment f0c877d1785a8ffc6241a735dd25c7b4c0c5594513e0aab54347fc3ba90e34f1
high 50ms baseline manifest 6729c509e1c93d580a988b91ae1f170410434517ddc6a2530c6a68a43be7f97f
high 50ms baseline assessment ed61ef9ba3b0cc8d85a75878b08d8b993927234b96d3ec5d0e82c050ebc78670
high 50ms rate27 manifest 6b5561aeec946ded706e8c29cf3f4c79c675d96b0a6e341eb4984b47c8310613
high 50ms rate27 assessment ea575aab74f6a3b9f03fa4a7fc001a257a7960b10161e80e55b6b86cd3343656
moderate 25ms rate-delay comparison 987d6aefc9e3a31a74beddf188b8ab77b655b27cda1aa5dc74f0c8c1b1147577
moderate 50ms rate-delay comparison 60d7636df645fd14d9c82d4c4b71ebcfa1e9f6f98c0072663c8d753e86d264f
high 25ms rate-delay comparison 1a811a9b8c865c4999ba2c24e31903bb30d126e3de4cf01c4d0b0918154c4c1b
high 50ms rate-delay comparison 1905605f1f86d74cb5de911b077d33ed6f9f5602c6cf3b2f20fa193c3c04459d
moderate baseline cadence comparison cd648efbac5b503f97d848b91217979bbd8c7fa81d996a3f4c30d0cfc54e4c55
moderate rate27 cadence comparison 07d97ec3f37e879465b5af81ff95cf69e3ab460c1009f06c1912c2e508b791cf
high baseline cadence comparison cbfd5eccdad97708236680e8df9d36461027ade72d8baf93554f389f010104fa
high rate27 cadence comparison 7161ae8837ebfd47fb2dbd2569d879c248a2b4c131bf90ac132c434f3df0087e
moderate 25ms correlated-residual model ebff281897328e59a951ab27f286c606bb00bc33043b4c2a43bd5083df6b78fc
moderate 50ms correlated-residual model 5f551606a6c8189bd24b97994fcda9b2dbc7ac50d4a3ae470b8eca0ce4bc9337
high 25ms correlated-residual model 8bea7872751ef631f254263ac278d89e19d853b8dab1946159c897d9535504b9
high 50ms correlated-residual model afecb1fec9fcc32c485bed747b113b903f0fe8327e0bd3870cdbbbcb4bf8d8d2
```

All eight assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Fresh-seed Fourier-rate replication

A second batch repeated the preceding design with independent motion seeds
742533--742535 and OU-noise seeds 20313704--20313706. It again crossed the full
192-convention sweep with moderate/high noise, 25/50 ms cadence, and 0/27 ms rate
latency. Each assessment contains 1,728 noisy cases plus three clean controls.

The two batches together cover six Fourier-rate motions and six noise seeds.
Aggregated rows below pool the same 1,728-case conditions from both batches
(3,456 paired cases per noise/cadence/latency cell).

| Noise | Cadence | Cases | Correct recommendations, baseline → delay | Evidence-set changes | Status changes |
|---|---:|---:|---:|---:|---:|
| Moderate | 25 ms | 3,456 | 2,304 → 2,304 | None | None |
| Moderate | 50 ms | 3,456 | 3,168 → 3,168 | None | None |
| High | 25 ms | 3,456 | 24 → 0 | 408 (360 expanded, 48 contracted) | 24 |
| High | 50 ms | 3,456 | 1,656 → 1,560 | 96 expanded | 96 |

Moderate-noise rate delay had no paired effect across the six motions. High-noise
delay effects were seed- and cadence-local: 24 recommendations became abstentions
at 25 ms in the first batch, while 96 did so at 50 ms in the second. The second
batch's 25 ms high-noise delay changed 24 evidence sets but no decision statuses.
There is no stable 27 ms latency response in this screen.

| Noise | Condition | Correct recommendations, 25 ms → 50 ms | Evidence-set changes (all contractions) | Status changes |
|---|---|---:|---:|---:|
| Moderate | Baseline | 2,304 → 3,168 | 864 | 864 |
| Moderate | Rate delay 27 ms | 2,304 → 3,168 | 864 | 864 |
| High | Baseline | 24 → 1,656 | 3,360 | 1,632 |
| High | Rate delay 27 ms | 0 → 1,560 | 3,384 | 1,560 |

The cadence response repeated in both batches, but its size depended strongly on
the motion seeds. Moderate-noise gains came from seed 742532 in the first batch
and 742534 in the second. High-noise 25 ms coverage was nearly zero, while 50 ms
recommendations were spread across different seeds in each batch.

Across the eight cross-cadence reports, every common-time input value matched
exactly: 279,936 shared rows per noise/latency pair pooled across both batches
(1,119,744 rows total), maximum absolute difference 0.0. Inference configurations
and implementation hashes matched. Across all 16 paired reports, recommendations
were correct; there were no winner, operational, evidence, evidence-insufficiency,
or false-recommendation misses, and no data-winner-set or operational-class changes.
All measured changes were in evidence sets and recommendation/abstention status.

These results make cadence sensitivity more repeatable for this synthetic
Fourier-rate family, not a general accuracy or calibration claim. The six motions
remain one family. Do not tune thresholds from these pooled numbers; the strong
seed dependence and non-repeatable latency effects call for broader families and
independent logger data.

The residual-covariance holdout also reversed between batches: the first batch's
fitted covariance beat the known scalar OU reference in 0/192 truth groups, while
this batch did so in 192/192. Each fit trained on one even noise seed (three motion
cases per convention) and held out two odd seeds (six cases). Median fitted/scalar
held-out block errors in this batch were 0.497/0.717 at 25 ms and 0.486/0.722 at
50 ms. This reversal shows that the tiny covariance holdout is seed-sensitive;
neither result justifies integrating fitted covariance into inference.

Reproduce the second batch and its reports with:

```bash
set -e
root=.validation-corpus/cross-family-replication-742533-535

for noise in moderate high; do
  for cadence in 25ms 50ms; do
    if [ "$cadence" = 25ms ]; then period=0.025; count=161; else period=0.05; count=81; fi
    for condition in baseline rate27; do
      if [ "$condition" = rate27 ]; then latency=0.027; else latency=0.0; fi
      outdir="$root/$noise-$cadence-$condition"
      python -m scripts.generate_validation_corpus \
        --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
        --noise-seed-start 20313704 --motion-profile sampled-rate-nonanalytic \
        --motion-seeds 742533 742534 742535 --noise-sample-counts "$count" \
        --sample-period-s "$period" --truth-convention-sweep \
        --orientation-latencies-s 0.0 --rate-latencies-s "$latency" \
        --orientation-noise-correlation-time-s 0.05 \
        --rate-noise-correlation-time-s 0.1
      python -m scripts.validate_corpus "$outdir/manifest.json" \
        --checkpoint "$outdir/assessment.checkpoint.json" \
        --checkpoint-every 100 --progress --output "$outdir/assessment.json"
    done
  done
done

for noise in moderate high; do
  for condition in baseline rate27; do
    baseline="$root/$noise-25ms-$condition"
    candidate="$root/$noise-50ms-$condition"
    python -m scripts.compare_cadence_assessments \
      "$baseline/assessment.json" "$candidate/assessment.json" \
      --baseline-manifest "$baseline/manifest.json" \
      --candidate-manifest "$candidate/manifest.json" \
      --output "$baseline/cadence-comparison-50ms-$condition.json"
  done
  for cadence in 25ms 50ms; do
    baseline="$root/$noise-$cadence-baseline"
    candidate="$root/$noise-$cadence-rate27"
    python -m scripts.compare_corpus_assessments \
      "$baseline/assessment.json" "$candidate/assessment.json" \
      --output "$candidate/rate-delay-comparison.json"
  done
  for cadence in 25ms 50ms; do
    outdir="$root/$noise-$cadence-baseline"
    python -m scripts.validate_correlated_residual_model "$outdir/manifest.json" \
      --seed-partition all --output "$outdir/correlated-residual-model.json"
  done
done
```

Artifact SHA-256 values:

```text
moderate 25ms baseline manifest ce300ceb1fe9a344d419f0b36f54b047fed97f18a63fd30cec77988b9017ff47
moderate 25ms baseline assessment 557dfdbd944912421bcce075a4b0f3709cf52ba041e4d0d6204aab3ae9e856b3
moderate 25ms rate27 manifest 4eceba0e174cb3f10dec5a5f9742e14103b2ecedcb4802154f1924cf13e72d5b
moderate 25ms rate27 assessment 2f4cb6a840904be738debf73bd2eceff31880617427280e43a7471de676fe81d
moderate 50ms baseline manifest 107ee7485d0ed90013d5347a615d59d597bc4cec6eb0af70d175af2de691c423
moderate 50ms baseline assessment dc36b92a716e14c68f69d45185b9d4bbbbef49ecbd7518f22771b466ec62f690
moderate 50ms rate27 manifest 72dde4008c0c23f573d0e2f934a7476986fb738256f40b9669b6cfbd2ab9d98a
moderate 50ms rate27 assessment 0afe71cadcd59aa9db3cc9c6f0ab96126d2a9077bb58e7c1c2d637a79d21a314
high 25ms baseline manifest 8979dd97d9bbbe95a27c1aa293f4dbac45ab311ace5b7350c0c2e58596769b3d
high 25ms baseline assessment e08394282e59b719275b4cb8b34abfa2b3d3be29a0fe5b36c39d12aa4aa57c25
high 25ms rate27 manifest 9b3b5bc8f4468570cbd894ae60d3c9c88bfa3dae076fca6975ec8da925809640
high 25ms rate27 assessment 8750957a849cd7514ab88add55c82b983a59fc4f5b81ed8f06ed7f431cc55f75
high 50ms baseline manifest 4c5539ff25614641831adaca37bdecc8d916d8a044f684aeed30ad4be5b50446
high 50ms baseline assessment 53d4feee5d612ec98974dc1f0b5e622a6a91a1021ec1987e26e992c49ba79ad3
high 50ms rate27 manifest b9e8d37afb7817cb267bb665ac572423aaa0fc4ace173bc3bb5925fda9eab761
high 50ms rate27 assessment fa26893f3971a93f886d328058bbf41ab9da3aab5c829133b64d11f8fabe16c9
moderate 25ms rate-delay comparison 62e140518ed48cd3bdd61a8cad4ea8a2458da9a5a6ba3e0a15a29ef445801eb2
moderate 50ms rate-delay comparison 88d232ae2777b621ea3dc953b33489f231066ac1a049574acf4cebc861eed823
high 25ms rate-delay comparison 40be27d7aeb1b786f69f63db1a3758373f061aac16beee6d1082d2f7fd7e9757
high 50ms rate-delay comparison 0026c77a2c61961b9a865d77700ff19342975fee2ea53f07835f45d2c3c1e4c0
moderate cadence-baseline comparison a4e53e45e198f82e21ca35a6e6db630dde7f076fd730e4495654d98128094190
moderate cadence-rate27 comparison 27eeba806ffea43d58a7a2524c7056144391e7a5b0f2ce31f4b46f2d899654e9
high cadence-baseline comparison 027ad52fb686808189c93b14fe05794e76868ba43febe225877793dff398e4f4
high cadence-rate27 comparison 9368d82e1dd9679d656041973d6905eb2bb69ee2bb612d5050635dbae05c792d
moderate-25ms-baseline correlated-residual model 3cd282814f594d76b8fc9b198bf86c6a6c650563c6ef3d17f80f61f7c47b3e4a
moderate-50ms-baseline correlated-residual model a582f8ba1d89c4e50330d02d6f33f841b4a02d631107177731d4ac75676ae202
high-25ms-baseline correlated-residual model 1538979197ae683789f569c0b90a1b6f93da8de590fec902066a035c40910bc9
high-50ms-baseline correlated-residual model 6597816655473de422afd8dbd0dab6979ee4e3eda33f2d265b2260270612d691
```

All eight assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Piecewise-linear sampled-rate convention sweep

On 2026-10-31, the new `sampled-rate-piecewise-linear` motion profile was checked
against the full quaternion/rate convention matrix. It draws seeded three-axis
normal rate knots with 0.45 rad/s standard deviation every 0.5 s, interpolates rates
linearly, and integrates attitudes on the independent fixed 2.5 ms grid. Three
motion seeds (731401--731403) were each crossed with all 192 truth conventions at
161 samples and 25 ms cadence. The low-noise profile used 0.0015 rad orientation
noise and 0.012 rad/s rate noise, with a paired noise seed for each motion.

The 576 noisy cases produced 384 correct recommendations and 192 abstentions. The
abstentions all had multiple evidence contenders. Every data winner matched truth,
truth remained in the evidence set, and no recommendation was incorrect. Coverage
was entirely motion-dependent in this small cohort:

| Motion seed | Recommendations / 192 | Abstentions / 192 |
|---:|---:|---:|
| 731401 | 0 | 192 |
| 731402 | 192 | 0 |
| 731403 | 192 | 0 |

Each generated corpus also includes three clean controls; the table and totals above
exclude those controls. This is a single seeded noise realization per motion and a
synthetic convention sweep, not a performance rate or evidence of calibrated
confidence.

### Piecewise-linear sampled-rate OU cadence follow-up

The balanced design above was repeated for the new rate family with motion seeds
731401--731403, 20 shared noise seeds (20311121--20311140), all three noise levels,
three orientation/rate OU time-constant pairs, and 10/40/100 Hz sampling. Each
recording lasted four seconds. As in the Fourier-family study, the 1 ms OU process
was paired at common timestamps across cadence; each time-constant/cadence/noise
cell contains 60 cases. Table entries are correct recommendations out of 60; the
remaining cases abstained.

| Orientation/rate tau | Cadence | Low noise | Moderate noise | High noise |
|---:|---:|---:|---:|---:|
| 12.5 / 25 ms | 10 Hz | 60 | 60 | 40 |
| 12.5 / 25 ms | 40 Hz | 40 | 20 | 0 |
| 12.5 / 25 ms | 100 Hz | 20 | 0 | 0 |
| 50 / 100 ms | 10 Hz | 60 | 59 | 39 |
| 50 / 100 ms | 40 Hz | 60 | 34 | 5 |
| 50 / 100 ms | 100 Hz | 40 | 20 | 0 |
| 200 / 400 ms | 10 Hz | 60 | 54 | 30 |
| 200 / 400 ms | 40 Hz | 60 | 40 | 20 |
| 200 / 400 ms | 100 Hz | 50 | 25 | 0 |

Across the nine cells, low noise yielded 450/540 recommendations, moderate noise
312/540, and high noise 134/540. In total, 896/1,620 cases were correctly
recommended and 724 abstained; every abstention cited `multiple_evidence_contenders`.
There were no data-winner misses, evidence misses, or incorrect recommendations.
Recommendation coverage by motion seed again varied substantially:

| Motion seed | Low recommendations / 180 | Moderate / 180 | High / 180 | Total / 540 |
|---:|---:|---:|---:|---:|
| 731401 | 110 | 53 | 0 | 163 |
| 731402 | 180 | 160 | 85 | 425 |
| 731403 | 160 | 99 | 49 | 308 |

The paired design isolates sensitivity within this generator, but the three motions
are not independent motion families. The results are conditional on this process,
one truth convention, four-second duration, and tested grid. They are not field
accuracy, false-positive guarantees, or calibration evidence and do not justify
threshold changes. The piecewise-rate signal also explicitly supports only
`[-60, 3600]` seconds and rejects out-of-range queries.

The assessments ran on Windows 11/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, and
package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`. The inference
implementation SHA-256 was
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`. At generation
time, `scripts/generate_validation_corpus.py` had SHA-256
`cbf37a8207e96f3fb42302ed01bb7ce6a487a827f312b817757e1b8e2559030e`,
`scripts/validation_noise.py` had SHA-256
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`, and
`scripts/validate_corpus.py` had SHA-256
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`.

Reproduce the three-motion full convention sweep with:

```bash
python -m scripts.generate_validation_corpus \
  --outdir .validation-corpus-piecewise-convention-731401-403 \
  --noise-profile low --noise-replicates 1 --noise-seed-start 20311101 \
  --motion-profile sampled-rate-piecewise-linear \
  --motion-seeds 731401 731402 731403 --noise-sample-counts 161 \
  --sample-period-s 0.025 --truth-convention-sweep
python -m scripts.validate_corpus \
  .validation-corpus-piecewise-convention-731401-403/manifest.json \
  --checkpoint .validation-corpus-piecewise-convention-731401-403/assessment.checkpoint.json \
  --checkpoint-every 50 --progress \
  --output .validation-corpus-piecewise-convention-731401-403/assessment.json
```

Reproduce the balanced OU matrix with:

```bash
for profile in fast reference slow; do
  case "$profile" in
    fast) orientation_tau=0.0125; rate_tau=0.025 ;;
    reference) orientation_tau=0.05; rate_tau=0.1 ;;
    slow) orientation_tau=0.2; rate_tau=0.4 ;;
  esac
  for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
    read -r name count period <<< "$cadence"
    outdir=".validation-corpus-piecewise-ou-balanced-731401-403-${profile}-${name}"
    python -m scripts.generate_validation_corpus \
      --outdir "$outdir" --noise-profile all --noise-replicates 20 \
      --noise-seed-start 20311121 \
      --motion-profile sampled-rate-piecewise-linear \
      --motion-seeds 731401 731402 731403 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --orientation-noise-correlation-time-s "$orientation_tau" \
      --rate-noise-correlation-time-s "$rate_tau"
    python -m scripts.validate_corpus "$outdir/manifest.json" \
      --checkpoint "$outdir/assessment.checkpoint.json" \
      --checkpoint-every 50 --progress --output "$outdir/assessment.json"
  done
done
```

Both experiments' generated corpus files remain ignored local artifacts; the source
and result summary here is the reproducible record.

### Smooth maneuver-burst sampled-rate convention sweep

The `sampled-rate-maneuver-bursts` profile adds a third non-Euler rate family. It
uses seeded, smooth `sin²` pulses on an eight-second cycle, with eight one-second
slots, independent per-axis start/duration/amplitude values, and zero-rate gaps on
each axis. Each axis has a small seeded timing offset from the others. Pulse
amplitudes are drawn with a 0.3 rad/s standard deviation. The trajectory is still
integrated on the independent 2.5 ms fixed grid; the signal passes the same
multi-seed DOP853 convergence and query-independence checks as the other sampled
rate families.

The first three-motion convention sweep used four-second records (161 samples at
25 ms) and the low-noise profile. A follow-up used eight-second records (321
samples) to include the full pulse cycle. In both sweeps, all 576 noisy cases
abstained, while the true class remained the operational winner and in the evidence
set. There were no evidence misses, data-winner misses, operational misses, or
incorrect recommendations. Coverage was zero for each of the three motion seeds in
both record lengths. The longer record did not improve the current resolver's
recommendation coverage under this noise profile.

To separate noise sensitivity from convention-label or signal-generation errors,
the same three-motion, four-second convention sweep was repeated with the new
`--noise-profile none` option. All 576 convention cases resolved correctly;
there were no abstentions or misses. This is evidence that the synthetic labels and
signal are distinguishable in the exact-data limit, but it exposes a pronounced
noise-to-confidence gap: these results do not establish recovery under realistic
measurement noise, calibrated confidence, or field performance. They do not
justify changing decision thresholds. The separate three clean analytic controls
in each corpus are excluded from the counts below.

| Design | Motion seeds | Samples | Correct recommendations / 576 | Abstentions | Any winner/evidence/operational misses |
|---|---|---:|---:|---:|---:|
| Low noise, 4 s | 732501--732503 | 161 | 0 | 576 | 0 |
| Low noise, 8 s | 732601--732603 | 321 | 0 | 576 | 0 |
| No noise, 4 s | 732701--732703 | 161 | 576 | 0 | 0 |

These are synthetic convention sweeps, not pooled performance rates. The zero-noise
and low-noise results should remain separate; do not average them into one accuracy
figure. A one-convention OU/cadence screen across three cadences and three
real-time correlation-time pairs is recorded below. Extending that screen across
truth conventions and timing artifacts remains open; the paired noise ladder below
is a separate near-tie diagnosis, not enough evidence to tune decision thresholds.

Reproduce all three sweeps with:

```bash
set -e

run_convention_sweep() {
  name="$1"; noise_profile="$2"; noise_seed_start="$3"; sample_count="$4"
  shift 4
  outdir=".validation-corpus-maneuver-${name}"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise_profile" \
    --noise-replicates 1 --noise-seed-start "$noise_seed_start" \
    --motion-profile sampled-rate-maneuver-bursts --motion-seeds "$@" \
    --noise-sample-counts "$sample_count" --sample-period-s 0.025 \
    --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 50 --progress --output "$outdir/assessment.json"
}

run_convention_sweep 4s-low-732501-503 low 20311301 161 732501 732502 732503
run_convention_sweep 8s-low-732601-603 low 20311401 321 732601 732602 732603
run_convention_sweep 4s-none-732701-703 none 20311501 161 732701 732702 732703
```

The generated corpora and assessments are ignored local artifacts; preserve
existing directories and use new names for reruns.

These assessments ran on Windows 11/Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
and package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261031`. The inference
implementation SHA-256 was
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`. At the
current source revision, `scripts/generate_validation_corpus.py` has SHA-256
`ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa`,
`scripts/validation_noise.py` has SHA-256
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`, and
`scripts/validate_corpus.py` has SHA-256
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`.

### Burst sampled-rate OU and cadence screen

The burst family also has a balanced, one-truth-convention OU screen crossing
three motion seeds (732401--732403), 20 paired noise seeds (20311201--20311220),
three noise levels, three real-time OU correlation-time pairs, and 10/40/100 Hz
cadences. Each OU/cadence cell contains 180 noisy cases (60 at each noise level).
The labeled convention was scalar-last, body-to-world quaternion orientation and
body-frame `pqr` rates in radians per second with positive sign. All records cover
four seconds, with 41, 161, or 401 samples at 10, 40, or 100 Hz respectively.
The three measurement-noise standard-deviation pairs were `(0.0015 rad,
0.012 rad/s)`, `(0.003 rad, 0.025 rad/s)`, and `(0.006 rad, 0.05 rad/s)`.

| OU profile | Orientation/rate time constants (s) | Cadence | Data/operational misses / 180 | Evidence-insufficient / 180 | Recommendations / 180 |
|---|---:|---:|---:|---:|---:|
| Fast | 0.0125 / 0.025 | 10 Hz | 3 | 72 | 0 |
| Fast | 0.0125 / 0.025 | 40 Hz | 3 | 120 | 0 |
| Fast | 0.0125 / 0.025 | 100 Hz | 5 | 136 | 0 |
| Reference | 0.05 / 0.1 | 10 Hz | 6 | 68 | 0 |
| Reference | 0.05 / 0.1 | 40 Hz | 7 | 78 | 0 |
| Reference | 0.05 / 0.1 | 100 Hz | 9 | 82 | 0 |
| Slow | 0.2 / 0.4 | 10 Hz | 13 | 39 | 0 |
| Slow | 0.2 / 0.4 | 40 Hz | 8 | 30 | 0 |
| Slow | 0.2 / 0.4 | 100 Hz | 10 | 24 | 0 |

Across the 1,620 cases, there were 64 data-winner/operational misses and no
evidence-set misses, false recommendations, or recommendations at all. All cases
abstained. There were 649 evidence-insufficient cases and 971 abstentions caused
by multiple evidence contenders. Seven cases passed the stable-evidence check even
though the raw data winner was wrong; each still retained truth among multiple
contenders and abstained. Data-winner misses were 0/540 at low noise, 6/540 at
moderate noise, and 58/540 at high noise. This is a useful calibration warning:
stable evidence sufficiency alone did not make the leading raw class correct.

Aggregated over the three OU time-constant pairs, the cadence/noise breakdown was:

| Cadence | Data-winner misses, low / moderate / high (each / 180) | Evidence-insufficient, low / moderate / high (each / 180) |
|---:|---:|---:|
| 10 Hz | 0 / 2 / 20 | 0 / 29 / 150 |
| 40 Hz | 0 / 1 / 17 | 0 / 81 / 147 |
| 100 Hz | 0 / 3 / 21 | 16 / 85 / 141 |

These results are synthetic and conditional on one convention, one burst generator,
three motion seeds, and the tested OU/cadence grid. They support retaining the
current abstention behavior; zero coverage is not successful recovery, and the
64 raw-winner misses require broader convention-specific checks. No threshold was
changed. Timestamp jitter, dropout, channel latency, and external logs are not
covered by this screen.

Reproduce with fresh output directories (the existing corpus/report directories
are ignored local artifacts; choose new suffixes for reruns):

```bash
set -e

run_burst_ou_screen() {
  profile="$1"; orientation_tau="$2"; rate_tau="$3"
  for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
    read -r cadence_name sample_count sample_period <<< "$cadence"
    outdir=".validation-corpus-maneuver-ou-balanced-732401-403-${profile}-${cadence_name}-repro"
    python -m scripts.generate_validation_corpus \
      --outdir "$outdir" --noise-profile all --noise-replicates 20 \
      --noise-seed-start 20311201 \
      --motion-profile sampled-rate-maneuver-bursts \
      --motion-seeds 732401 732402 732403 \
      --noise-sample-counts "$sample_count" --sample-period-s "$sample_period" \
      --orientation-noise-correlation-time-s "$orientation_tau" \
      --rate-noise-correlation-time-s "$rate_tau"
    python -m scripts.validate_corpus "$outdir/manifest.json" \
      --checkpoint "$outdir/assessment.checkpoint.json" \
      --checkpoint-every 50 --progress --output "$outdir/assessment.json"
  done
}

run_burst_ou_screen fast 0.0125 0.025
run_burst_ou_screen reference 0.05 0.1
run_burst_ou_screen slow 0.2 0.4
```

The nine assessments used the same package-plus-validator implementation SHA-256
as the burst convention sweep above. Manifest and assessment SHA-256 values:

```text
# Each row: profile-cadence, manifest SHA-256, assessment SHA-256
fast-10hz      b5896874c7ef245f24dcf11ce7ab9ee07b55431ef4f69ef6e5394c436b36e9db 0ba0229a0ff77b5c26ccbe6dc94d48dc6c9e9c6eb8c30a229f2132b009155104
fast-40hz      c35e09f554355fa7f793f3c5e6332c6038547363a2c270ce7a0f843c07b9f7b0 c0707edb474c736e28d6c739db28018f3657b87d7ac29a1448541fba852dc7d0
fast-100hz     009cc2a7c7d3143b97484cd710a74b0a1ed290d7ac671cb00175d5c35e8398a3 a33fa9ac1bdfb58c460962149b3fb2d6d7cd10cf598204d7835bde0406eaeef7
reference-10hz 9f422c4ebd0878d7875aebdd3e096dc9e559290303b49cdf0e406e5e9542adc9 58fad21c12674e66048d051c04d3d3efcf89e81c876fb6c13dd7a5c3790820db
reference-40hz 9b241b1bbcafe1d872eb7db8ffac226c5a3586a267bdaa8f15f10fdcbc307675 cc80ee09a1ff2dfc4742145c5597c9e2a911d9216c0ce2084b53fbfc63c2d18a
reference-100hz 21a40f379dd08f791924f3c0930f1bbce611273299e368d317dfe1ac5fb8a87b c8cf5d7426fd3139b524cf35897e8212c1323accee221c1315fc2003e87d9f80
slow-10hz     b55d5a37ff448270a65e2e9c35f9e6094a0ffd83ba4ec35d186e524c82d6d878 f0346c0cd52ea1dc7d70d565aaddcbe08bb51eeb01794bac048bb012de9454f1
slow-40hz     de54b8b19c60a463de0b35eaff47c1c30177ece43cc443dfa2666ecbfe9ff6c5 28b33fff34447d0b76c22389ee12e9017bd4398f37ea4ed0de33d7cfa0bad9d0
slow-100hz    4f71a02e04081b3b9cb7fb549a28f1d7798fcbb73fb587bfa36a397c67cbd544 e5f5dc7da4d474ddf148f094cd61d61ce3bf1561b482d8d14078151d0d7b743c
```

### Full-convention burst OU cadence screen

The reference OU condition was extended across all 192 truth convention encodings
at each of 10, 40, and 100 Hz. Each cadence crossed those conventions with motion
seeds 732401--732403 and the low, moderate, and high noise profiles, using one OU
realization per convention/motion/noise cell (1,728 noisy cases per cadence;
5,184 overall). The records span four seconds with 41, 161, or 401 samples. The
orientation/rate OU correlation times were 0.05/0.1 s, with noise seed 20312001.
The three clean analytic controls in each assessment are excluded from the
outcomes below. All cadence reports used the same package-plus-validator
implementation SHA-256 as the preceding burst studies.

| Cadence | Noise level | Correct recommendations / 576 | Raw-winner misses / 576 | Evidence-insufficient / 576 |
|---:|---|---:|---:|---:|
| 10 Hz | Low | 192 | 0 | 96 |
| 10 Hz | Moderate | 0 | 96 | 192 |
| 10 Hz | High | 0 | 96 | 384 |
| 40 Hz | Low | 0 | 0 | 96 |
| 40 Hz | Moderate | 0 | 0 | 192 |
| 40 Hz | High | 0 | 96 | 384 |
| 100 Hz | Low | 0 | 0 | 96 |
| 100 Hz | Moderate | 0 | 0 | 192 |
| 100 Hz | High | 0 | 96 | 384 |

There were no false recommendations or evidence-set misses at any cadence; every
recommendation was correct. The 192 low-noise recommendations at 10 Hz all came
from motion seed 732403. At moderate noise, 10 Hz had 96 raw-winner misses (all
from seed 732402), while the same matched cases had correct raw winners at 40 and
100 Hz. At high noise, all three cadences had the same 96 raw-winner misses, again
on seed 732402. The 40 and 100 Hz outcomes were identical throughout this screen.
This cadence dependence and concentration by motion seed argue against reading
the counts as general accuracy rates. The screen uses one noise realization per
cell and omits jitter, dropout, latency, and external recordings; it does not
justify a threshold change.

Reproduce with fresh output directories (the generated corpora and reports are
ignored local artifacts; use new names if those directories already exist):

```bash
set -e

for cadence in "10hz 41 0.1" "40hz 161 0.025" "100hz 401 0.01"; do
  read -r name sample_count sample_period <<< "$cadence"
  outdir=".validation-corpus-maneuver-ou-convention-cadence-reference-${name}-732401-403"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile all --noise-replicates 1 \
    --noise-seed-start 20312001 \
    --motion-profile sampled-rate-maneuver-bursts \
    --motion-seeds 732401 732402 732403 \
    --noise-sample-counts "$sample_count" --sample-period-s "$sample_period" \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment-cadence-conventions.checkpoint.json" \
    --checkpoint-every 100 --progress \
    --output "$outdir/assessment-cadence-conventions.json"
done
```

Manifest and assessment SHA-256 values:

```text
# Each row: cadence, manifest SHA-256, assessment SHA-256
10hz  111524e0426dcb6124b041b6ee0eea80d719f7de191f373dde081ab0c170f895 2f9471085ce636d41077f8156c72f91d357205638e64722e9de6c3c779008d05
40hz  214f571add21c6a50331fabd3c28d47be1d4853822994bc30bc596a326ec85d9 d27d56731ef043f1f853e66c8ebf0804d47c45dbacec316cf43a92b79328350e
100hz 83aee1ca30d341781da22f227d655e8f0df0ec413c4b2b632a9aea6618d7b639 eb35a50c1098c90d8df5797dfc6649c31b469f553fb0b3288a73c072db0fbd81
```

### Burst timing-artifact convention screen

The same reference OU condition was crossed with timing perturbations at 40 Hz
(161 scheduled samples over four seconds), covering all 192 truth conventions,
motion seeds 732401--732403, and low/moderate/high noise. Orientation/rate OU time
constants were 0.05/0.1 s. The jitter and dropout matrices each used noise seed
20313001; the latency matrix used the same seed and included all four combinations
of orientation delays 0/13 ms and rate delays 0/27 ms. Each cell therefore contains
576 noisy cases, with one realization per truth/motion/noise cell. The three clean
analytic controls per report are excluded below. Across the three reports there
were 15,552 noisy cases; all retained truth in the evidence set, with no false
recommendations.

For timestamp jitter, 0 and 3 ms maximum absolute jitter produced identical public
decision outcomes and reasons in all 1,728 matched pairs. There were no noisy
recommendations at either level. The raw-winner misses and evidence-insufficient
counts by noise level were also identical:

| Noise | Correct recommendations, 0 -> 3 ms / 576 | Raw-winner misses, 0 -> 3 ms / 576 | Evidence-insufficient, 0 -> 3 ms / 576 | Status/reason changes |
|---|---:|---:|---:|---:|
| Low | 0 -> 0 | 0 -> 0 | 96 -> 96 | 0/576 |
| Moderate | 0 -> 0 | 0 -> 0 | 192 -> 192 | 0/576 |
| High | 0 -> 0 | 96 -> 96 | 384 -> 384 | 0/576 |

The evidence-contender membership was unchanged by jitter, though its ordering
changed in 432/1,728 matched cases. Thus this small timing perturbation did not
change the final recommendation or abstention, but it was not bit-for-bit inert in
the ranked evidence report.

For whole-row dropout, the table reports each requested probability over 576 cases
per noise level. The mean retained sample counts were 161, 146, and 107 at 0%, 10%,
and 35% dropout respectively; duration endpoints are protected by the generator.

| Noise | Dropout | Correct recommendations / 576 | Raw-winner misses / 576 | Evidence-insufficient / 576 |
|---|---:|---:|---:|---:|
| Low | 0% | 0 | 0 | 96 |
| Low | 10% | 0 | 0 | 0 |
| Low | 35% | 192 | 0 | 0 |
| Moderate | 0% | 0 | 0 | 192 |
| Moderate | 10% | 0 | 0 | 192 |
| Moderate | 35% | 0 | 0 | 192 |
| High | 0% | 0 | 96 | 384 |
| High | 10% | 0 | 96 | 384 |
| High | 35% | 0 | 96 | 384 |

The 192 low-noise recommendations at 35% dropout were all correct and all came
from motion seed 732403. All 96 high-noise raw-winner misses remained on seed
732402 at every dropout level. No evidence misses or false recommendations
occurred. Dropout often narrowed evidence-contender sets, so this apparent coverage
increase is a sensitivity result, not evidence that missing samples improve real
systems.

For channel latency, the four combinations (0/0, 13/0, 0/27, and 13/27 ms for
orientation/rate) left recommendation status, raw-winner correctness, and decision
reason unchanged from the paired zero-delay cases. Every noisy case abstained; the
96 high-noise raw-winner misses and all evidence-insufficient counts were identical
to the zero-delay baseline. Evidence-contender sets nevertheless changed in 672/1,728
orientation-only pairs, 864/1,728 rate-only pairs, and 504/1,728 pairs with both
delays; the sets generally expanded, while truth remained present. Timing offsets
therefore affected internal ambiguity even where the public decision stayed
conservative.

The 35% dropout result was repeated at low noise with three fresh paired
noise/dropout seeds (20313101--20313103), again across the full convention matrix
and the same three motion seeds. The 0%-dropout baseline made no recommendations.
At 35%, 480/1,728 cases received correct recommendations: 192, 96, and 192 for the
three seeds respectively. Every recommendation came from motion seed 732403; the
other two motion seeds had none. The realized retained sample counts were 106, 108,
and 101 of 161 for the three dropout masks. There were no data-winner, operational,
or evidence misses and no false recommendations. This replication confirms a
repeatable effect for this particular burst seed under the tested noise model, not
a general benefit of dropout. It remains one motion family, three motion seeds,
and one dropout level; do not tune thresholds from it.

#### Fresh burst-seed replication

A second matched block used fresh motion seeds 732404--732406, the same 40 Hz
sampling and OU time constants, all 192 truth conventions, all three noise levels,
and three paired noise realizations (seeds 20313201--20313203). Each condition
contains 5,184 noisy cases plus the three clean analytic controls. The clean
baseline had no noisy recommendations or winner/evidence misses; 576 high-noise
cases were evidence-insufficient, split as 288 on each of seeds 732404 and 732406
and none on 732405.

The matched 3 ms jitter condition changed no public decision status or reason, raw
winner correctness, recommendation, or evidence-sufficiency result. Evidence
contender membership expanded in 216/5,184 cases (96 moderate-noise and 120
high-noise); it never contracted. This reproduces the earlier pattern that small
jitter affects internal ambiguity more often than the public outcome.

At 35% whole-row dropout, 576 low-noise cases received correct recommendations,
all on motion seed 732405; the other two fresh motions still abstained at low
noise. The clean baseline had no noisy recommendations. There were no false
recommendations, winner misses, or evidence misses. At high noise, evidence-
insufficient cases fell from 576 to 288 (96 fewer on seed 732404 and 192 fewer on
732406), without changing winner correctness or producing recommendations. The
mean retained sample count was 108/161. As in the earlier cohort, contender sets
generally narrowed under dropout. The coverage change now appears on a second
fresh motion seed, but remains seed- and noise-dependent within one synthetic
motion family; it is not evidence that dropout improves real measurements and is
not a basis for threshold tuning. The 13/27 ms latency assessment on these same
fresh motions left decision status/reason, recommendations, raw and operational
class selection, and evidence sufficiency unchanged. Evidence-contender membership
changed in 816/5,184 noisy cases: 792 expanded and 24 contracted, all at moderate
or high noise (312 expansions at moderate noise; 480 expansions and 24 contractions
at high noise). There were no evidence misses. Across these new seeds, timing
perturbations altered ambiguity more often than the conservative public outcome.

Reproduce the timing matrices with fresh output directories (the existing corpus
directories are ignored local artifacts):

```bash
set -e

run_burst_timing_matrix() {
  name="$1"; shift
  outdir=".validation-corpus-maneuver-ou-timing-${name}-732401-403"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile all --noise-replicates 1 \
    --noise-seed-start 20313001 \
    --motion-profile sampled-rate-maneuver-bursts \
    --motion-seeds 732401 732402 732403 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

run_burst_timing_matrix jitter --timestamp-jitter-amplitudes-s 0 0.003
run_burst_timing_matrix dropout --dropout-probabilities 0 0.1 0.35
run_burst_timing_matrix latency \
  --orientation-latencies-s 0 0.013 --rate-latencies-s 0 0.027
```

Reproduce the three-seed dropout follow-up with:

```bash
outdir=.validation-corpus-maneuver-ou-dropout35-low-replicates-732401-403-seeds20313101-103
python -m scripts.generate_validation_corpus \
  --outdir "$outdir" --noise-profile low --noise-replicates 3 \
  --noise-seed-start 20313101 \
  --motion-profile sampled-rate-maneuver-bursts \
  --motion-seeds 732401 732402 732403 --noise-sample-counts 161 \
  --sample-period-s 0.025 \
  --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1 \
  --dropout-probabilities 0 0.35 --truth-convention-sweep
python -m scripts.validate_corpus "$outdir/manifest.json" \
  --checkpoint "$outdir/assessment.checkpoint.json" \
  --checkpoint-every 100 --progress --output "$outdir/assessment.json"
```

Reproduce the fresh-motion-seed timing conditions with separate ignored output
directories:

```bash
set -e

run_fresh_burst_timing_case() {
  name="$1"; shift
  outdir=".validation-corpus-maneuver-ou-new-motions-${name}-732404-406"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile all --noise-replicates 3 \
    --noise-seed-start 20313201 \
    --motion-profile sampled-rate-maneuver-bursts \
    --motion-seeds 732404 732405 732406 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

run_fresh_burst_timing_case baseline
run_fresh_burst_timing_case jitter --timestamp-jitter-amplitudes-s 0.003
run_fresh_burst_timing_case dropout --dropout-probabilities 0.35
run_fresh_burst_timing_case latency \
  --orientation-latencies-s 0.013 --rate-latencies-s 0.027
```

Manifest and assessment SHA-256 values:

```text
# Each row: experiment, manifest SHA-256, assessment SHA-256
jitter        f398261116afa77bef8321a5d928699f06de82ac8ca14175e01877f27fbfac14 e1aadb3b5fa4da6ec046a50598397065caa46a33a0b3aeab78ef11c9bb6be879
dropout       3318f42a2124f54e5f63b40761f47848c2decbc02740d3f81015b5dbd703b30f 0d193777f9faba573a90c2936a863fdae5e5a59422e63f4218eee8269a88f3da
latency       326763dc1f05742912d755136a698c7c80376e46ce710269004780d8485dc207 196974c7b82461c66a962bfe2cedbd7dfc24dd386dbcffce7e227670db8f2704
dropout-rep    6ca98e5a7bf7517697074a6276218e2b17963eb719deb72ace0a7e2ce6c8a460 8628dd8c4900743e7d3766e957f610a95cc03292a35bb334d2045c58e4d63f1c
```

All four assessments used the same package-plus-validator implementation
SHA-256, `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.
The generator source SHA-256 was
`ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa`, the OU
noise helper was
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`, and the
validator was
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`.

The fresh-seed experiment artifact hashes are:

```text
# Each row: experiment, manifest SHA-256, assessment SHA-256
baseline      0d24e3315d6ffe052bb6bca68fc64b196b58378cdce1adedbfeeec8813b8b180 701cf1124f73f8ba138f2a79fffcb86422e8de1da0deabfff18bc6748acd1c46
jitter        3c112ae9338b219ae984abe004aac112fbcbfeef2038715c2f53c18f5e4c83ad 9390ab7fa3e639fc6550e37938c5e8cb21ea1c538a4bdc74391d22de9cd2213f
dropout       1db72b606dd872be9e275e99da67b4e187ce3df0c4cebc00c381e70f40c79523 5cb58c3ab79954e67e72ce0f9b515732deddf73062b0285313af5a0e773c4ef7
latency       d5dd18818f0d14cd6772b009b5934b027db38edf07cf9578e6a54938d05256a4 52b6758738fd8cc89b9673ead25c57564acbfcb9d47dd1510d7cd76686f80c1d
```

All four fresh-seed assessments used the same package-plus-validator
implementation SHA-256, `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.
Generator, OU-noise helper, and validator source hashes matched those recorded
above.

#### Fresh burst-dropout replication

The earlier 35% whole-row-dropout screen produced low-noise recommendations on
one burst-motion seed. To test whether that response transferred, a new block
used motion seeds 732407--732409 and noise seeds 20313304--20313306. Each of the
low, moderate, and high OU-noise tiers had paired no-dropout and 35%-dropout
assessments, with 161 samples at 25 ms, 0.05/0.1 s orientation/rate correlation
times, three noise realizations per motion, and the full 192-convention truth
sweep. Each assessment contains 1,728 noisy cases plus three clean controls.
The dropout cases retained a mean of 107/161 rows (range 99--115); an exact
CSV-row audit confirmed every retained dropout row also appears unchanged in its
same-seed baseline.

The table summarizes the 1,728 paired noisy cases; the clean controls are
excluded.

| Noise | Correct recommendations, baseline -> dropout | Top-winner misses | Evidence-insufficient cases | Evidence-set contractions | Decision-status changes |
|---|---:|---:|---:|---:|---:|
| Low | 0 -> 0 | 0 -> 0 | 96 -> 0 | 336/1,728 | 0 |
| Moderate | 0 -> 0 | 96 -> 192 | 864 -> 384 | 888/1,728 | 0 |
| High | 0 -> 0 | 192 -> 192 | 1,248 -> 1,152 | 1,728/1,728 | 0 |

All 1,728 noisy cases abstained in both conditions at every noise tier; all
three clean controls per assessment recommended correctly. Truth remained in
the evidence set in every noisy case, with no false recommendations. Dropout
contracted evidence sets across all three fresh motion seeds, but this was not
a recovery benefit: at moderate noise, top-winner misses doubled on seed 732409
(96 -> 192) while decisions remained abstentions. At low noise, the earlier
recommendation gain did not recur at all. These results reinforce that evidence
size and top-winner accuracy are seed-dependent and that synthetic dropout
cannot justify changing thresholds.

Reproduce the paired block in a fresh ignored output directory:

```bash
set -e

root=".validation-corpus/burst-dropout-replication-732407-409-repro"

run_burst_dropout_case() {
  noise="$1"; condition="$2"; dropout="$3"
  outdir="$root/${noise}-${condition}"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313304 \
    --motion-profile sampled-rate-maneuver-bursts \
    --motion-seeds 732407 732408 732409 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 \
    --dropout-probabilities "$dropout" --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in low moderate high; do
  run_burst_dropout_case "$noise" baseline 0
  run_burst_dropout_case "$noise" dropout 0.35
  python -m scripts.compare_corpus_assessments \
    "$root/${noise}-baseline/assessment.json" \
    "$root/${noise}-dropout/assessment.json" \
    --output "$root/$noise/dropout-comparison.json"
done
```

The recorded artifact hashes are:

```text
# Each row: noise/condition, manifest SHA-256, assessment SHA-256
low baseline      bf24c9061b5e6c5a23e9fdb3c5261fe161298fd6fa5394e699dbfeb4e4b9b04c f39dccb1ed64558c99d9093bd69be0814d37ba7188f04c01198b746fdb8dbbe8
low dropout       a42fbaca5fe23cee2566385033a5b6b52a177be123b7614507d18ba1082cdbaf 5ca1d6cf5568648467caf88650a28dbf350ea3fe1b1d4608831cd6b92eb57c4e
moderate baseline 950c4ca004dc25bc13a27f7dc570f433f5b93cfb8b3b9eef7ccf312cd7f74373 de6f8f04dd8fd9d82e87680ba9d2885923986d0cfd3dfe0710a9ccaa8572c2ef
moderate dropout  242d350946d6ebe2da61a27249c1611e07e6e2c0971cc74ee201dc0dfab9483e a8734d2d5fe15e460efa4281189a1276dbce424208471fa8d7a3d7e12ebb3191
high baseline     d3fd73c0f77e9e5a4ded42dcb5a49734acabf690fa7fbe73c192d45a6f429fea 7acaad3ea3496abc0f08c334f64c8012a51ffbc34405d0b548b2c4749473b00d
high dropout      121a23e2697fb3d5a690db6a5ae2754430da3f46ca60a9a3ae58f6916e5ab9e6 d323a7d6417b1f996ef34441147d900f46e54ef50a4d270240866efa00c8b832
```

```text
# Each row: paired comparison report, SHA-256
low      0b07a50dbd37b7bf6fe8e99324404813c86684824909f0d062243c33fe0f9c3f
moderate b590578dd2c0f64d3f28e6b9c1e12a19b56725ae14f0d2a9d07c07ebf946ec82
high     3a021d9a981ba5a27ef847675ac05262fd85cfcc08fa382006bb9f9bb9bbb659
```

The generator, OU-noise helper, validator, and comparison-tool SHA-256 values
were respectively `ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa`,
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`,
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`, and
`48a406330f3a89d208ffe5b1e3c8c1ef84545e62ce2a58503e4ae8144552ae0a`. All six
assessments used the same package-plus-validator implementation hash,
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849` (Python
3.14.6, NumPy 2.5.2, SciPy 1.18.0, Windows 11). These are synthetic,
generator/evaluator-derived labels, not independent logger evidence.

### Cross-family dropout replication

To check whether the burst result transferred to other synthetic signals, the
low-noise 40 Hz OU condition (0.0015 rad orientation noise, 0.012 rad/s rate noise;
0.05/0.1 s correlation times) was repeated on three fresh seeds (742401--742403)
for the Fourier-rate and piecewise-linear-rate profiles. Each clean and 35%
whole-row-dropout condition used all 192 truth conventions and three paired noise
seeds (20313301--20313303), for 1,728 noisy cases plus three analytic controls per
assessment.

| Motion family | Correct recommendations, clean → 35% dropout | Changed motion seed | Winner misses, evidence misses, false recommendations |
|---|---:|---:|---:|
| Fourier-rate | 1,536/1,728 → 1,728/1,728 | 742402: 384 → 576 | 0 in both conditions |
| Piecewise-linear-rate | 1,728/1,728 → 1,728/1,728 | None | 0 in both conditions |

The Fourier seeds 742401 and 742403 already recommended correctly in all 576
cases each; dropout resolved the 192 ambiguous cases on seed 742402 without
changing winner correctness. The piecewise-linear family was already uniquely
resolved across all three seeds and was unchanged by dropout. The mean retained
sample count under 35% dropout was 103/161. Together with the burst results above,
this shows a repeatable, motion-dependent change in ambiguity on two synthetic
families, not a general benefit of dropout. These are generator/evaluator-derived
labels, not independent logger data; do not tune thresholds from this result.

Reproduce the cross-family low-noise assessments with fresh ignored output
directories:

```bash
set -e

run_cross_family_low_case() {
  family="$1"; profile="$2"; condition="$3"; shift 3
  outdir=".validation-corpus-cross-family-low-${condition}-${family}-742401-403"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile low --noise-replicates 3 \
    --noise-seed-start 20313301 --motion-profile "$profile" \
    --motion-seeds 742401 742402 742403 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

run_cross_family_low_case fourier sampled-rate-nonanalytic baseline
run_cross_family_low_case fourier sampled-rate-nonanalytic dropout35 \
  --dropout-probabilities 0.35
run_cross_family_low_case piecewise sampled-rate-piecewise-linear baseline
run_cross_family_low_case piecewise sampled-rate-piecewise-linear dropout35 \
  --dropout-probabilities 0.35
```

Cross-family artifact SHA-256 values:

```text
# Each row: experiment, manifest SHA-256, assessment SHA-256
fourier baseline    d02c01ea3c5ece92a87a3b1e605ef4561d4ebd1c76c40f48412bd56f20dfc6c9 c4c285a3612bb7e9670a37ba99e376185a6917d922ea3ba14dde5f647dd7aca5
fourier dropout35   3d08228369475f0259bee0363c5a10489059fdc2ef0d70f557f2048962dbd5b5 75b60aabeb40da338c4a3097945043edcde74f88c5b18b1ade641886fb8ef5a5
piecewise baseline  79ded860e793da99956c2b9583031083540927a967f1c43e6b0734777370793e fe70e36b39314f0aee5ef796941ad347576cfb185fab3f90103268678ecf7822
piecewise dropout35 23858748cdf4f341e79971949ea2fe476ec22bd90abd55cc43afa62741998061 b5f0b501bb1637d9c2063b6204ac163e7f7c1a48271dadbee0094566b9e4d442
```

All four cross-family assessments used the same package-plus-validator
implementation SHA-256, `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Cross-family jitter and latency replication

The same three fresh Fourier and piecewise-linear motion seeds were evaluated
with 3 ms timestamp jitter and with 13 ms orientation / 27 ms rate latency. These
low-noise assessments again paired 1,728 cases per condition to the clean reports
above. Jitter changed no decisions, class selections, evidence sufficiency, or
contender membership in either family. Channel latency expanded contenders on the
previously ambiguous motion seed 742402, leading to conservative abstention in
192 Fourier cases (384 → 192 recommendations) and 96 piecewise-linear cases
(576 → 480). Winner and operational correctness remained intact, evidence
sufficiency did not change, and there were no false recommendations or evidence
misses. The other two motion seeds were unchanged in both families.

Reproduce these timing conditions with the same paired seeds and ignored output
directories:

```bash
set -e

run_cross_family_timing_case() {
  family="$1"; profile="$2"; condition="$3"; shift 3
  outdir=".validation-corpus-cross-family-low-${condition}-${family}-742401-403"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile low --noise-replicates 3 \
    --noise-seed-start 20313301 --motion-profile "$profile" \
    --motion-seeds 742401 742402 742403 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

run_cross_family_timing_case fourier sampled-rate-nonanalytic jitter3ms \
  --timestamp-jitter-amplitudes-s 0.003
run_cross_family_timing_case fourier sampled-rate-nonanalytic latency13-27 \
  --orientation-latencies-s 0.013 --rate-latencies-s 0.027
run_cross_family_timing_case piecewise sampled-rate-piecewise-linear jitter3ms \
  --timestamp-jitter-amplitudes-s 0.003
run_cross_family_timing_case piecewise sampled-rate-piecewise-linear latency13-27 \
  --orientation-latencies-s 0.013 --rate-latencies-s 0.027
```

Cross-family timing artifact SHA-256 values:

```text
# Each row: experiment, manifest SHA-256, assessment SHA-256
fourier jitter3ms    e7b7134492ad50902b437216d73da8efbdd3a1493bec234a57ab283b9ce2a2c1 deb7f50dba36687369bc51df5e731b1dc3a280abc37e90b9ac5a494309c128b8
fourier latency13-27 47b57332e2d07c8ff45084b98b86116454988db6d77a4c1b1af0935e429a2d9d 7b58ca97b33cc2605f5377525e75017c10f4f0c3f81bf334a4a0f721bcbe9fc3
piecewise jitter3ms  87bc7746aa9d52fc8d8d00afa14bfbd5c636ffd45a68cebb31234a8d7def380f fa1b3899b3440fe985ab57f1466bf5a53bb01923371a4dcb4a897e510521bad9
piecewise latency13-27 91899d4d682a3f76ad03ac4a32173e82408674dae22f98ff30afeb2284b0f8fe 9fd902b6308f9eb3ab60fc46ef0d5ff8a9d4899209f459740c0663c94e1e27d1
```

All four timing assessments used the same package-plus-validator implementation
SHA-256, `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Moderate/high-noise latency holdout

The 13/27 ms delay was then tested at moderate and high noise on fresh motion
seeds 742404--742406, for both Fourier-rate and piecewise-linear-rate trajectories.
Each clean and delayed condition used all 192 conventions and three paired noise
seeds (20313401--20313403), giving 1,728 noisy cases plus three analytic controls
per assessment. The only condition difference within each pair was channel latency.

| Noise | Motion family | Correct recommendations, clean → latency | Public decision changes | Contender membership changes |
|---|---|---:|---:|---|
| Moderate | Fourier-rate | 1,728 → 1,728 | None | None |
| Moderate | Piecewise-linear-rate | 1,152 → 1,152 | None | 120 expanded on seed 742406 |
| High | Fourier-rate | 624 → 672 | 48 additional correct recommendations on seed 742406 | 144 contracted (48 on 742405; 96 on 742406) |
| High | Piecewise-linear-rate | 264 → 264 | None | 384 expanded on seed 742406 |

All 6,912 paired cases retained the correct data winner and operational class;
there were no evidence misses, evidence-insufficient cases, or false
recommendations. At high noise, latency reduced ambiguity for Fourier motion
seed 742406, increasing correct recommendations, while increasing ambiguity for
piecewise-linear seed 742406 without changing its abstentions. Moderate Fourier
was invariant; moderate piecewise latency expanded some contender sets in a cell
that already abstained. The response is therefore neither monotonic nor uniform
across motion families. These synthetic outcomes do not justify changing
thresholds or treating latency as beneficial; they show that timing metadata can
alter ambiguity and recommendation coverage in a motion-dependent way.

Reproduce the moderate/high-noise latency pairs with fresh ignored directories:

```bash
set -e

run_cross_family_latency_case() {
  noise="$1"; family="$2"; profile="$3"; condition="$4"; shift 4
  outdir=".validation-corpus-cross-family-${noise}-${condition}-${family}-742404-406"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313401 --motion-profile "$profile" \
    --motion-seeds 742404 742405 742406 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in moderate high; do
  run_cross_family_latency_case "$noise" fourier sampled-rate-nonanalytic baseline
  run_cross_family_latency_case "$noise" fourier sampled-rate-nonanalytic latency13-27 \
    --orientation-latencies-s 0.013 --rate-latencies-s 0.027
  run_cross_family_latency_case "$noise" piecewise sampled-rate-piecewise-linear baseline
  run_cross_family_latency_case "$noise" piecewise sampled-rate-piecewise-linear latency13-27 \
    --orientation-latencies-s 0.013 --rate-latencies-s 0.027
done
```

Moderate/high-noise artifact SHA-256 values:

```text
# Each row: experiment, manifest SHA-256, assessment SHA-256
moderate Fourier baseline  665607b329d1ac58be98144f0cf28b52b8f4f3df60bd8f03bd898d81c730249b 05106c6859dcfab757a4238c951cb90ecbd099a22882ed49274b120ea86430da
moderate Fourier latency   6c4f71f8bb5a11d1980f9fcbf1e621435d778e16efe2e859ecbc81457ec1ce69 5f96ec512189818ae1de3b2e2a1dd2b5105f9f562eaae21c91e1ef9c1811a369
moderate piecewise baseline 3dbcb0b7602910784c47ee5d8ab821d08e2ea60f9d07376e940254bd7d72184b 35b5491ec510d3be90d41bb289a30e30aad5738c8f48869093454a7cf50c5687
moderate piecewise latency 41aa964997d476fe383b2b1097285de29f5a1c9165d0b3aaa0852b29e54f02ad e8ee7a4325a7ffe65fdfa9a949f18427457b5f2e054f53d632b1af3b3ec1e4d5
high Fourier baseline      47598b13d8e89b8e88cc03e1908a9fab7957fa6b5777d6ea9161f98a382d8736 ea11a834adc58555bdb0eb71b46abbc03f42c8f2bf7d349a6324141db18ca5ad
high Fourier latency       0780b6dfef7be9089b15d2c83c83dd8d8c3e253f7a00dfa166feca5b8fdd57b5 4a9035b9df4150f01bd0ba29f9d764e792e92e80b6d63c6b00ae600dc481ce83
high piecewise baseline    0e4358d56c9ac831a6a6e810cd715ae327d09be68e9d3c825b7df41f4806a48c 675368d351aabfddcee3c25c88401930165535f79d11091283fcf47ba590d51b
high piecewise latency     e97acef62a8252539a3e765d208659531c6e07a81b926abb97e133b9d52f5591 aa97c9a6f6b7b93e927d1142c5d5a779366361765e68dfa2204ee9fd5a331878
```

All eight assessments used the same package-plus-validator implementation
SHA-256, `f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Isolated orientation and rate latency holdout

The same moderate/high-noise cohort was rerun with each timing channel delayed
independently: 13 ms orientation-only or 27 ms rate-only, with the other
channel at zero latency. The paired clean baseline, three motion seeds
(742404--742406), three noise seeds (20313401--20313403), 161 samples over
4 seconds, and all 192 conventions were held fixed. Each assessment contains
1,728 paired noisy cases plus three clean analytic controls, for 13,824 noisy
comparisons across the eight delayed assessments.

| Noise | Motion family | Delay | Correct recommendations, clean → delayed | Evidence-set changes | Decision changes |
|---|---|---|---:|---|---|
| Moderate | Fourier-rate | Orientation 13 ms | 1,728 → 1,728 | None | None |
| Moderate | Fourier-rate | Rate 27 ms | 1,728 → 1,728 | None | None |
| Moderate | Piecewise-linear-rate | Orientation 13 ms | 1,152 → 1,152 | 72 expanded on seed 742406 | None |
| Moderate | Piecewise-linear-rate | Rate 27 ms | 1,152 → 960 | 576 expanded (192 on 742405; 384 on 742406) | 192 correct recommendations became abstentions on 742405 |
| High | Fourier-rate | Orientation 13 ms | 624 → 672 | 144 contracted (48 on 742405; 96 on 742406) | 48 additional correct recommendations on 742406 |
| High | Fourier-rate | Rate 27 ms | 624 → 576 | 48 expanded on seed 742406 | 48 correct recommendations became abstentions on 742406 |
| High | Piecewise-linear-rate | Orientation 13 ms | 264 → 192 | 240 expanded (96 on 742404; 48 on 742405; 96 on 742406) | 72 correct recommendations became abstentions on 742404 |
| High | Piecewise-linear-rate | Rate 27 ms | 264 → 72 | 984 expanded (336 on 742404; 144 on 742405; 504 on 742406) | 192 correct recommendations became abstentions on 742404 |

All 13,824 cases retained the correct data winner and operational class, with
truth in the evidence set; there were no evidence-insufficient cases or false
recommendations. The rate-only and orientation-only effects are not
interchangeable: at high noise the two channels move Fourier recommendation
coverage in opposite directions, while both expand ambiguity for the
piecewise-linear family. These results are seed- and motion-dependent synthetic
evidence, not a threshold-tuning basis or a field-logger claim. The 3×3
motion/noise replication is still narrow, and the separately measured channel
effects should not be assumed to add linearly when both channels are delayed.

Reproduce the isolated-channel assessments against the paired clean baselines
from the preceding section using fresh ignored output directories:

```bash
set -e

run_cross_family_isolated_latency_case() {
  noise="$1"; family="$2"; profile="$3"; condition="$4"; shift 4
  outdir=".validation-corpus-cross-family-${noise}-${condition}-${family}-742404-406"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313401 --motion-profile "$profile" \
    --motion-seeds 742404 742405 742406 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in moderate high; do
  run_cross_family_isolated_latency_case "$noise" fourier sampled-rate-nonanalytic orientation13 \
    --orientation-latencies-s 0.013
  run_cross_family_isolated_latency_case "$noise" fourier sampled-rate-nonanalytic rate27 \
    --rate-latencies-s 0.027
  run_cross_family_isolated_latency_case "$noise" piecewise \
    sampled-rate-piecewise-linear orientation13 \
    --orientation-latencies-s 0.013
  run_cross_family_isolated_latency_case "$noise" piecewise \
    sampled-rate-piecewise-linear rate27 \
    --rate-latencies-s 0.027
done
```

Isolated-channel artifact SHA-256 values:

```text
# Each row: noise, motion family, delay, manifest SHA-256, assessment SHA-256
moderate Fourier orientation13   7daa246bb51bc43bfce82fd04b6f2ce79a97e5b2231231dc59f3058f28e2a0af 2c8e6861a0075b2052c4c7b7114afabe8a3b15694e16f2a5f35acbe7673d3ec2
moderate Fourier rate27          476f3b8d8682107027aaa48570f2929c9713ff079dc2b6d52f2cd0e904d754ca 48bc7cc3bca84edfe12a4f5fb81d4fd2f01185e7f1dee3268081f471af32e1c6
moderate piecewise orientation13 03c4fb608cd8b78fee5419a078385c118477750732d7a69959dec381de5d9e93 f446b448b066d08f2df39f31bb26968a21986ce3f5c27ea462f382b87b83a458
moderate piecewise rate27        352b28eb291690d58dc428e859e46c1e742168b11392458f98df9f4b641f81ca 22fab7895757e0cb10931cce0daa3be5c0415af57f15921243dccbf8d5e6a91f
high Fourier orientation13       e9304ab0f3716f56d244e54d8c863568d5d166f73cdb13cf21730861e1f39a53 0c1813a6a29a198afec51643456f2c2eeb19d8b5fb1a6e0d43dabff7a7ca9a4a
high Fourier rate27              91baa6f823ef9c9e5ff0b89e0d82ae84b3433657272d3137944a45844aaa6719 d389d4788bc23666b2ed3967f7a234157f651cebc2ca0dae5f71b92076b5b0a1
high piecewise orientation13     ae009d3692533e27349735b08076381da59d7b2410edf46ae52f3bb06edc850a 5914364c29d46a1bf33042cee64113a687d207efcf3483048e68c04bcba3752e
high piecewise rate27            bcc8dab73c4cecee0076985796606fa452605ea75d0c84294b4eced13e454684 ab1d8d4f749248e15bbe2a26614611557284994e08871d8615cfe86b84a4190d
```

All eight isolated-channel assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

Create a machine-readable paired summary for any baseline/condition pair with:

```bash
python -m scripts.compare_corpus_assessments \
  .validation-corpus-cross-family-moderate-baseline-fourier-742404-406/assessment.json \
  .validation-corpus-cross-family-moderate-orientation13-fourier-742404-406/assessment.json \
  --output .validation-corpus-cross-family-moderate-orientation13-fourier-742404-406/paired-comparison.json
```

The comparator requires identical case IDs, truth classes, representations,
label sources, and provenance. Its report records both runtimes, condition
fields that changed, overall and seeded-noisy outcome counts, winner/evidence/
decision changes, and per-motion-seed summaries. It validates serialized
`duration_s` values against sample count and period before loading them.

For nested-cadence comparisons, `scripts.compare_cadence_assessments` pairs
cases by their stable motion, noise, truth, and acquisition identity rather than
requiring equal sample-grid case IDs. It requires both manifests and assessments,
matching per-case inference configurations and runtime, equal-duration nested
OU grids, and zero timestamp jitter/dropout; cadence-derived sample counts,
periods, and OU AR(1) coefficients may differ. It also reads each paired noisy
CSV and verifies all common-time orientation/rate values agree within `2e-12`;
the report records the number of shared rows and maximum difference. It is
intentionally narrower than the exact-case comparator and should not be used
for jitter/dropout studies.

### Fresh-seed piecewise rate-latency replication

To test whether the piecewise 27 ms rate-delay response generalized beyond
seeds 742404--742406, the same full-convention comparison was repeated on fresh
motion seeds 742510--742512 and noise seeds 20313501--20313503. The profile,
25 ms sample period, 161 samples, noise model, and convention sweep were fixed;
each delayed corpus differed from its baseline only by the 27 ms rate latency.
Each assessment contains 1,728 noisy cases and three clean analytic controls.

| Noise | Correct recommendations, baseline → rate delay (noisy cases) | Evidence-set changes | Decision changes |
|---|---:|---|---|
| Moderate | 1,728 → 1,728 | None | None |
| High | 0 → 0 | 72 expanded on seed 742512 | None |

At moderate noise, all three fresh motion seeds had 576/576 correct
recommendations under both conditions, with no paired changes. This does not
reproduce the 192 recommendation-to-abstention transitions previously seen on
seed 742405. At high noise, all 1,728 cases abstained in both conditions; the
72 contender-set expansions on seed 742512 did not change any decisions. Across
the 3,456 baseline-delayed noisy pairs (6,912 individual case assessments), both
sides retained the correct data winner and operational class, kept truth in
evidence, and had sufficient evidence; there were no false recommendations.
The earlier recommendation losses are therefore not a stable response across
these fresh seeds. This remains a synthetic, three-seed
replication at one cadence, not a population estimate or threshold-tuning basis.

Reproduce and compare the fresh-seed baseline and rate-delay assessments:

```bash
set -e

run_piecewise_rate_latency_replication() {
  noise="$1"; condition="$2"; shift 2
  outdir=".validation-corpus-cross-family-${noise}-replication-${condition}-piecewise-742510-512"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313501 --motion-profile sampled-rate-piecewise-linear \
    --motion-seeds 742510 742511 742512 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in moderate high; do
  run_piecewise_rate_latency_replication "$noise" baseline
  run_piecewise_rate_latency_replication "$noise" rate27 --rate-latencies-s 0.027
done

for noise in moderate high; do
  baseline=".validation-corpus-cross-family-${noise}-replication-baseline-piecewise-742510-512"
  delayed=".validation-corpus-cross-family-${noise}-replication-rate27-piecewise-742510-512"
  python -m scripts.compare_corpus_assessments \
    "$baseline/assessment.json" "$delayed/assessment.json" \
    --output "$delayed/paired-comparison.json"
done
```

Fresh-seed replication artifact SHA-256 values:

```text
# Each line: noise, condition, artifact, SHA-256
moderate baseline manifest 92a1db5cfd1552eebb6b8276086907266d76c58ad3c90f81edb65eec7841b58b
moderate baseline assessment 829399e56e5e4577007d614bd604f576209ae5e7b98b1dcaccc94ae133ea9e7e
moderate rate27 manifest ffeccf558045f5da39adfd9e87f3405032d8d34de071685276d17f3dc35171e2
moderate rate27 assessment 39155663ef26e07bd0fb32910c2db90336b0653f1882e62b23aa649355cf686a
moderate rate27 paired comparison 787fae13cc1eb84a32fea8bc20251d3c65a167f2d2be414545efd9e0c765955f
high baseline manifest bcc89551c0a250fbf071c19f1372cc7a442e97076a4593f5b7ef85620d67908e
high baseline assessment 672e350b5ec397f545434081f2ec65aab85f51969afc73be9fc75a71b9ab014b
high rate27 manifest 3541c8d364c9a43cefa7e1a570205c5d6c5acf0d0898942699ba903a10a47c86
high rate27 assessment 195e907b6e9e00ed3768a9b1c5a8468d9a83d20c599d246e178a127680b98d41
high rate27 paired comparison 865373e1f01fa1fec8b64b4ee636a03b386adfca31b03f2ebf7203c8d56e50d1
```

All four fresh-seed assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### 50 ms cadence rate-latency replication

The rate-latency comparison was repeated at a 50 ms period with 81 samples,
the same three motion seeds (742520--742522), and new noise seeds
20313601--20313603. Each of the four moderate/high baseline and rate27
assessments contains 1,728 noisy cases and three clean controls. The paired
comparison isolates rate latency within each noise level: the delayed
condition differs only by `rate_latency_s=0.027`.

| Noise | Correct recommendations, baseline → rate delay (noisy cases) | Evidence-set changes | Decision changes |
|---|---:|---|---|
| Moderate | 1,728 → 1,728 | None | None |
| High | 1,152 → 1,152 | None | None |

At moderate noise, every motion seed had 576/576 correct recommendations in
both conditions. At high noise, seeds 742520 and 742522 each had 576/576
correct recommendations, while seed 742521 abstained on all 576 cases in both
conditions. Across the 3,456 baseline-delayed noisy pairs (6,912 individual
case assessments), the correct data winner and operational class were retained,
truth remained in the evidence set, and evidence was sufficient; there were no
false recommendations or paired changes in winners, evidence, or decisions.

The earlier, independently seeded 25 ms replication at these motion seeds had no
high-noise recommendations, whereas this 50 ms cohort had 1,152. That contrast
does not isolate cadence because its noise seeds differ. The matched-noise
baseline comparison below controls that source of variation. Both replications
remain synthetic, use only three motion seeds, and do not establish population
performance or justify threshold changes.

Reproduce the 50 ms baseline and rate-delay assessments:

```bash
set -e

run_piecewise_rate_latency_50ms() {
  noise="$1"; condition="$2"; shift 2
  outdir=".validation-corpus-cross-family-${noise}-replication-50ms-${condition}-piecewise-742520-522"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313601 --motion-profile sampled-rate-piecewise-linear \
    --motion-seeds 742520 742521 742522 --noise-sample-counts 81 \
    --sample-period-s 0.05 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep "$@"
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in moderate high; do
  run_piecewise_rate_latency_50ms "$noise" baseline
  run_piecewise_rate_latency_50ms "$noise" rate27 --rate-latencies-s 0.027
done

for noise in moderate high; do
  baseline=".validation-corpus-cross-family-${noise}-replication-50ms-baseline-piecewise-742520-522"
  delayed=".validation-corpus-cross-family-${noise}-replication-50ms-rate27-piecewise-742520-522"
  python -m scripts.compare_corpus_assessments \
    "$baseline/assessment.json" "$delayed/assessment.json" \
    --output "$delayed/paired-comparison.json"
done
```

The 50 ms corpora contain 1,731 cases each (1,728 seeded noisy and three
controls). Their paired-comparison summaries report 1,728 noisy pairs per noise
level. Artifact SHA-256 values follow; each label is followed by its manifest,
assessment, or paired-comparison hash on the next line.

```text
moderate baseline manifest
cbc353b9c7e5f1670806c2768d8d789ba141def425c6ad7afec1e68e03364b61
moderate baseline assessment
112f4dea30aae5bf4a822a531857b807efe8937f75a54d72fa8caf74bf0030c7
moderate rate27 manifest
550014f3e9479d68823bdd342cd8524beb7d5b0bae135bf537c056d80076f38d
moderate rate27 assessment
eb8de0287a87c40a9b1c8516f399a1f095735d1964779b72c4774979a306e4a6
moderate rate27 paired comparison
5b23291ceb97a0a177f053c3dcc58a0a95a51bbb00adc39ba7804a51f5f70fa0
high baseline manifest
b701a8a1a1b7189f83305d9202a2c260894004a5f3e1db925133bf95aff35777
high baseline assessment
3223212cfbc103f3883226a12149cb0e7bc3ea12cedbc84f737b63ba6489148b
high rate27 manifest
2e9af3593b968fe70941ced589f2fc214ba2bb21a7a85daec155a59ccde563a8
high rate27 assessment
ca854ba7b4eb5def6e5b3c319704f0ecf139dd1bc62cb85c6cea54cd55bc0bc1
high rate27 paired comparison
c9e5d206f341cbba505e56bd4b5755ffabd71313bdfb9a56c9044266ac9e8410
```

All four assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Shared-path cross-cadence baseline comparison

To isolate sampling cadence from the noise realization, the 25 ms piecewise-rate
baseline was regenerated with the same motion seeds (742520--742522), noise
seeds (20313601--20313603), moderate/high noise profiles, OU correlation times,
truth conventions, and inference configs as the 50 ms baseline above. Both grids
cover four seconds; the 50 ms observations are every second row of the 25 ms
recordings. The cadence comparator checks the manifests/configs and verifies the
actual timestamp, orientation, and rate values at every shared time. For each
noise level, all 139,968 shared noisy input rows matched exactly across the
1,728 paired cases (maximum absolute difference 0.0).

| Noise | Correct recommendations, 25 ms → 50 ms | Evidence-set changes | Decision-status changes |
|---|---:|---:|---:|
| Moderate | 1,728 → 1,728 | None | None |
| High | 1,128 → 1,152 | 312 contracted | 24 |

At high noise, the 24 additional correct recommendations at 50 ms all came from
motion seed 742522: seed 742520 recommended 576/576 at both cadences, seed 742521
abstained on all 576 at both, and seed 742522 increased from 552/576 to 576/576.
Every recommendation at either cadence was correct; neither cadence had a data-
winner, operational, or evidence miss. This is a measured cadence sensitivity for
these synthetic paths, not a population effect or a threshold-tuning basis. It is
confined to high-noise piecewise-rate cases and three motion/noise seeds.

Regenerate the matched 25 ms baselines and compare them with the 50 ms baselines
from the preceding section:

```bash
set -e

for noise in moderate high; do
  outdir=".validation-corpus-cross-family-${noise}-replication-25ms-shared-noise-baseline-piecewise-742520-522"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313601 --motion-profile sampled-rate-piecewise-linear \
    --motion-seeds 742520 742521 742522 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
  candidate=".validation-corpus-cross-family-${noise}-replication-50ms-baseline-piecewise-742520-522"
  python -m scripts.compare_cadence_assessments \
    "$outdir/assessment.json" "$candidate/assessment.json" \
    --baseline-manifest "$outdir/manifest.json" \
    --candidate-manifest "$candidate/manifest.json" \
    --output "$outdir/cadence-comparison-50ms-baseline.json"
done
```

The 50 ms baseline artifacts are hashed in the preceding section. Matched 25 ms
baseline artifact SHA-256 values:

```text
moderate manifest
55ec28eb2c6c74cc3e13d1641a1a57fb7e90cebf8d7eedfec19500ccbbdb39ee
moderate assessment
73433ccbc27b45089be14239fa2c15dd840a8e694fb86bc3245614a8d0a1e6ac
moderate cadence comparison
f7ca76866902fdc2744dfc16254a28c7aaa482bda63b5d263ba72728be670f83
high manifest
a1f357ab6e96134557ab431cc136ca2585201f72ef003a2b258d5cef8a19db39
high assessment
2fff04bb7efe14e0a6baef19a1f610c4b99d0ae77f47106015a50193c52397ca
high cadence comparison
3ca3099a5f6ca05fbf145cdc59b22d6790ab85e5654fc7d9cdca590aea948598
```

### Shared-path rate-delay comparison across cadences

The 27 ms rate-delay conditions were also compared at both cadences using the
same motion seeds, noise seeds, and shared OU paths. Within each cadence, the
baseline and delayed manifests differ only in `rate_latency_s` (0 versus
0.027 s). Each assessment contains 1,728 noisy cases and three clean controls.

| Noise | Cadence | Correct recommendations, baseline → rate delay | Evidence-set changes | Decision-status changes |
|---|---:|---:|---:|---:|
| Moderate | 25 ms | 1,728 → 1,152 | 576 expanded | 576 |
| Moderate | 50 ms | 1,728 → 1,728 | None | None |
| High | 25 ms | 1,128 → 1,056 | 72 expanded | 72 |
| High | 50 ms | 1,152 → 1,152 | None | None |

At 25 ms, the moderate-noise decision/evidence changes were all on motion seed
742521 (576 recommendations became truth-retaining abstentions); at high noise,
all 72 changes were on seed 742522. The 50 ms rate-delay cohort had no changes
on any of its three motion seeds. Across all four within-cadence comparisons,
every recommendation was correct and there were no data-winner, operational,
evidence, or evidence-sufficiency misses.

The cross-cadence comparisons verify that the 25 and 50 ms CSVs have the same
orientation and rate values at every shared timestamp: 139,968 exact common-time
rows per comparison, with maximum absolute difference 0.0. The resulting
recommendation counts were:

| Noise | Condition | Correct recommendations, 25 ms → 50 ms | Evidence-set changes | Decision-status changes |
|---|---|---:|---:|---:|
| Moderate | Baseline | 1,728 → 1,728 | None | None |
| Moderate | Rate delay 27 ms | 1,152 → 1,728 | 576 contracted | 576 |
| High | Baseline | 1,128 → 1,152 | 312 contracted | 24 |
| High | Rate delay 27 ms | 1,056 → 1,152 | 384 contracted | 96 |

With the rate delay, the moderate-noise 50 ms cohort gained 576 correct
recommendations over 25 ms, all on seed 742521; at high noise it gained 96,
all on seed 742522. These paired results show a cadence-dependent response to
this synthetic 27 ms rate delay, not a general latency effect. There were no
false recommendations or winner/evidence misses. The three motion seeds and one
piecewise-rate family are too narrow for a performance or calibration claim; no
thresholds were changed.

Reproduce the 25 ms delayed cohorts and their within-/cross-cadence comparisons
using the 50 ms cohorts from the preceding section and the 25 ms baselines above:

```bash
set -e

for noise in moderate high; do
  delayed=".validation-corpus-cross-family-${noise}-replication-25ms-shared-noise-rate27-piecewise-742520-522"
  python -m scripts.generate_validation_corpus \
    --outdir "$delayed" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313601 --motion-profile sampled-rate-piecewise-linear \
    --motion-seeds 742520 742521 742522 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep \
    --rate-latencies-s 0.027
  python -m scripts.validate_corpus "$delayed/manifest.json" \
    --checkpoint "$delayed/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$delayed/assessment.json"
  baseline=".validation-corpus-cross-family-${noise}-replication-25ms-shared-noise-baseline-piecewise-742520-522"
  baseline50=".validation-corpus-cross-family-${noise}-replication-50ms-baseline-piecewise-742520-522"
  delayed50=".validation-corpus-cross-family-${noise}-replication-50ms-rate27-piecewise-742520-522"
  python -m scripts.compare_corpus_assessments \
    "$baseline/assessment.json" "$delayed/assessment.json" \
    --output "$delayed/paired-comparison.json"
  python -m scripts.compare_cadence_assessments \
    "$baseline/assessment.json" "$baseline50/assessment.json" \
    --baseline-manifest "$baseline/manifest.json" \
    --candidate-manifest "$baseline50/manifest.json" \
    --output "$baseline/cadence-comparison-50ms-baseline.json"
  python -m scripts.compare_cadence_assessments \
    "$delayed/assessment.json" "$delayed50/assessment.json" \
    --baseline-manifest "$delayed/manifest.json" \
    --candidate-manifest "$delayed50/manifest.json" \
    --output "$delayed/cadence-comparison-50ms-rate27.json"
done
```

Rate-delay replication artifact SHA-256 values for the new 25 ms corpora and
reports (the 50 ms artifact hashes are recorded in the preceding section):

```text
moderate rate27 manifest
791661c33d7cb8d61bc9a37b3d6acd6a2a6945cf7f65d1feb04180f3a249fc97
moderate rate27 assessment
1105f1abe051c2495cf69027833ed5e7d774a911067a95cc669b60ab7d380c85
moderate rate27 paired comparison
9e503595b4405ce283d9e4c7df3628d35f0b6481d45cfe9a21640db9120c4357
moderate rate27 cadence comparison
a6fba90506f3573f13ca84b1002b4492c2d5c1bf713a3efff94f02e71a31d604
high rate27 manifest
f1f7ddabbf56d6bdd9ec322d20a3759231e7533d4bdafb79aa425666c75c31ba
high rate27 assessment
1154e8fe921ff92612fe72d149ab562d2ea57fc444bec23b6b1fef4313a03c34
high rate27 paired comparison
e6c47ce860441cb63c43eb955c5879fc89629b68d50c0df668e97c1e196439ec
high rate27 cadence comparison
7882d11d19615a9e1f2ae4cf5ebd530255644f56e6a1e70be4d04a370e6ec6e4
```

All eight 25/50 ms assessments used implementation SHA-256
`f362951c90089589bad38c9a7b8854006c4edfb1c011eed5a5b47f6e8f8f1849`.

### Fresh-seed piecewise 40 Hz/10 Hz cadence screen

A fresh piecewise-linear-rate block tested a wider cadence gap using motion
seeds 742540--742542 and paired noise seeds 20313801--20313803. Moderate and
high OU noise, the full 192-convention sweep, and 0.05/0.1 s orientation/rate
correlation times were held fixed. Both grids cover four seconds: 161 samples
at 25 ms (40 Hz) and 41 samples at 100 ms (10 Hz). Each of the four assessments
contains 1,728 noisy cases plus three clean controls.

The cadence comparator verified identical inference configurations and all
70,848 shared-time rows per noise tier (1728 cases × 41 rows), with maximum
timestamp/orientation/rate difference 0.0 and stride ratio 4.

| Noise | Correct recommendations, 40 Hz -> 10 Hz | Top-winner misses | Evidence-set changes | Decision-status changes |
|---|---:|---:|---|---:|
| Moderate | 1,728 -> 1,728 | 0 -> 0 | None | 0 |
| High | 0 -> 1,728 | 0 -> 0 | 1,728 contracted | 1,728 |

At moderate noise, every seed had 576/576 correct recommendations at both
cadences. At high noise, each seed changed from 0/576 recommendations at 40 Hz
to 576/576 correct recommendations at 10 Hz. Every recommendation was correct;
there were no data-winner, operational, evidence, evidence-sufficiency, or false-
recommendation misses. The three clean controls per assessment also recommended
correctly.

This is a large synthetic cadence/noise interaction, not evidence that 10 Hz is
generally better. With the configured OU time constants, adjacent 25 ms noise
samples have much stronger correlation than adjacent 100 ms samples. That makes
temporal dependence a plausible explanation for the 40 Hz ambiguity under the
current evidence model, but this experiment does not isolate that mechanism from
the changed sample set. It is a reason to test dependence-aware evidence before
changing thresholds, not a threshold-tuning result.

Reproduce in a fresh ignored output directory:

```bash
set -e

root=".validation-corpus/piecewise-cross-cadence-742540-542-repro"

run_piecewise_cadence_case() {
  noise="$1"; cadence="$2"; sample_count="$3"; period="$4"
  outdir="$root/${noise}-${cadence}"
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$noise" --noise-replicates 3 \
    --noise-seed-start 20313801 \
    --motion-profile sampled-rate-piecewise-linear \
    --motion-seeds 742540 742541 742542 --noise-sample-counts "$sample_count" \
    --sample-period-s "$period" \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --truth-convention-sweep
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 100 --progress --output "$outdir/assessment.json"
}

for noise in moderate high; do
  run_piecewise_cadence_case "$noise" 25ms 161 0.025
  run_piecewise_cadence_case "$noise" 100ms 41 0.1
  python -m scripts.compare_cadence_assessments \
    "$root/${noise}-25ms/assessment.json" \
    "$root/${noise}-100ms/assessment.json" \
    --baseline-manifest "$root/${noise}-25ms/manifest.json" \
    --candidate-manifest "$root/${noise}-100ms/manifest.json" \
    --output "$root/$noise/cadence-comparison-100ms.json"
done
```

Artifact SHA-256 values:

```text
# Each row: noise/cadence, manifest SHA-256, assessment SHA-256
moderate 25ms  6fc263761c834ddf279f840e452441e49d45e3070f1c6a53ce0e46f3f8686fa4 04783c4e5e7a6ab72bc8a53a1c0f2b407fd5968c07369d7d88ef5c993803695b
moderate 100ms 6bbc795af440ce9750e4ae8aaa9937ae968f1843f8bb8513f765523a92f439fc 29f92aa81757c24573c24c6ed93239547fc85e70ea7fb6561d91ac1491795d3e
high 25ms      d0a492f3d1ca0bc07e80a8452b8e2d5f6a923c99a8be9c33a95443f1ce052c60 89f576abbc84f50ba06a7475b078685aba9d72988fbe225070c9472631ea110c
high 100ms     055464ca32a5a48ad3ac0be442e680236b09557ebac19a3a56b8debaa4d122c3 ee79ff98f59b7d1421bc7b5e7a8dd27f232ddb884c0c9dac99303b71e37e269c
```

```text
# Each row: noise tier, cadence-comparison report SHA-256
moderate 0312d3084c961fa5d44e8e91f748d375d4bcccacc0aa7e53ed1c7f8851813831
high     70e74517b9d2978b9d8430ac17329772be6eb27da07a265db84b51ff0a92c0b9
```

The generator, OU-noise helper, validator, and cadence comparator source hashes
were respectively `ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa`,
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`,
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`, and
`1c64450e36d8f27e44fbd0edb6c94292a4ed4806444ea19fc8480bddb2e82e58`. All four
assessments used the same package-plus-validator implementation SHA-256,
`5d8f1b5b189ea7ced6ccc2cc6d00d470e314d64a09bfdd51b120d2a24fc39c38`
(Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, Windows 11). These generator-derived
labels are synthetic and are not independent logger evidence.

### Fresh-seed OU covariance holdout and colored-evidence screen

A larger synthetic holdout revisited the first-order interval-residual covariance
using the independent `sampled-rate-nonanalytic` Fourier-rate generator. It used
fresh motion seeds 743400--743402 and 20 OU noise seeds (20314001--20314020) at
each of moderate and high noise. Every noisy trajectory had one labeled truth
convention, 161 uniform samples at 25 ms (four seconds), no latency, jitter, or
dropout, and orientation/rate noise time constants of 0.05/0.1 s. The moderate
standard deviations were 0.003 rad and 0.025 rad/s; the high values were doubled.
Each tier has 60 noisy cases (three motions by 20 noise seeds) plus three clean
controls, which are excluded from the colored-noise reports.

The covariance validator fitted cross-axis covariance blocks on the ten even
noise seeds and evaluated on the ten disjoint odd seeds: 30 training and 30
held-out trajectories per tier after pairing each noise seed with all three
motion seeds. For truth-convention residuals, centered and uncentered whitened
sum-of-squares were 0.989 and 1.007 per degree of freedom, respectively, and the
replicate-mean residual was 0.313 noise standard deviations. On held-out lag
blocks, the known scalar OU reference had lower RMS error than the empirical
3x3 covariance fitted on even seeds (0.059 versus 0.077 of truth lag-zero
variance) at both noise levels.

The full-recording colored-evidence comparison used the known manifest noise
parameters; it does not alter package ranking or prefix-stability decisions.
Only the empirical covariance fit uses the even/odd split; the pairwise
likelihood comparison is descriptive on these same full-recording cases, not a
held-out likelihood or calibration test.
All 60 package data winners per tier matched the synthetic label, and no package
recommendation was wrong. Recommendation coverage nevertheless varied by motion
seed:

| Noise | Winners | Correct rec. (abstain) | Package LLR (i.i.d.) | Known LLR (i.i.d.) | Known LLR (OU) |
|---|---:|---:|---:|---:|---:|
| Moderate | 60/60 | 40 (20) | 810.78 | 766.49 | 1,808.00 |
| High | 60/60 | 15 (45) | 200.67 | 189.72 | 447.54 |

For seeds 743400/743401/743402, moderate-noise recommendation counts were
20/20/0 and high-noise counts were 0/15/0. The known-noise i.i.d. and colored
winner-versus-best-competitor scores were positive in all 60 cases at both noise
levels. Across six one-field convention counterfactuals per case, both reference
scorings favored truth in 360/360 paired comparisons. The larger colored scores
show that temporal covariance changes evidence magnitude under this idealized
model, not that package confidence or recommendation thresholds are calibrated.

This holdout supports the first-order covariance calculations for one known OU
noise process and three seeds from one motion family. It does not establish
general evidence calibration, evaluate transformed-candidate covariance exactly,
explain the separate cadence interaction, or validate real recordings. Do not
change package thresholds or confidence claims from this result.

Reproduce in a fresh ignored output directory:

```bash
set -e

root=".validation-corpus/ou-residual-holdout-743400-402-repro"
for noise in moderate high; do
  python -m scripts.generate_validation_corpus \
    --outdir "$root/$noise" --noise-profile "$noise" --noise-replicates 20 \
    --noise-seed-start 20314001 \
    --motion-profile sampled-rate-nonanalytic \
    --motion-seeds 743400 743401 743402 --noise-sample-counts 161 \
    --sample-period-s 0.025 \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1
  python -m scripts.validate_correlated_residual_model \
    "$root/$noise/manifest.json" --seed-partition all \
    --output "$root/$noise/correlated-residual-model.json"
  python -m scripts.compare_correlated_evidence \
    "$root/$noise/manifest.json" --max-cases-per-condition 60 \
    --output "$root/$noise/correlated-evidence-comparison-all60.json"
done
```

Artifact SHA-256 values:

```text
# Each label is followed by its SHA-256 digest.
moderate manifest
b5d5f2923e80f64d38ea8ad5ea542668534324ea091872aac7282790df88d550
moderate residual-model report
13395c80da44f7b1fe2a7c79f5be0e0a58d3fcd69e460564c542e0bf7d62bf10
moderate all-60 evidence report
e3fa7df3aa3bfb08f679b81afbd4760cf6bc0d33d74e207d18cd3378fc3b34b7
high manifest
2c0f94f2d90a927019cc9ab40c2b0ee98e192d2cf096aa0a153809433daa10a0
high residual-model report
b849d6f451cd6c39b4ded1eea749ed12aed5e475d354b5dc9d4a422fba5a51c3
high all-60 evidence report
8a92afe03c1ee36940a8b3f4628fc3a9f1e7485ac8845b00e5dd079f794887ae
```

The generator, OU-noise helper, corpus validator, covariance helper, residual
validator, and evidence comparator SHA-256 values were respectively
`ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa`,
`a06f9832616344fd3142767982148a91f825a49397fe10685196429d2ac2df5f`,
`d369451eb6d991cb6baa776078a1ea5220ac9bc342eddf152ab8c85eedad285d`,
`1e0599aec9e7e6f00604f55289ad33733c9b802b44037bfe24fae6b9b7b0d5fb`,
`2df21bfe990622f67de3d819fcfc90349b1b9b652b0ecf837c394bd4f47de842`, and
`4f9ee7b121fe0e9d4e93eddfeffbd38ee1b84c9428976e30b9740494bd5402f9`. The
package-plus-corpus-validator implementation SHA-256 was
`5d8f1b5b189ea7ced6ccc2cc6d00d470e314d64a09bfdd51b120d2a24fc39c38`. The run
used package version `0.0.1a0.post1.dev18+gc54c51d8d`, Python 3.14.6, NumPy
2.5.2, SciPy 1.18.0, and Windows 11. All labels are generator-derived synthetic
labels, not independent logger evidence.

### Paired noise ladder and contender diagnosis

A 4-second, single-truth-convention noise ladder used the same three burst motion
seeds (732501--732503) and 20 paired noise seeds (20311601--20311620) at each of
the three configured noise levels. All 180 cases had the correct data winner and
operational class, with truth retained in the evidence set; every case abstained,
and none produced a wrong recommendation. All 60 low-noise and all 60 moderate-noise
cases passed the stable-evidence check, but still had multiple evidence contenders.
At high noise, 12/60 passed the evidence check but retained multiple contenders;
the other 48/60 abstained because evidence was insufficient. Every motion seed had
the same all-abstain pattern across its 60 trials.

| Noise level | Cases | Evidence sufficient | Abstention reason | Median evidence LLR | Median residual lag-1 autocorrelation |
|---|---:|---:|---|---:|---:|
| Low | 60 | 60 | 60 multiple contenders | 34.46 | -0.460 |
| Moderate | 60 | 60 | 60 multiple contenders | 8.48 | -0.459 |
| High | 60 | 12 | 12 multiple contenders; 48 insufficient | 2.15 | -0.459 |

A representative low-noise case had a unique correct data winner and an evidence
LLR of 72.7, yet abstained because the next class remained within the configured
near-tie margin: its log-risk delta was 1.11 against the default 2.0 cutoff. This is
consistent with the public decision contract: sufficient evidence does not imply a
unique contender. The residual lag-one autocorrelation near -0.46 also warns that
the i.i.d. evidence scale is approximate for these sampled bursts. A paired
zero-noise baseline on the same motion seeds resolved all three cases correctly.

These results localize the current sensitivity to the separation/margin and noise
model, not to truth encoding or a wrong leading class in this cohort. They are not a
basis for narrowing the contender margin: they use one synthetic signal family,
one truth convention, and one acquisition cadence, and the residual dependence
remains unmodeled. Cross-family validation and independent logger data are still
required.

Reproduce the 180-case ladder and its paired clean baseline with:

```bash
set -e

for profile in all none; do
  if [ "$profile" = all ]; then
    outdir=.validation-corpus-maneuver-noise-ladder-732501-503
    replicates=20
  else
    outdir=.validation-corpus-maneuver-noise-none-paired-732501-503
    replicates=1
  fi
  python -m scripts.generate_validation_corpus \
    --outdir "$outdir" --noise-profile "$profile" \
    --noise-replicates "$replicates" --noise-seed-start 20311601 \
    --motion-profile sampled-rate-maneuver-bursts \
    --motion-seeds 732501 732502 732503 --noise-sample-counts 161 \
    --sample-period-s 0.025
  python -m scripts.validate_corpus "$outdir/manifest.json" \
    --checkpoint "$outdir/assessment.checkpoint.json" \
    --checkpoint-every 50 --progress --output "$outdir/assessment.json"
done
```

### Fresh-seed noisy decision screen across all motion profiles

A new synthetic screen used motion seeds `743601`--`743603` and noise seeds
`20316001`--`20316010`, which were checked against the existing local validation
manifests and were not previously used there. The independent generator produced
1,560 noisy trajectories plus three fixed control cases. It covered all seven
generator motion profiles, all three configured noise levels, and the paired
combinations of zero or 2 ms maximum timestamp jitter with zero or 0.1 whole-row
dropout probability. Nominal records contained 161 samples at 25 ms cadence;
dropout reduced observed sample counts in affected cases. Orientation and rate noise used OU time
constants of 50 ms and 100 ms. The truth was fixed to body-to-world, scalar-last
quaternions and positive body-frame `pqr` rates in radians per second; the inference
pipeline evaluated 48 candidate classes per case.

The four analytic motion profiles are deterministic generator paths, not independent
motion draws. The three sampled-rate profiles used each of the three fresh motion
seeds. Noise seeds were paired across profiles, noise levels, jitter, and dropout, so
the 1,560 rows are not 1,560 independent draws from a deployment population.

Across the 1,560 noisy cases, the package made 526 recommendations and abstained on
1,034. All 526 recommendations matched the synthetic label. The truth remained in
the evidence contender set in every case; there were 224 cases where it was not the
raw data winner, but the policy abstained rather than recommend a wrong class. Of
the abstentions, 492 were classified as insufficient evidence and 542 as multiple
evidence contenders. The confidence label remained `indeterminate` for 1,354 cases;
these labels are heuristic and are not calibrated probabilities. Of the 526
recommendations, 320 had an `indeterminate` score label and 206 had a `low` label;
none were labeled `moderate` or `high`. This is expected because `report.decision`
and the diagnostic score are separate outputs, not contradictory probability claims.

| Noise tier (orientation / rate sigma) | Cases | Correct recommendations | Data-winner misses | Evidence misses | Evidence insufficient |
|---|---:|---:|---:|---:|---:|
| Low (0.0015 rad / 0.012 rad/s) | 520 | 244 | 50 | 0 | 144 |
| Moderate (0.003 rad / 0.025 rad/s) | 520 | 200 | 72 | 0 | 160 |
| High (0.006 rad / 0.05 rad/s) | 520 | 82 | 102 | 0 | 188 |

Motion-family behavior varied substantially:

| Motion profile | Cases | Correct recommendations | Data-winner misses | Evidence misses |
|---|---:|---:|---:|---:|
| `rich-three-axis` | 120 | 2 | 0 | 0 |
| `low-excitation` | 120 | 0 | 56 | 0 |
| `near-gimbal` | 120 | 0 | 32 | 0 |
| `single-axis` | 120 | 0 | 108 | 0 |
| `sampled-rate-nonanalytic` | 360 | 280 | 0 | 0 |
| `sampled-rate-piecewise-linear` | 360 | 242 | 0 | 0 |
| `sampled-rate-maneuver-bursts` | 360 | 2 | 28 | 0 |

This screen supports the existing conservative behavior on these generated inputs:
weakly identifying motions often failed to place truth first, while truth remained
available for abstention. It does not estimate field recommendation accuracy,
false-recommendation risk, or confidence calibration. The results are paired, cover
one truth convention, and use a small set of motion families and seeds; no thresholds
or confidence rules were changed from this screen.

Reproduce in a new ignored output directory with:

```bash
set -e

root=".validation-corpus/fresh-noise-holdout-743601-603-repro"
python -m scripts.generate_validation_corpus \
  --outdir "$root" --noise-profile all --noise-replicates 10 \
  --noise-seed-start 20316001 --motion-seeds 743601 743602 743603 \
  --noise-sample-counts 161 --sample-period-s 0.025 \
  --timestamp-jitter-amplitudes-s 0 0.002 \
  --dropout-probabilities 0 0.1 \
  --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1 --motion-profile all
python -m scripts.validate_corpus "$root/manifest.json" \
  --checkpoint "$root/assessment.checkpoint.json" --checkpoint-every 50 \
  --workers 8 --progress --output "$root/assessment.json"
```

Artifact SHA-256 values from the recorded run:

```text
manifest: 67b5e8a3dbf38d326c71016579640eac8732d5672f7eff3050a6140dfad37db4
assessment: 3b7400f2bd72a0ff7eb5c765451e51ca59831fa7d26192eb9b7bf24fa4d96b07
generator source: ac18a3f4c37a1ff1da9781511e41e7aed164160ccf1cfc114ed0ba5306dca2aa
validator source: 7d97dd8ad7db527e82fa4607330b180d5b0dbbbd2e8a89e1608b69a2fa1179e8
package plus validator implementation: fba171f4d34cdf1cfb00342b29bff39a4470426cb4ed09381cc228e6a6092e8c
```

The assessment was produced on Windows 11 with Python 3.14.6, NumPy 2.5.2, SciPy
1.18.0, and package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261101`. These
remain generator-derived labels, not independent logger evidence.

### Fresh clean full-convention screen across sampled-rate families and cadences

A clean, full-convention screen extended the sampled-rate comparison to three
fresh motion seeds (`751101`--`751103`), three independently integrated rate
families (`sampled-rate-nonanalytic`, `sampled-rate-piecewise-linear`, and
`sampled-rate-maneuver-bursts`), and 20, 40, and 80 Hz over the same four-second
duration. Each family/cadence cell contained all 192 quaternion/rate convention
labels for each motion seed (576 convention cases) plus the three fixed generator
controls. The nine cells therefore comprise 5,184 convention cases and 27 controls.
The new motion seeds were checked against the local validation manifests and
documentation before use.

All three channels were generated without measurement noise, timestamp jitter,
dropout, or latency. Across all 5,211 cases there were no data-winner, operational,
or evidence misses and no incorrect recommendations. The 576 convention cases in
eight cells were all recommended correctly. At 20 Hz, the Fourier-rate family
recommended all 192 conventions for seed `751101`, but abstained for all 192 each
for seeds `751102` and `751103` because evidence was insufficient. In those 384
abstentions, the labeled class still won the data ranking and remained the
operational class. The three fixed controls were recommended correctly in every
cell.

| Sampled-rate family | 20 Hz recommendations / abstentions (of 576) | 40 Hz recommendations / abstentions (of 576) | 80 Hz recommendations / abstentions (of 576) |
| --- | ---: | ---: | ---: |
| Fourier (`sampled-rate-nonanalytic`) | 192 / 384 | 576 / 0 | 576 / 0 |
| Piecewise-linear | 576 / 0 | 576 / 0 | 576 / 0 |
| Maneuver bursts | 576 / 0 | 576 / 0 | 576 / 0 |

To verify the nested cadence construction rather than assume it, the canonical
no-noise base-convention CSV rows were compared at their common timestamps for
each family/seed pair. All eight serialized fields (`time_s`, quaternion, and
angular-rate components) matched exactly across 20/40, 40/80, and 20/80 Hz:
17,496 field comparisons, zero differences. This checks paired-input integrity
for these generated paths; it does not estimate noisy cadence robustness or
confidence calibration. The result is an implementation/identifiability check,
not validation on independent loggers.

Reproduce into new ignored directories with:

```bash
set -e

for profile in \
  sampled-rate-nonanalytic \
  sampled-rate-piecewise-linear \
  sampled-rate-maneuver-bursts; do
  for cadence in 20hz 40hz 80hz; do
    case "$cadence" in
      20hz) period=0.05; count=81 ;;
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/clean-conventions-751101-103-${profile}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile none \
      --motion-seeds 751101 751102 751103 --noise-seed-start 20261030 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$profile"
    python -m scripts.validate_corpus "$root/manifest.json" \
      --checkpoint "$root/assessment.checkpoint.json" --checkpoint-every 100 \
      --workers 8 --progress --output "$root/assessment.json"
  done
done
```

The manifest and assessment SHA-256 values are:

```text
sampled-rate-nonanalytic 20 Hz  manifest 53ee31e6cf05a43102a9dd123c267abb051430ab08b34ec47ff72ea6989f4d29
sampled-rate-nonanalytic 20 Hz  assessment da2d982e044f158fe67657e2c289a96141ecb31d4bec00abcd977c73366a0331
sampled-rate-nonanalytic 40 Hz  manifest 84824eaafa107fa26286dbb9d4dac206ce84074575012e024ce193b175ebf75f
sampled-rate-nonanalytic 40 Hz  assessment 24defe14b6aa38512264e9741fce1736f55d882dbf7a0a553fbc3b7e1699c37d
sampled-rate-nonanalytic 80 Hz  manifest e061d2933ec8bc879d6de60a7c3e1765f3fe6206229d9f3a0a9813b2c917df86
sampled-rate-nonanalytic 80 Hz  assessment dbf908238cd6ad6b3cdcc2e5b68d7e5086ef8950abed79d1555a065cad0b9c0c
sampled-rate-piecewise-linear 20 Hz  manifest e1fb816cc04ab31f1ebb09bdac7c922b9165632e255f06c4d066a52bec0a37d7
sampled-rate-piecewise-linear 20 Hz  assessment a126a90d0a35a5ce149b4663f959b302f104131ce9ff5787f4143f39d52b4931
sampled-rate-piecewise-linear 40 Hz  manifest 9ae76a78ac6323b0c0ffed2f5d59ed9877657601130ae3b8d04c43e22ea470bb
sampled-rate-piecewise-linear 40 Hz  assessment 63533d713c1409ad8ae0d5021cc4155352b6a20e7b6150af2d52c6357d80f2c4
sampled-rate-piecewise-linear 80 Hz  manifest 00d3a2834f2df5024f377d2fa83a5d849ae740200de13b4cc9876672ae7fb585
sampled-rate-piecewise-linear 80 Hz  assessment 663ed03e6660724b28f98c8257ed51d17bf7a8b2391699dd6cc8e156483318a8
sampled-rate-maneuver-bursts 20 Hz  manifest d2eafb886dd485483f0f33347de5c73eb2a2e3d9f1eca22252f7d4039e0aaf39
sampled-rate-maneuver-bursts 20 Hz  assessment eeadfc612596a120b245d4f4e4a1b2c918f6b130b19248da97220d5cfd0f547d
sampled-rate-maneuver-bursts 40 Hz  manifest b5797bc30e300175b7a48c4c9c44170665660365c252d0dfea7ee37ef145ec2b
sampled-rate-maneuver-bursts 40 Hz  assessment 3379fe76f0faf4689d14b39ba991f1a79ae87eab8400173a4f0920b9abc9b9f5
sampled-rate-maneuver-bursts 80 Hz  manifest 1c933ed2e94726d174d776255b852997c39e05a2621623d1cf5c730bb65531ea
sampled-rate-maneuver-bursts 80 Hz  assessment f4eb8ce56ece837377e794abb79747a08b71b7a6ec2853ad29a13048a2475152

generator SHA-256: 5d9879782f523115cce806fa20b5b5a1609b1988939aeff984031b8390e32898
validator SHA-256: a9e8ed8d7086c20176d2bbd24be99891e55f753e0ca13c7ab3b4b304d4c27752
package plus validator implementation SHA-256: 76c065ead2c5c7c011dfd268355f6d10ea164df6037c450685e582edddfabd3d
```

The assessments were produced on Windows 11 with Python 3.14.6, NumPy 2.5.2,
SciPy 1.18.0, and package version `0.0.1a0.post1.dev25+gc2594b4a8.d20261101`.
The all-clean recommendation result at 40/80 Hz is specific to these noiseless
generator paths; the Fourier 20 Hz abstentions show the policy remains sensitive
to motion seed even with correct rankings. No noise thresholds or confidence
labels were changed.

### Convention-sweep generator reuse

The independent generator now prepares each physical trajectory, seeded channel
noise, timestamp realization, and dropout mask once per distinct measurement
condition, then applies the 192 truth encodings to those shared arrays. A regression
test checks that motion integration is not repeated for each convention. This is a
corpus-generation optimization only; it does not change inference behavior.

On this Windows workstation, generating a matching 579-case cell took about 0.86 s
at 20 Hz and 1.48 s at 80 Hz, compared with earlier measured runs of about 39.5 s
and 69.5 s. These single-host elapsed times imply roughly a 46x reduction, not a
portable throughput guarantee. The optimized generator reproduced all 580 files
(579 CSVs and the manifest) byte-for-byte in each compared cell. The manifest hashes
therefore match the values above, so the recorded assessments still correspond to
the exact same inputs; inference did not need to be rerun for this optimization.

Optimized generator SHA-256: `FF497728BE95164114B4D0A35494DA54ADF34A20FC3848D1C89761290525B306`.

### Fresh moderate-noise full-convention screen across sampled-rate families and cadences

A fresh screen crossed motion seeds `751201`--`751203`, the Fourier, piecewise-linear,
and maneuver-burst sampled-rate families, and 20, 40, and 100 Hz over four seconds.
Each family/cadence cell contains 192 raw quaternion/rate convention encodings per
motion seed: 5,184 convention cases plus 27 fixed controls. Each case uses moderate
noise (`0.003 rad` orientation and `0.025 rad/s` rate standard deviation), with
50 ms and 100 ms OU correlation times. The noise seed `20416101` is reused across
the cells; this is one paired noise-seed block, not 5,184 independent trials.
Timestamp jitter, dropout, and channel latency are zero.

This screen predates support for 80 Hz, so it tested 100 Hz as its densest cadence.
The OU sampler now supports 12.5 ms through conditional half-millisecond bridges;
the historical 20/40/100 Hz corpora retain their original seeded values at
integer-millisecond timestamps. A focused three-seed 80 Hz run now exercises
canonical body-frame rate noise; its results are recorded below.

Across the 5,184 convention cases, there were no data-winner misses, operational
misses, evidence misses, or false recommendations. The package made 2,016 correct
recommendations and abstained on 3,168 cases: 480 for insufficient evidence and
2,688 for multiple evidence contenders. All 27 fixed controls were recommended
correctly. Recommendation coverage varied sharply by motion family and seed:

| Sampled-rate family | 20 Hz recommendations / abstentions | 40 Hz recommendations / abstentions | 100 Hz recommendations / abstentions |
| --- | ---: | ---: | ---: |
| Fourier (`sampled-rate-nonanalytic`) | 576 / 0 | 384 / 192 | 384 / 192 |
| Piecewise-linear | 384 / 192 | 192 / 384 | 96 / 480 |
| Maneuver bursts | 0 / 576 | 0 / 576 | 0 / 576 |

For example, the piecewise-linear 100 Hz cell recommended 96/192 conventions for
motion seed `751202`, but none for the other two seeds. In that seed, all 96 world-
frame rate encodings were recommended while all 96 body-frame encodings abstained.
This is one motion/noise realization, not evidence of a general frame preference.

A channel-isolation probe on that same piecewise-linear motion (`751202`, 100 Hz,
noise seed `20416101`) compared clean, orientation-only, rate-only, and combined
moderate noise. Each run contained all 192 convention encodings plus the controls:

| Condition | Body-frame recommendations | World-frame recommendations | Outcome |
| --- | ---: | ---: | --- |
| Clean | 96 / 96 | 96 / 96 | Unique sufficient evidence |
| Orientation noise only | 96 / 96 | 96 / 96 | Unique sufficient evidence |
| Rate noise only | 0 / 96 | 0 / 96 | Insufficient evidence for both frames |
| Combined moderate noise | 0 / 96 | 96 / 96 | Multiple contenders for body; unique evidence for world |

There were no winner or evidence misses in these probes. The combined-noise split
suggests an interaction rather than a clean- or orientation-noise frame effect, but
does not establish its cause. The generator now exposes `--rate-noise-frame body`
and `--rate-noise-frame world`: each samples one shared noise process in the named
frame and rotates it into the other rate frame using the rate-time attitude.
Manifest metadata records the selected frame. The default
`--rate-noise-frame truth-rate-frame` preserves the historical per-convention
injection and is not a physically paired body/world noise vector. Coordinate-pair
regression tests cover both canonical choices. The three-seed holdout below is
the first full-convention screen under this model; broader motion-family and
noise-seed coverage remains future validation work. No inference thresholds were
changed.

Reproduce the nine main cells into new ignored directories with:

```bash
set -e

for profile in \
  sampled-rate-nonanalytic \
  sampled-rate-piecewise-linear \
  sampled-rate-maneuver-bursts; do
  for cadence in 20hz 40hz 100hz; do
    case "$cadence" in
      20hz) period=0.05; count=81 ;;
      40hz) period=0.025; count=161 ;;
      100hz) period=0.01; count=401 ;;
    esac
    root=".validation-corpus/moderate-full-convention-751201-203-20416101-${profile}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 1 \
      --noise-seed-start 20416101 --motion-seeds 751201 751202 751203 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$profile" \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1
    python -m scripts.validate_corpus "$root/manifest.json" \
      --checkpoint "$root/assessment.checkpoint.json" --checkpoint-every 50 \
      --workers 8 --progress --output "$root/assessment.json"
  done
done
```

### Canonical body-frame rate noise at 80 Hz

On 2026-11-01, the new half-millisecond OU bridge and canonical-frame rate-noise
mode were exercised together in a fresh piecewise-linear holdout. It used motion
seeds `751301`--`751303`, noise seed `20416111`, 80 Hz (321 samples over four
seconds), moderate orientation/rate noise (`0.003 rad` / `0.025 rad/s`), and OU
time constants of 50/100 ms. `--rate-noise-frame body` samples one body-fixed rate
noise vector process and rotates it into world coordinates with the rate-time
attitude. The manifest records `rate_noise_frame: "body"` for every generated
convention case.

The corpus contained 576 convention cases plus three fixed controls. There were
no data-winner, operational, evidence, or recommendation misses. All 576
convention cases retained truth among multiple evidence contenders and abstained;
none was recommended. The split was identical by truth rate frame and motion
seed:

| Truth rate frame | Cases | Recommendations | Truth-retaining abstentions | Misses |
| --- | ---: | ---: | ---: | ---: |
| Body | 288 | 0 | 288 | 0 |
| World | 288 | 0 | 288 | 0 |

All three fixed controls were recommended correctly. This demonstrates that the
80 Hz acquisition and the paired coordinate transformation run end to end, but
the zero convention coverage is a significant limitation—not evidence of useful
recovery performance. It is one motion family and one OU-noise seed block, so
the next validation step is to broaden motion/noise diversity, not tune inference
thresholds against this result.

Reproduce in a new ignored output directory with:

```bash
root=.validation-corpus/canonical-body-piecewise-80hz-751301-303-20416111-repro
python -m scripts.generate_validation_corpus \
  --outdir "$root" --noise-profile moderate --noise-seed-start 20416111 \
  --motion-seeds 751301 751302 751303 --noise-sample-counts 321 \
  --sample-period-s 0.0125 --truth-convention-sweep \
  --motion-profile sampled-rate-piecewise-linear \
  --orientation-noise-correlation-time-s 0.05 \
  --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
python -m scripts.validate_corpus "$root/manifest.json" \
  --workers 8 --checkpoint "$root/assessment.checkpoint.json" \
  --checkpoint-every 50 --progress --output "$root/assessment.json"
```

Artifact SHA-256 values are:

```text
manifest   8230c8e0dd0abb9ab586fffe553bf9697b2e9b94c826cc07db38ed551df711ea
assessment 2f82f49fe326bc72b4cdb951836e329631fff21fa3006e68b8514cd85c35b08f
generator  b972c40bb8450b11362267396f25663292e5ccec89a698b8fc24e5d58b1f81f8
OU helper a870d40c5447bc4f4e2c552e9be3ff554a27ef91ae4c4a1d2b0700b4e63551bc
validator fe5f36629757c0d9933a1bc87c5bfdf15743e51c70f2cbaa334636216a2fa20a
```

Assessment environment: Windows 11, Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0;
the package implementation SHA-256 was
`20b8ea122ec67045d9720a26b57bf970b141bbc96d3ec4fca30fdc1b2ff1d8c2`.

### Paired 80 Hz family matrix and Fourier cadence comparison

A second noise block (`20416112`) crossed all three sampled-rate families with
the same motion seeds (`751301`--`751303`) at 80 Hz, using the same moderate
noise levels, OU time constants, and body-referenced rate-noise model. Each
family had 576 convention cases (three motions × 192 encodings) plus three fixed
controls. No convention case was recommended, and there were no data-winner,
operational, evidence, or recommendation misses. All nine fixed controls were
recommended correctly.

| Sampled-rate family | Body-frame recommendations / abstentions | World-frame recommendations / abstentions | Abstention reasons per truth frame |
| --- | ---: | ---: | --- |
| Fourier | 0 / 288 | 0 / 288 | 288 multiple evidence contenders |
| Piecewise-linear | 0 / 288 | 0 / 288 | 288 multiple evidence contenders |
| Maneuver bursts | 0 / 288 | 0 / 288 | 192 insufficient evidence; 96 multiple contenders |

For the same Fourier motions and noise seed, separate 40 and 100 Hz assessments
provided cadence context. Forty hertz recommended 384/576 conventions (192 per
truth frame); 80 and 100 Hz recommended none. A paired 40-to-80 Hz comparison
matched all 576 noisy cases and 92,736 common-timestamp input rows exactly
(maximum absolute difference 0). The denser assessment expanded the evidence set
in all 576 cases and changed 384 decisions from recommendation to abstention;
there were no winner or recommendation misses. Thus this block shows a strong
cadence-associated loss of recommendation coverage despite exact shared samples,
not a body/world asymmetry. It does not identify the cause or establish how often
this behavior occurs: this remains one paired noise-seed block. Do not tune
thresholds from it.

#### Full-cohort diagnosis of the 40-to-80 Hz shift

The v8 validation assessment records `nearest_alternative_class_id` and
`nearest_alternative_log_risk_delta` for every case with at least two quotient
classes. The latter is the log ratio of the nearest competing class's
floor-adjusted geometric risk to the best class's floor-adjusted risk. The cadence
comparison report now summarizes paired deltas overall and by motion seed/truth rate
frame. Older assessments remain readable, but naturally have no recorded margin.

Across the 576 paired noisy convention cases, all sequential evidence checks passed
at both cadences, there were no data-winner or evidence misses, and the nearest
alternative class ID was unchanged in all cases. Yet its log-risk delta decreased
in every pair:

| Cadence | q10 log-risk delta | Median | q90 | Unique-contender cases |
| ---: | ---: | ---: | ---: | ---: |
| 40 Hz | 1.1542 | 2.4944 | 2.5702 | 384 / 576 |
| 80 Hz | 0.6630 | 1.7456 | 1.8229 | 0 / 576 |
| 80 Hz minus 40 Hz | −0.7699 | −0.7435 | −0.4782 | — |

The default margin is `2.0`. Thus 384 cases crossed from just outside to inside
the relative-risk margin; the other 192 were already ambiguous at 40 Hz. There were
no transitions from multiple contenders to one. The six seed/frame strata show the
same pattern:

| Motion seed | Truth rate frame | Cases | 40 Hz median delta | 80 Hz median delta | Unique → multiple |
| ---: | --- | ---: | ---: | ---: | ---: |
| 751301 | body | 96 | 2.4907 | 1.7501 | 96 |
| 751301 | world | 96 | 2.5693 | 1.8229 | 96 |
| 751302 | body | 96 | 1.1833 | 0.7051 | 0 |
| 751302 | world | 96 | 1.1542 | 0.6630 | 0 |
| 751303 | body | 96 | 2.5702 | 1.8003 | 96 |
| 751303 | world | 96 | 2.4981 | 1.7412 | 96 |

Each stratum's 96 convention encodings have the same margin to numerical precision;
they are convention variants, not 96 independent motion/noise replicates. This is
one OU-noise seed and three motion seeds from one family. The result strongly
localizes this cohort's recommendation loss to a systematic cadence-associated
shift in the geometric-risk margin—not sequential evidence insufficiency. The
window scales are specified in sample counts, so their physical durations also
shrink as cadence increases; that is a plausible contributor, not yet a general
causal result. Do not adjust thresholds from this one synthetic block. Replicate
with fresh noise seeds and varied motion families before considering a
physical-duration or cadence-aware scoring policy.

As a cross-check, for the paired scalar-last `pqr` body-to-world/body-rate case at
motion seed `751301`, the package i.i.d. log-likelihood ratio was `639.29` at 40 Hz
and `550.07` at 80 Hz, with sufficiency true at both. An offline first-order
known-noise colored reference gave `1243.65` and `1248.45`. That reference assumes
small rotations, synchronized uniform samples, and stationary isotropic noise; it
is diagnostic, not calibration. It is consistent with the full-cohort conclusion
that correlated-row evidence accumulation is not what caused these abstentions.

The enriched assessments and comparison report have SHA-256 values:

```text
40 Hz assessment  9c2de79e0e3e223c394400bcd15e28651ac9eedb304f1ac3de2b23240c067120
80 Hz assessment  71d9100e985290953b9edc6f0c80ca741cf9a8beb84371ab900021690960d95b
comparison       23f2fdbfcc5c7f8c90eff2eaacd2639c83982ee8573e883a9d8b4510f7052368
```

Recreate the enriched assessments and comparison using fresh checkpoint/output
filenames in the existing ignored corpus directories:

```bash
root40=.validation-corpus/canonical-body-fourier-40hz-751301-303-20416112
root80=.validation-corpus/canonical-body-80hz-sampled-rate-nonanalytic-751301-303-20416112
python -m scripts.validate_corpus "$root40/manifest.json" --workers 4 \
  --checkpoint "$root40/assessment-risk-margin.checkpoint.json" \
  --checkpoint-every 50 --progress --output "$root40/assessment-risk-margin.json"
python -m scripts.validate_corpus "$root80/manifest.json" --workers 4 \
  --checkpoint "$root80/assessment-risk-margin.checkpoint.json" \
  --checkpoint-every 50 --progress --output "$root80/assessment-risk-margin.json"
python -m scripts.compare_cadence_assessments \
  "$root40/assessment-risk-margin.json" "$root80/assessment-risk-margin.json" \
  --baseline-manifest "$root40/manifest.json" \
  --candidate-manifest "$root80/manifest.json" \
  --output "$root40/comparison-risk-margin.json"
```

#### Independent cadence-margin replication

A second paired block used new Fourier motion seeds `751304`--`751306` and
independent OU noise seed `20416113`, with the same four-second duration, moderate
noise levels, OU time constants, body-referenced rate noise, and 40/80 Hz grids.
The shared-timestamp measurements again matched exactly: 576 paired noisy cases,
92,736 common rows, maximum absolute difference zero. At 40 Hz all 576 cases were
recommended; at 80 Hz 432 remained recommendations and 144 abstained with multiple
contenders. There were no data-winner misses, evidence misses, or insufficient
evidence outcomes at either cadence. The nearest alternative class was unchanged
in all 576 pairs.

| Cadence | q10 log-risk delta | Median | q90 | Unique-contender cases |
| ---: | ---: | ---: | ---: | ---: |
| 40 Hz | 2.5020 | 3.6921 | 4.1476 | 576 / 576 |
| 80 Hz | 1.7897 | 2.8667 | 3.2801 | 432 / 576 |
| 80 Hz minus 40 Hz | −0.8655 | −0.8191 | −0.7123 | — |

The threshold crossing was concentrated in the middle motion seed:

| Motion seed | Truth rate frame | Cases | 40 Hz median delta | 80 Hz median delta | Unique → multiple |
| ---: | --- | ---: | ---: | ---: | ---: |
| 751304 | body | 96 | 3.7201 | 2.8841 | 0 |
| 751304 | world | 96 | 4.0098 | 3.1704 | 0 |
| 751305 | body | 96 | 2.5020 | 1.7897 | 96 |
| 751305 | world | 96 | 2.7934 | 2.0338 | 48 |
| 751306 | body | 96 | 3.8210 | 2.9980 | 0 |
| 751306 | world | 96 | 3.9022 | 3.0647 | 0 |

This independently reproduces the downward cadence shift in all cases while
showing that recommendation impact depends on motion: the previous block lost
384 recommendations, this block lost 144. As before, 96 convention encodings per
seed/frame are repeated representations of one physical trajectory, not independent
replicates. Two independent noise blocks support investigating cadence sensitivity
in the relative-fit score; they do not calibrate the margin or establish field
coverage. Do not tune the threshold. The next useful synthetic test varies motion
family while retaining fresh noise blocks; a physical-duration window policy still
requires held-out validation.

The new manifests, enriched assessments, and comparison report have SHA-256 values:

```text
40 Hz manifest    aee5ef31a97c6d4aedeb1e7611dab21e11cb9d0989b35e9da8aab3413e08447e
80 Hz manifest    7dcefbcbc14e1b4afe1ffebe042dfc86358b81ec78d9e8768232b182321cb321
40 Hz assessment dec72642544349756838b6f91a405adc1eb7e017d0cdd1c93703db498a19af0e
80 Hz assessment 717bee354eb3cd0ead1ddb9a3804da06393361bf8a589239f553d1f2f9f5a64f
comparison       12afa17ba9dd6534233c19244f8a17296e330a60f04b02f68bb5f3966b457e8c
```

Recreate in fresh ignored directories with:

```bash
set -e

for cadence in 40hz 80hz; do
  case "$cadence" in
    40hz) period=0.025; count=161 ;;
    80hz) period=0.0125; count=321 ;;
  esac
  root=".validation-corpus/canonical-body-fourier-${cadence}-751304-306-20416113-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$root" --noise-profile moderate --noise-seed-start 20416113 \
    --motion-seeds 751304 751305 751306 --noise-sample-counts "$count" \
    --sample-period-s "$period" --truth-convention-sweep \
    --motion-profile sampled-rate-nonanalytic \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
  python -m scripts.validate_corpus "$root/manifest.json" --workers 4 \
    --checkpoint "$root/assessment-risk-margin.checkpoint.json" \
    --checkpoint-every 50 --progress --output "$root/assessment-risk-margin.json"
done

root40=.validation-corpus/canonical-body-fourier-40hz-751304-306-20416113-repro
root80=.validation-corpus/canonical-body-fourier-80hz-751304-306-20416113-repro
python -m scripts.compare_cadence_assessments \
  "$root40/assessment-risk-margin.json" "$root80/assessment-risk-margin.json" \
  --baseline-manifest "$root40/manifest.json" \
  --candidate-manifest "$root80/manifest.json" \
  --output "$root40/comparison-risk-margin.json"
```

#### Cross-family cadence screen with a fresh noise block

Noise seed `20416114` and new motion seeds `751307`--`751309` were then tested
at both 40 and 80 Hz for piecewise-linear and maneuver-burst rates. Each cell
again used 576 convention cases and three fixed controls. The 576 paired noisy
inputs per family shared 92,736 common-timestamp rows exactly (maximum absolute
difference zero); there were no data-winner misses or evidence misses, and the
nearest alternative class remained unchanged in every pair.

| Motion family | 40 → 80 Hz recommendations | Contender-set expansions | Unique → multiple | Median log-risk delta, 40 → 80 Hz | Evidence insufficient, 40 → 80 Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Piecewise-linear | 288 → 0 | 288 / 576 | 288 / 576 | 1.9929 → 1.3601 (−0.6320) | 0 → 0 |
| Maneuver bursts | 0 → 0 | 576 / 576 | 0 / 576 | 0.1449 → 0.0683 (−0.0766) | 192 → 192 |

The piecewise result repeats a substantial recommendation loss, while the burst
cohort was already far inside the multiple-contender region and had no
recommendations to lose. Its contender sets still expanded at 80 Hz, but the
decision status did not change. Quantiles also shifted downward in both families:

| Motion family | 40 Hz q10 / median / q90 | 80 Hz q10 / median / q90 | Median paired shift |
| --- | --- | --- | ---: |
| Piecewise-linear | 1.6115 / 1.9929 / 2.1076 | 0.9937 / 1.3601 / 1.4263 | −0.6320 |
| Maneuver bursts | 0.0600 / 0.1449 / 0.4041 | 0.0249 / 0.0683 / 0.1997 | −0.0766 |

Across the two Fourier blocks, the piecewise block, and this burst block, the
nearest alternative's log-risk delta decreased at denser cadence. Whether that
crosses the cutoff—and therefore affects recommendations—depends on motion and
the starting margin. The 96 convention encodings per motion-seed/frame are not
independent replicates; each family still contributes only three motion seeds
under one new noise seed. These synthetic results justify testing the
sample-count-window effect across more acquisition conditions; they do not
calibrate the cutoff or prove a universal cadence correction.

The new manifests, enriched assessments, and comparison reports have these
SHA-256 values:

```text
piecewise 40 Hz manifest    6f74475480ed10bbb6804fe16055a25c64079be1ee7ca65a54d7667f28f609be
piecewise 80 Hz manifest    05c170b47455584ecbf9ddb1d34bc94550d3807cd70ddf7367e7372b8944574d
bursts 40 Hz manifest       f9119dc53c0fcb3efa35c984fd8b45f0eef33869837052ba5bbc71997777d5c0
bursts 80 Hz manifest       84359ddf2673e4f19c3cef95a57a73a27a359745a60a27532f00bf8e51823872
piecewise 40 Hz assessment  8f75e9f3767bcba1513b83e4ed6c450e6bb1aa04fe628be1279dc3cdd74ff712
piecewise 80 Hz assessment  f6c9fc21dbcdbe15e3942ca443bcaba2e7162a2a584d05d41a64f75b3953c321
piecewise comparison        cb0543942c9017fbde7dd2647a8469758534ab33c1f29f1c4adc0a0f574e9090
bursts 40 Hz assessment     71984da0fc0fc56860e99075a72c52b1e0fb81d6984735f1e1bece78a3692263
bursts 80 Hz assessment     aada997ff665c92f37440e7516ecc31e191ed310cb7bfe2dc71ca39a255735cf
bursts comparison           9aa37a628c86ba8addfd2805504bb5feb6ebfaa9606118e688cefefefc7dd342
```

Recreate both motion-family pairs in fresh ignored directories with:

```bash
set -e

for family in piecewise bursts; do
  case "$family" in
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/canonical-body-${family}-${cadence}-751307-309-20416114-repro"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-seed-start 20416114 \
      --motion-seeds 751307 751308 751309 --noise-sample-counts "$count" \
      --sample-period-s "$period" --truth-convention-sweep \
      --motion-profile "$profile" --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 2 \
      --checkpoint "$root/assessment-risk-margin.checkpoint.json" \
      --checkpoint-every 50 --progress --output "$root/assessment-risk-margin.json"
  done
done

for family in piecewise bursts; do
  root40=".validation-corpus/canonical-body-${family}-40hz-751307-309-20416114-repro"
  root80=".validation-corpus/canonical-body-${family}-80hz-751307-309-20416114-repro"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-risk-margin.json" "$root80/assessment-risk-margin.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-risk-margin.json"
done
```

Reproduce the 80 Hz family matrix and Fourier 40/100 Hz extensions in fresh
ignored directories with:

```bash
set -e

for profile in \
  sampled-rate-nonanalytic \
  sampled-rate-piecewise-linear \
  sampled-rate-maneuver-bursts; do
  root=".validation-corpus/canonical-body-80hz-${profile}-751301-303-20416112-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$root" --noise-profile moderate --noise-seed-start 20416112 \
    --motion-seeds 751301 751302 751303 --noise-sample-counts 321 \
    --sample-period-s 0.0125 --truth-convention-sweep \
    --motion-profile "$profile" --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
  python -m scripts.validate_corpus "$root/manifest.json" \
    --workers 8 --checkpoint "$root/assessment.checkpoint.json" \
    --checkpoint-every 50 --progress --output "$root/assessment.json"
done

for cadence in 40hz 100hz; do
  case "$cadence" in
    40hz) period=0.025; count=161 ;;
    100hz) period=0.01; count=401 ;;
  esac
  root=".validation-corpus/canonical-body-fourier-${cadence}-751301-303-20416112-repro"
  python -m scripts.generate_validation_corpus \
    --outdir "$root" --noise-profile moderate --noise-seed-start 20416112 \
    --motion-seeds 751301 751302 751303 --noise-sample-counts "$count" \
    --sample-period-s "$period" --truth-convention-sweep \
    --motion-profile sampled-rate-nonanalytic \
    --orientation-noise-correlation-time-s 0.05 \
    --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
  python -m scripts.validate_corpus "$root/manifest.json" \
    --workers 8 --checkpoint "$root/assessment.checkpoint.json" \
    --checkpoint-every 50 --progress --output "$root/assessment.json"
done

root40=.validation-corpus/canonical-body-fourier-40hz-751301-303-20416112-repro
root80=.validation-corpus/canonical-body-80hz-sampled-rate-nonanalytic-751301-303-20416112-repro
python -m scripts.compare_cadence_assessments \
  "$root40/assessment.json" "$root80/assessment.json" \
  --baseline-manifest "$root40/manifest.json" \
  --candidate-manifest "$root80/manifest.json" \
  --output "$root40/comparison-40hz-vs-80hz.json"
```

Artifact SHA-256 values:

```text
80 Hz Fourier manifest      c07137978b93dd27703a2348671f55efe2cdd18c3249ba4e900dc66be951139c
80 Hz Fourier assessment    ac7b5ef5e6423813f29cf4f39c893fba5691e5a47760e59bed238fbebfda8c19
80 Hz piecewise manifest    99f5b33771553f0e73ea09ffc27b4217e986d442fddc252a7c169cdbc9e47f0e
80 Hz piecewise assessment  efc81d70f4cd5936e601088dc00e76ae0a5d0c6e289384c5fefd156c46ecb217
80 Hz bursts manifest       9e2e3e7d71def843a9caf533a979af5780faad92bf1c41c561c6c49841568ee3
80 Hz bursts assessment     f854d7acec74b5f4aae222b29a3ac0ca2bd6277f1ae04570f758dc7489c0148a
40 Hz Fourier manifest     34c598dce4ce9e79c8d12e1484aa78c0dca5382e7911dff939d76c1e2bbb048f
40 Hz Fourier assessment   472d07395ec786c3525f947b8ca6b822247306dec134bfe800ddc0b6fb567f97
100 Hz Fourier manifest    d09c8affb26a6f1ce695dca8594568dbf437f0c642daa63410c805f30a6f9e06
100 Hz Fourier assessment  ef3a2eb14a278719d345a3338d6dc07fd08eb7f77413125578af15caf95984b8
40-to-80 comparison        397dc138772da933eb54e8902102148c2f17827e64b6f428a3c7406439ced273
```

### Five-noise-seed cadence replication across rate families

A fresh paired block used motion seeds `751310`--`751312` and five consecutive
moderate-noise seeds, `20416115`--`20416119`, for Fourier, piecewise-linear, and
maneuver-burst rate profiles. Every profile was assessed at 40 and 80 Hz over the
same four-second duration, with the full 192-convention sweep, orientation/rate OU
time constants of 0.05/0.1 s, and body-referenced rate noise. The same motion/noise
seed pairs were reused across the two cadences and three families so these are
paired comparisons. The synthetic-data generator is independent of the inference
package; this block is still synthetic robustness evidence, not field accuracy.

Each family/cadence manifest contains 2,880 noisy convention cases (three motion
seeds × five noise seeds × 192 convention encodings) plus three fixed controls.
The 192 convention encodings are repeated representations of the same seeded
motion/noise paths, not 192 independent trials. There are 15 motion/noise seed
combinations per family; each is represented under two truth rate frames. The
comparator's new `risk_margin_by_motion_noise_seed_and_truth_frame` groups therefore
report 30 paired strata per family, each containing 96 convention encodings. The
report schema is now version 3; it retains the aggregate motion-seed grouping and
adds explicit noise-seed groups rather than folding independent noise replicates
together.

For every family, the comparator paired 2,880 noisy cases and verified all 463,680
common-timestamp rows exactly (maximum absolute difference 0). Every one of the 30
motion/noise/truth-frame group medians shifted downward. Recommendation counts and
risk-margin summaries were:

| Motion family | Recommendations, 40 → 80 Hz | Paired log-risk shift, q10 / median / q90 | Unique → multiple encodings | Nearest alternative changed | Evidence insufficient, 40 → 80 Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fourier | 1,728 → 576 | −0.8166 / −0.6854 / −0.5867 | 1,152 / 2,880 | 48 / 2,880 | 0 → 0 |
| Piecewise-linear | 1,920 → 1,056 | −0.8097 / −0.6888 / −0.5486 | 864 / 2,880 | 0 / 2,880 | 0 → 0 |
| Maneuver bursts | 0 → 0 | −0.3515 / −0.2543 / −0.1450 | 0 / 2,880 | 0 / 2,880 | 0 → 0 |

There were no data-winner misses or evidence misses in any family/cadence cell.
The Fourier and piecewise recommendation losses track the unique-to-multiple
contender transitions; the burst cohort was already ambiguous at both cadences,
although its relative-risk margins also narrowed in every stratum. The Fourier
nearest-alternative identity changed in 48 convention encodings; it did not change
for piecewise or bursts.

The quantiles and recommendation counts above describe paired convention
encodings, not a population distribution or confidence interval. In particular,
the 2,880 rows per family are not 2,880 independent trajectories: the independent
seed factors are three motion seeds and five noise seeds, crossed into 15
motion/noise combinations, while truth-frame and convention variants are paired.
Do not tune the near-tie margin, evidence threshold, or confidence labels from
these results. The consistent relative-margin shift justifies broader synthetic
noise/cadence testing; physical-duration or cadence-aware scoring still requires
held-out labeled acquisition logs.

Recreate these ignored artifacts from the repository root with:

```bash
set -e

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) profile=sampled-rate-nonanalytic ;;
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/cadence-noise-replication-751310-312-20416115-119-${family}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 5 \
      --noise-seed-start 20416115 --motion-seeds 751310 751311 751312 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$profile" \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 16 \
      --checkpoint "$root/assessment-risk-margins.checkpoint.json" \
      --checkpoint-every 100 --progress --output "$root/assessment-risk-margins.json"
  done
  root40=".validation-corpus/cadence-noise-replication-751310-312-20416115-119-${family}-40hz"
  root80=".validation-corpus/cadence-noise-replication-751310-312-20416115-119-${family}-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-risk-margins.json" "$root80/assessment-risk-margins.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-risk-margins.json"
done
```

Artifact SHA-256 values:

```text
Fourier 40 Hz manifest      5d7f3accf3298d65ff59976f79b7edc850b35b5bfb48e6ad8605ba2e75e1f3b6
Fourier 40 Hz assessment    54259052fe23273d4fba664489c66700dae0bff0e49760a9977f80720dc77d25
Fourier 80 Hz manifest      bf6330aacb80583e2d410b21464d5c0d702cd7ed0f26dbaf56c6c805370ec8ab
Fourier 80 Hz assessment    8acddaa961737e2a5577e27a66f11b605a94fb56fcbe6b8adb59fc926f69f4ea
Fourier comparison          e7b37c042906d8ac0058213c57faf1690c0adbc2710b12bce7f9668ec29b4c7f
Piecewise 40 Hz manifest    d4f1b50157a3e08882732aa92fee5e82a8c8593154690476abd8ff76ca5d9afb
Piecewise 40 Hz assessment  1186f2f0ceb728be9571b06452f7f66b2cc8baf49073a16aadbf3b87c1a640a4
Piecewise 80 Hz manifest    fc067d302f0f3cac0dd598742c8a81a6e7a17f4c6e3e94223d85a3c16b46d509
Piecewise 80 Hz assessment  91709abd2e8aced071e55c144756407501758c03356d24fc217ce003af4c54e2
Piecewise comparison        d21afedfaa9c78d58e4bb78cd80156a2238eaea4b321d026a1e23b4dc4fd25cc
Bursts 40 Hz manifest       ef23df2ed17e3dbed5f9f6ef1d8e4dd97d4ae0c963625a4a248807c38c6b5ba4
Bursts 40 Hz assessment     474a8cda56845966445c3421702393e3eefbbd6c98cf6d59060ee0f96502ffb5
Bursts 80 Hz manifest       4a1dd246cec93412712868f4ba1c37a44bd1a92d13e3d79a66af629b6616937f
Bursts 80 Hz assessment     1e173916c769b685679a02153c8d652dee21935bbbb74a69e1872bd05d8f4907
Bursts comparison           2c95b0a246e2de1d99814fe69dc262c110a055409c5ef23bd967aa4ee95cb900
```

### Paired low/high-noise cadence extension

The five-noise-seed moderate block above was extended to the generator's low and
high tiers, reusing motion seeds `751310`--`751312` and noise seeds
`20416115`--`20416119` across noise tiers, rate families, and 40/80 Hz. The
configured orientation/rate standard deviations are `0.0015 rad` / `0.012 rad/s`
for low noise and `0.006 rad` / `0.05 rad/s` for high noise. Both use OU correlation
times of 0.05/0.1 s for orientation/rate noise, with rate noise expressed in the
body frame. The intermediate moderate tier is `0.003 rad` / `0.025 rad/s`.

Each family/tier/cadence manifest has 2,880 noisy convention cases plus three
fixed controls: three motion seeds × five noise seeds × 192 convention encodings.
These are 15 motion/noise combinations per family/tier, not 2,880 independent
trajectories. Truth-frame and convention encodings are paired representations.
Each 40-to-80 Hz comparison pairs 2,880 noisy cases, reports 30 motion/noise/truth-
frame strata, and verifies all 463,680 common-time rows are identical (maximum
absolute difference 0). All 12 assessments record the same implementation digest,
`d276f1e50c2afe8aa48aa403551f2025e5e6ee280124e44a4a33c7e699855718`.

Recommendation coverage and paired relative-risk changes were:

| Noise | Motion family | Recommendations, 40 → 80 Hz | Paired log-risk shift, q10 / median / q90 | Unique → multiple | Nearest alternative changed | Evidence insufficient, 40 → 80 Hz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | Fourier | 2,880 → 2,208 | −0.9107 / −0.8105 / −0.7306 | 672 | 0 | 0 → 0 |
| Low | Piecewise-linear | 2,880 → 1,920 | −0.8772 / −0.7894 / −0.7196 | 960 | 0 | 0 → 0 |
| Low | Maneuver bursts | 0 → 0 | −0.6577 / −0.5266 / −0.3693 | 0 | 0 | 0 → 0 |
| High | Fourier | 480 → 0 | −0.7198 / −0.4452 / −0.3102 | 480 | 144 | 0 → 0 |
| High | Piecewise-linear | 0 → 0 | −0.6389 / −0.5432 / −0.2577 | 0 | 0 | 0 → 0 |
| High | Maneuver bursts | 0 → 0 | −0.1307 / −0.0833 / −0.0332 | 0 | 96 | 672 → 672 |

Every one of the 30 paired strata per family/tier had a lower median relative-risk
margin at 80 Hz. There were no data-winner misses or evidence misses. The high-noise
Fourier cohort lost its remaining 480 recommendations; piecewise and burst cohorts
were already ambiguous, with 672 burst cases remaining evidence-insufficient at
both rates. Nearest-alternative identities changed in 144 high-noise Fourier and
96 high-noise burst encodings; no low-noise identities changed. This sensitivity
means margin movement alone is not evidence of improved convention identification.

These results test known labels generated by this simulator, not physical sensors.
They neither establish field accuracy nor calibrate the margin, evidence threshold,
or confidence levels. Sensor-specific bias, calibration errors, timing defects,
and real noise distributions still need independent, labeled validation. Keep the
synthetic tiers as regression/behavioral coverage, not a substitute for such data.

Recreate the ignored corpora and assessments from the repository root with:

```bash
set -e

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) profile=sampled-rate-nonanalytic ;;
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for noise in low high; do
    for cadence in 40hz 80hz; do
      case "$cadence" in
        40hz) period=0.025; count=161 ;;
        80hz) period=0.0125; count=321 ;;
      esac
      root=".validation-corpus/cadence-noise-tier-replication-751310-312-20416115-119-${family}-${noise}-${cadence}"
      python -m scripts.generate_validation_corpus \
        --outdir "$root" --noise-profile "$noise" --noise-replicates 5 \
        --noise-seed-start 20416115 --motion-seeds 751310 751311 751312 \
        --noise-sample-counts "$count" --sample-period-s "$period" \
        --truth-convention-sweep --motion-profile "$profile" \
        --orientation-noise-correlation-time-s 0.05 \
        --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
      python -m scripts.validate_corpus "$root/manifest.json" --workers 8 \
        --checkpoint "$root/assessment-risk-margins.checkpoint.json" \
        --checkpoint-every 100 --progress \
        --output "$root/assessment-risk-margins.json"
    done
    root40=".validation-corpus/cadence-noise-tier-replication-751310-312-20416115-119-${family}-${noise}-40hz"
    root80=".validation-corpus/cadence-noise-tier-replication-751310-312-20416115-119-${family}-${noise}-80hz"
    python -m scripts.compare_cadence_assessments \
      "$root40/assessment-risk-margins.json" "$root80/assessment-risk-margins.json" \
      --baseline-manifest "$root40/manifest.json" \
      --candidate-manifest "$root80/manifest.json" \
      --output "$root40/comparison-risk-margins.json"
  done
done
```

Artifact SHA-256 values (paths are relative to the matching family/tier/cadence
corpus directory; comparison reports are in the 40 Hz directory):

```text
low fourier 40hz manifest.json                       f36d90ae13e84b4c3532b48b954cb5329cfb8ecd65d8423d199fba325b39f37a
low fourier 40hz assessment-risk-margins.json         723efd5ab76c6864d51d54c1b088bccb02650dade771be9c5838a0c4c1268a04
low fourier 80hz manifest.json                       bf3f6b2a56f5bcad524b2accb3564882e4855946c7e6fb144d339dc8925a7e5e
low fourier 80hz assessment-risk-margins.json         0373d6b334a57cc97d63d33e8b2bf3e87ac9925bd001783524191149aae8c233
low fourier comparison-risk-margins.json             48841821f8f9693ef288b687daf5c5f1780fdca1cba89fd678f6ed60b8fc1803
low piecewise 40hz manifest.json                     4a8fed04cb5927ec8ea90106cf70b86dd411cde13089608bc8c705f166976f7d
low piecewise 40hz assessment-risk-margins.json       478e39d3260748333e3b3d8b21f427c3706b534ef4c7cdf425d14af3cc839337
low piecewise 80hz manifest.json                     a6614d6adc252be423b7f90307dad2d1ba6df4d4599fc973674589c9c088e68a
low piecewise 80hz assessment-risk-margins.json       b98203df95c5ea75a5773b84fa18546ec8209023300c00efa6f59a81786adb1b
low piecewise comparison-risk-margins.json           5563f0dcfdbe7d774f71c3b4565dec1839eb53de7bb33d7fd0bf2dece32efed7
low bursts 40hz manifest.json                        67477625e734eebfbb5c54e575b33bb6c8e143d9cc7953a972c3a164262ba4d2
low bursts 40hz assessment-risk-margins.json          6234ba81f984fca06f6f8f96dc2ea0d297868c6c8d230b89a2567cf183e97375
low bursts 80hz manifest.json                        08b5ba077bd59c1f1cea5a45df008e892b1c13f0d0cb1fcfb1dbd0b79ea746a6
low bursts 80hz assessment-risk-margins.json          b377c630c0469fa0f7b94c3537cef86f3145f79887a7e04be6d4d4f9f2bed93f
low bursts comparison-risk-margins.json              dc801d85c114fdf39d2307e9b9b2d4fdbcded7ab8affe1d1ae11363dd3f352b9
high fourier 40hz manifest.json                      eec6af0c6275c0b4be5191f8dd9e55e7d840e4001ac5cc5480b23f1c43804dae
high fourier 40hz assessment-risk-margins.json        7d02876af7560c483ee28d66d72fd4e10c92ee830fed464ebd3d11ea9a1772f9
high fourier 80hz manifest.json                      418f580bdf9d022cbc38cfb3f89853be429a85a3e06903ddc380ecf30beed299
high fourier 80hz assessment-risk-margins.json        18879372e318becc08ef315bccf974df3cb0add77647a835154db87e9c712dfd
high fourier comparison-risk-margins.json            8be789913f1f657825724f39d5b4e74f9b191b7ee17db2b6d90d7e2db1bde984
high piecewise 40hz manifest.json                    9a59385f3aed8c74e554408c7d242cf2f01428cdeaaf9311e6afaeba8dcb22dc
high piecewise 40hz assessment-risk-margins.json      8bc3cf0aa36b748d4880712c57db11ec0aa342ad7de9206ff91d0e50a9b77b7c
high piecewise 80hz manifest.json                    3ad6f1b9a750c83a101f1bd39ac7876e45118688f72e9ff1e20a118dc30d3e95
high piecewise 80hz assessment-risk-margins.json      aecec1f33a0573c0a26b03138f4595021ab2ea03650176122d162de2c74ea615
high piecewise comparison-risk-margins.json          581951692267725a358652b17036758294975dd4750afef5b90c08233096a3c5
high bursts 40hz manifest.json                       14fbded4181e2a7d551dcbd042483aebba7eef783f5911af269bed2a9accd46f
high bursts 40hz assessment-risk-margins.json         a71a3af9f0e7fc8ce55e24c8febf0823b8f6e5f161ff0b941e73ea1acab07047
high bursts 80hz manifest.json                       9ae0405816229f8a384dd1179597ee73b7c98d3895d92e2d5ef968e56b283612
high bursts 80hz assessment-risk-margins.json         b56f14c7c553b36995b6afe021491134e3546498bfd72c272ee8245632fee62a
high bursts comparison-risk-margins.json             8a420e2f17e733aeb94c7b296188ee29d03f259118ceebfd77562f0585178ee4
```

### Paired cadence factorial with jitter and dropout

To test whether the cadence-associated margin shift changes under acquisition
defects, a fresh full-convention factorial crossed 40/80 Hz with timestamp jitter
(`0` or `2 ms` maximum amplitude) and whole-row dropout (`0` or `10%`). It covered
the clean control, each defect alone, and their combination. All six family/cadence
corpora used moderate OU noise (`0.003 rad` orientation and `0.025 rad/s` rate,
with 50/100 ms correlation times) and body-frame rate noise. Motion seeds
`751320`--`751322` and noise seeds `20416120`--`20416124` were fresh relative to the
prior recorded blocks and reused across families, acquisition conditions, and
cadences.

Each family/cadence manifest contains 11,520 noisy cases plus three fixed controls:
three motion seeds × five noise seeds × four acquisition conditions × 192 convention
encodings. That is 15 motion/noise realizations per family/condition; each of the
30 motion/noise/truth-frame strata per condition contains 96 paired convention
encodings. These are simulator realizations and paired representations, not
independent field trials.

The generator now draws jitter and dropout on the shared 0.5 ms acquisition grid
before decimating to cadence. Therefore the jitter timestamps and keep/drop flags
at common nominal sample times are identical across 40 and 80 Hz. The cadence
comparator accepts labeled jitter/dropout only after checking every common-time
measurement row, while also checking declared jitter bounds, observed/dropped row
counts, nested timestamps, and exact seeded case pairing. All three schema-v4
reports paired 11,520 noisy cases, covered 1,768,320 shared-time rows, and found a
maximum common-input difference of zero. Each report has 120 risk-margin strata:
30 per acquisition condition.

Recommendations and nearest-alternative log-risk shifts were:

| Motion family | Acquisition condition | Recommendations, 40 → 80 Hz | Paired log-risk shift, q10 / median / q90 | Unique → multiple | Nearest alternative changed | Evidence insufficient, 40 → 80 Hz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fourier | Clean | 1,920 → 1,920 | −0.8622 / −0.8080 / −0.6029 | 0 | 0 | 0 → 0 |
| Fourier | 2 ms jitter | 1,920 → 1,920 | −0.8602 / −0.8088 / −0.6014 | 0 | 0 | 0 → 0 |
| Fourier | 10% dropout | 1,920 → 1,920 | −0.8772 / −0.8132 / −0.6088 | 0 | 0 | 0 → 0 |
| Fourier | Jitter + dropout | 1,920 → 1,920 | −0.8774 / −0.8108 / −0.6084 | 0 | 0 | 0 → 0 |
| Piecewise-linear | Clean | 1,920 → 1,920 | −0.8475 / −0.7974 / −0.5269 | 0 | 0 | 0 → 0 |
| Piecewise-linear | 2 ms jitter | 1,920 → 1,920 | −0.8487 / −0.7975 / −0.5259 | 0 | 0 | 0 → 0 |
| Piecewise-linear | 10% dropout | 1,920 → 1,920 | −0.8661 / −0.7944 / −0.5364 | 0 | 0 | 0 → 0 |
| Piecewise-linear | Jitter + dropout | 1,920 → 1,920 | −0.8655 / −0.7951 / −0.5361 | 0 | 0 | 0 → 0 |
| Maneuver bursts | Clean | 0 → 0 | −0.2858 / −0.1936 / −0.1185 | 0 | 0 | 0 → 0 |
| Maneuver bursts | 2 ms jitter | 0 → 0 | −0.2845 / −0.1949 / −0.1188 | 0 | 0 | 0 → 0 |
| Maneuver bursts | 10% dropout | 0 → 0 | −0.3155 / −0.2176 / −0.1416 | 0 | 0 | 0 → 0 |
| Maneuver bursts | Jitter + dropout | 0 → 0 | −0.3121 / −0.2190 / −0.1438 | 0 | 0 | 0 → 0 |

All 30 paired risk-margin medians moved downward in every family/condition cell.
There were no data-winner misses, evidence misses, evidence-insufficient cases,
recommendation-status changes, or recommended-class changes. Fourier and piecewise
each recommended 1,920/2,880 convention encodings at both rates in every condition;
the burst family remained ambiguous with zero recommendations. Its evidence set
expanded at 80 Hz in all 11,520 noisy cases, but that did not change the decision.
The 40 Hz dropout corpora each omitted 43,200 rows per family for the dropout-only
and combined conditions (15 rows per noisy case on average, with trajectory
endpoints protected).

This controlled block suggests 2 ms timestamp jitter and 10% dropout did not alter
recommendation counts in these moderate-noise synthetic cohorts, while relative
risk margins still narrowed at the denser cadence. That is not evidence that the
acquisition defects are harmless generally: the study has only 15 distinct
motion/noise pairs per family/condition and tests one noise tier and selected motion
families. No scoring, threshold, or confidence-policy changes follow from it. Real
logger characteristics and independently labeled trajectories remain necessary
for field claims.

Recreate the ignored corpora, assessments, and cadence comparisons with:

```bash
set -e

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) profile=sampled-rate-nonanalytic ;;
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/cadence-acquisition-factorial-751320-322-20416120-124-${family}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 5 \
      --noise-seed-start 20416120 --motion-seeds 751320 751321 751322 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --timestamp-jitter-amplitudes-s 0 0.002 --dropout-probabilities 0 0.1 \
      --truth-convention-sweep --motion-profile "$profile" \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 8 \
      --checkpoint "$root/assessment-factorial.checkpoint.json" \
      --checkpoint-every 200 --output "$root/assessment-factorial.json"
  done
  root40=".validation-corpus/cadence-acquisition-factorial-751320-322-20416120-124-${family}-40hz"
  root80=".validation-corpus/cadence-acquisition-factorial-751320-322-20416120-124-${family}-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-factorial.json" "$root80/assessment-factorial.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-factorial-verified.json"
done
```

The implementation digest recorded by all six assessments is
`d276f1e50c2afe8aa48aa403551f2025e5e6ee280124e44a4a33c7e699855718`. Source
SHA-256 values:

```text
scripts/generate_validation_corpus.py  a4e718ff146b64e73d25c40fc3ccf8538dcdb78a1cdf4415eb8294c25380c4bb
scripts/validation_noise.py            41ae49ab4de6c662abe37c59d6eca624cf7e89a7113975e63127c7be88ffad4f
scripts/compare_cadence_assessments.py 0a62fb9e4dffe97d38ce505c2a232f06da6323295c66118217969a8b9ebcdb80
scripts/validate_corpus.py             d1c337106197bdd4a718f7493287dc3d8188ad435949e73c080685527980f275
```

Artifact SHA-256 values (paths are relative to each family/cadence corpus; each
comparison report is in its family's 40 Hz directory):

```text
fourier 40hz manifest.json                    b7568feb5d1a50e05052f273f927d51deb6c37d65a6bae6747e8e37d33a8d427
fourier 40hz assessment-factorial.json        cd0357517bc06775bf927862785ac06c90c055ef168384a17ba255fadc2c12e9
fourier 80hz manifest.json                    758fe4dd72ed1db14803fb4bf3eb3c29f6abd15fd49cf55982652755bd16db89
fourier 80hz assessment-factorial.json        7cea2c38561824631684f4a845245198ecb7df8c996efd759c3dcbc6b4754d1f
fourier comparison-factorial-verified.json   3c27b380c6ab7b276e067026df6ba23cefbfa1c8ac1ff764061f6eec7a3072f4
piecewise 40hz manifest.json                 b61244de017fc2957ac3bba8b5efe414e2e8a13ed44ae2498b330317e0ac2704
piecewise 40hz assessment-factorial.json     232fbbe5f72bfbb2dea46d3b5cfcb7b90d46d214cabd8a9891695cfbe7632cd3
piecewise 80hz manifest.json                 8b0c07d37ed57ee758a94690a446536dca45a19a7b188bb044096d110de195e2
piecewise 80hz assessment-factorial.json     c2f898010d80ae8fb0a135ecd7f8bcd44923938d2f7813a736c9ec867fe2d50c
piecewise comparison-factorial-verified.json db974b570e3dd660ee2ca8d4ea40362ab88ca98206cf95ff6b90f648946398a1
bursts 40hz manifest.json                    fe3f5aa3b84b1b796c605b9574e0299374bb02e0fee41809f0122b6b4ce927b9
bursts 40hz assessment-factorial.json        8a7e02f0786a5ab482048da8f2d454c560242899757fc76a80286c15efea6153
bursts 80hz manifest.json                    aa08785dcc6b2576e3f439dc5dca86028d8ab17d13c3d480960ab5929d0f71a4
bursts 80hz assessment-factorial.json        e8da5c1d9acc8be862e0e311db8464804db88b609ce96323446ed75754c54885
bursts comparison-factorial-verified.json    122973918c071f991ad0bb9d30eb2ba3b9f1260a4bee5276c0aabc06dbd10c29
```

### Paired cadence factorial with channel latency

A fresh moderate-noise full-convention factorial paired 40 and 80 Hz for Fourier,
piecewise-linear, and maneuver-burst rates. It crossed the four combinations of
`0/13 ms` orientation latency and `0/27 ms` rate latency, with zero timestamp
jitter and row dropout. Positive channel latency samples that channel at physical
time `t - latency`. The study used motion seeds `751330`--`751332` and noise
seeds `20416130`--`20416134`, distinct from the earlier cadence blocks.
Orientation/rate OU noise was `0.003 rad` / `0.025 rad/s`, with `50/100 ms`
correlation times and body-frame rate noise.

Each family/cadence manifest contains 11,520 noisy cases plus three fixed
controls. There are 15 motion/noise pairs per family/latency condition and 30
motion-seed/noise-seed/truth-frame strata; each stratum contains 96 paired
convention encodings. These encodings are paired representations, not independent
trials. The schema-v5 comparator checked all 11,520 noisy pairs and 1,854,720
common-time rows per family. The maximum shared-input difference was exactly
zero, and each report has 120 risk-margin strata (30 for each latency pair).

Recommendations and the distribution of the 30 within-stratum median margin
shifts (80 Hz minus 40 Hz) were:

| Motion family | Orientation / rate latency | Recommendations, 40 → 80 Hz | q10 / median / q90 of 30 stratum-median shifts |
| --- | ---: | ---: | ---: |
| Fourier | 0 / 0 ms | 2,496 → 1,800 | −0.8618 / −0.7934 / −0.6726 |
| Fourier | 0 / 27 ms | 1,896 → 1,800 | −0.7675 / −0.6562 / −0.5710 |
| Fourier | 13 / 0 ms | 2,136 → 1,800 | −0.8215 / −0.7589 / −0.6394 |
| Fourier | 13 / 27 ms | 2,064 → 1,800 | −0.8195 / −0.7588 / −0.6414 |
| Piecewise-linear | 0 / 0 ms | 2,880 → 1,920 | −0.8738 / −0.7895 / −0.6925 |
| Piecewise-linear | 0 / 27 ms | 1,920 → 1,920 | −0.6285 / −0.5703 / −0.4752 |
| Piecewise-linear | 13 / 0 ms | 2,784 → 1,920 | −0.7910 / −0.7286 / −0.6324 |
| Piecewise-linear | 13 / 27 ms | 2,592 → 1,920 | −0.7996 / −0.7155 / −0.6192 |
| Maneuver bursts | 0 / 0 ms | 0 → 0 | −0.3236 / −0.2590 / −0.0526 |
| Maneuver bursts | 0 / 27 ms | 0 → 0 | −0.2705 / −0.1813 / −0.0406 |
| Maneuver bursts | 13 / 0 ms | 0 → 0 | −0.3035 / −0.2451 / −0.0555 |
| Maneuver bursts | 13 / 27 ms | 0 → 0 | −0.3045 / −0.2273 / −0.0479 |

All 30 stratum median shifts were negative in every family/latency cell. Across
Fourier and piecewise cases, recommendations fell by 1,392 and 2,496,
respectively, with no false recommendations, data-winner misses, evidence misses,
or evidence-insufficient cases. The burst family remained ambiguous: it had no
recommendations at either rate, and evidence insufficiency increased from 1,056
to 1,152 cases. Evidence sets only expanded at 80 Hz (3,576 Fourier, 2,856
piecewise, and 11,520 burst cases); none contracted or were replaced. The
latency-specific recommendation counts varied with motion and noise seed, so these
results do not isolate a general latency effect or establish logger robustness.
They show that this cadence-associated margin shift persisted under each tested
latency pair. Do not tune thresholds from this synthetic block.

Recreate the ignored corpora, assessments, and comparisons with fresh output
directories:

```bash
set -e

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) profile=sampled-rate-nonanalytic ;;
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/cadence-channel-latency-factorial-751330-332-20416130-134-${family}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 5 \
      --noise-seed-start 20416130 --motion-seeds 751330 751331 751332 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$profile" \
      --orientation-latencies-s 0 0.013 --rate-latencies-s 0 0.027 \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 8 \
      --checkpoint "$root/assessment-factorial.checkpoint.json" \
      --checkpoint-every 200 --output "$root/assessment-factorial.json"
  done
  root40=".validation-corpus/cadence-channel-latency-factorial-751330-332-20416130-134-${family}-40hz"
  root80=".validation-corpus/cadence-channel-latency-factorial-751330-332-20416130-134-${family}-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-factorial.json" "$root80/assessment-factorial.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-factorial-v5.json"
done
```

The implementation digest in all six assessments is
`d276f1e50c2afe8aa48aa403551f2025e5e6ee280124e44a4a33c7e699855718`. Source
SHA-256 values:

```text
scripts/generate_validation_corpus.py  a4e718ff146b64e73d25c40fc3ccf8538dcdb78a1cdf4415eb8294c25380c4bb
scripts/validation_noise.py            41ae49ab4de6c662abe37c59d6eca624cf7e89a7113975e63127c7be88ffad4f
scripts/compare_cadence_assessments.py 4feaf22ffa98cdb41704d9b9681b73fc286874962b4f35fcafc319245f971e10
scripts/validate_corpus.py             d1c337106197bdd4a718f7493287dc3d8188ad435949e73c080685527980f275
```

Artifact SHA-256 values (paths are relative to each family/cadence corpus; each
comparison report is in its family's 40 Hz directory):

```text
fourier 40hz manifest.json                    9f824caa4974a5be423658d82bcc3fbcfead3685260f619851d21fee30dac1d7
fourier 40hz assessment-factorial.json        f83cf5b0ca40515dfcbc65d1c330609785385973a96ee66f04e7f9b8f4b2b236
fourier 80hz manifest.json                    8623d97e0f2ed7b3e4b18823158015bdcdf0b7c690328d73b8736df3c939d665
fourier 80hz assessment-factorial.json         73e273e1b495e21fbcda66ead273c0061ad66b88be071cdef3ad0dd4e253b43b
fourier comparison-factorial-v5.json           9642d48d63f7360fe710fc2d4ad0ee45da20576a3d11ddd3a32993cae2fb717d
piecewise 40hz manifest.json                 d5aabd42b3d58561f9cb5bee0a8117a4298eed9c9afe19a9a9fda472de7dea8d
piecewise 40hz assessment-factorial.json     9b726daac4671a8062760f82710ac9425072aa6341782410511f7a9835f11e36
piecewise 80hz manifest.json                 89524b8e6737ce5b53881cd75672d1cb06ecd441558537623c45b729be46bb99
piecewise 80hz assessment-factorial.json     0bd8cd5316a537dcd40f4cff07664c497289d120dc6dbe08c9e8c4439a6f6b3f
piecewise comparison-factorial-v5.json      c0c9735d7161b5b866069d68611133e0cd315f3657afbfd7bf57a2918328f215
bursts 40hz manifest.json                    c6ce990990fda567ec9edfb211bc91efa334f385ef2403dbf30ab0b3654b4725
bursts 40hz assessment-factorial.json       b7754ab77f0d4432b2377e8465e2dc1d9a3d9a3cc842198781a7cf4440f85821
bursts 80hz manifest.json                    2400f18c0ead8603099f6cb457d676128daf410a0dc4bed54b1439691a9f67c1
bursts 80hz assessment-factorial.json        7947f2f685f48878837d0f3a1df4983cf1feabe1f4a654ccc0dfa588e7dc955a
bursts comparison-factorial-v5.json          eafd9ae6d9950416070b14fb9eeac698e782ee046a58b55d05a1aa70254bb6a9
```

### Paired cadence factorial with rate-sensor calibration defects

This synthetic screen extends the paired 40/80 Hz comparison across four
rate-channel conditions: identity, a 15-degree right-handed axis misalignment,
a 150 mrad/s body-`p` bias, and per-axis scale `(1.3, 0.8, 1.1)`. The
misalignment axis is normalized from `(0.3, -0.5, 0.8)`. Calibration is applied
to the physical angular-rate vector
in its declared body or world frame, before axis-order, sign, and unit encoding.
The transform is rotation → bias → scale → polarity → additive measurement
noise → clipping. Polarity and clipping are supported by the typed calibration
contract but are not crossed in this first factorial. These are controlled
synthetic channel defects, not claims about a real sensor mount or measured
instrument error.

The design uses three sampled-rate motion families, three new motion seeds,
five new noise seeds, the full 192-convention sweep, and moderate stationary OU
noise (`0.003 rad` orientation, `0.025 rad/s` rate; `50/100 ms` correlation
times). Rate noise is body-frame. Jitter, dropout, channel latency, and clipping
are zero. The 40 and 80 Hz logs each span four seconds and reuse the same
continuous-time motion/noise paths. Each family/cadence corpus therefore has
11,520 noisy cases plus three controls; one paired family comparison checks
11,520 noisy pairs and 1,854,720 common-time rows. There are 120
motion-seed/noise-seed/truth-frame/calibration strata per family, with 96
paired convention encodings per stratum. These encodings are not independent
replicates.

The manifests use schema 5, checkpoints schema 9, assessments schema 10, and
the cadence comparison report schema 6. The comparer requires identical
calibration metadata and verifies equality of every shared timestamp and
measurement row before reporting outcomes. Do not tune inference thresholds
from this synthetic screen, and do not treat it as held-out field validation.

#### Results

All six assessments were evaluated on 2026-11-01 with implementation
fingerprint `3a4067cd402bef00bfd1239715d42f342e8ceb614cf8512d693410658357d6b8`
(Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
`0.0.1a0.post1.dev25+gc2594b4a8.d20261101`). Each family comparison passed the
implementation/configuration and shared-input gates: 11,520 noisy pairs plus
three controls, 1,854,720 common-time rows, and zero maximum absolute difference
in shared measurements.

The table reports recommendation counts among 2,880 noisy cases per
family/calibration at 40 Hz → 80 Hz. The risk-margin column gives q10 / median /
q90 of the 30 seed-and-truth-frame stratum-level median candidate-minus-
baseline shifts. These are descriptive stratum quantiles, not confidence
intervals; the 96 convention encodings within a stratum are not independent
replicates.

| Motion family | Calibration | Recommendations, 40 → 80 Hz | Risk-margin shift q10 / median / q90 | Negative strata |
| --- | --- | ---: | ---: | ---: |
| Fourier | identity | 2,640 → 1,056 | −0.888 / −0.814 / −0.735 | 30/30 |
| Fourier | 15° misalignment | 1,536 → 672 | −0.452 / −0.233 / −0.115 | 30/30 |
| Fourier | 150 mrad/s p-bias | 960 → 888 | −0.259 / −0.211 / −0.111 | 30/30 |
| Fourier | axis scale `(1.3, 0.8, 1.1)` | 1,200 → 864 | −0.325 / −0.242 / −0.213 | 30/30 |
| Piecewise | identity | 1,920 → 1,920 | −0.889 / −0.794 / −0.693 | 30/30 |
| Piecewise | 15° misalignment | 480 → 480 | −0.326 / −0.216 / −0.121 | 30/30 |
| Piecewise | 150 mrad/s p-bias | 768 → 96 | −0.238 / −0.171 / −0.119 | 30/30 |
| Piecewise | axis scale `(1.3, 0.8, 1.1)` | 672 → 0 | −0.244 / −0.219 / −0.193 | 30/30 |
| Maneuver bursts | identity | 0 → 0 | −0.318 / −0.202 / −0.020 | 30/30 |
| Maneuver bursts | 15° misalignment | 0 → 0 | −0.174 / −0.074 / −0.026 | 30/30 |
| Maneuver bursts | 150 mrad/s p-bias | 0 → 0 | −0.050 / −0.015 / −0.004 | 29/30 |
| Maneuver bursts | axis scale `(1.3, 0.8, 1.1)` | 0 → 0 | −0.198 / −0.072 / −0.016 | 30/30 |

Across the 11,520 noisy cases per family, Fourier recommendations changed
6,336 → 3,480 and piecewise recommendations 3,840 → 2,496; maneuver bursts
remained at 0 → 0. There were no candidate evidence misses, incorrect
evidence-sufficient outcomes, or recommendation misses. Fourier and piecewise
also had zero data-winner or operational misses at either cadence. Maneuver
bursts had 672 → 576 data-winner/operational misses, despite remaining
recommendation-free. The consistently negative risk-margin shifts therefore
must not be interpreted alone as recommendation quality; their effect on
coverage depends on motion family and calibration.

#### Reproducibility fingerprints

The source hashes identify the exact implementation used for the study. The
artifact hashes identify the exact ignored local manifests, assessments, and
comparison reports; the fresh-generation commands below will create new local
artifacts. Artifact paths below are relative to the common prefix
`.validation-corpus/cadence-calibration-factorial-751340-342-20416140-144-`.

```text
implementation_sha256  3a4067cd402bef00bfd1239715d42f342e8ceb614cf8512d693410658357d6b8

cdfc7650dad112df95ba2f7eecc38640c806e60540646338395ee2bb733fabe1  scripts/generate_validation_corpus.py
6a003169a2faa11c2570e93df00daa16e7efd8a0030b3eba30e64cbca4cfa36b  scripts/rate_sensor_calibration.py
2f84cd180cbd6654016398b46a665a5ba52709a27e273664eb7ff5e40195a922  scripts/validate_corpus.py
f10829534dd218506df4ab181b655cf59d2c39eedd4a9d5ea45af59bda87fb79  scripts/compare_cadence_assessments.py
41ae49ab4de6c662abe37c59d6eca624cf7e89a7113975e63127c7be88ffad4f  scripts/validation_noise.py

dbf103b43c2aaa97813c9130a75f1873a0e5e1f9683c932f15b5e522f2639adc  fourier-40hz/manifest.json
b9581eebf31c977c4efcc65b14378726fd5375c0db8511e169205147da671bab  fourier-40hz/assessment-calibration-current.json
6641e05313f69a3e4eae3f000e7c73b1b737a5ed23ffbd1fc977cae7ae886ba8  fourier-80hz/manifest.json
c24bd6d3cd2f31abdb85e3bb36cfe1ca339da4ff462e81c414bf1d1f5889c316  fourier-80hz/assessment-calibration-current.json
a71a065f31a2481e18551dfd1b71ad8a5c47926ac48a63ce378c3ef6c63ba523  fourier-comparison-calibration-current-v6.json
59fe5c1218e758730af465d58974bfbc4a72fc82f92fa8933e4c1266c79c62f2  piecewise-40hz/manifest.json
cdd0447e7fff81bb5593bd6a53093ad2c9739cc65c639c763ac3e03f20c96931  piecewise-40hz/assessment-calibration-current.json
e26c499028517fe37aae968832edddb40d366f179187a58b990c3510c1d097c0  piecewise-80hz/manifest.json
69537c09cd77a61a9376d066e67a2a8cb8891a488356c9e9f9dcf6c83c5b61bc  piecewise-80hz/assessment-calibration-current.json
28b1bbd0bb5d446cb1df2043718ccbcc9a3805eb2a77fda6d4ee6b12b68d924d  piecewise-comparison-calibration-current-v6.json
4b4c7e5580ef7e5015ccfa92b640500f1525effc486dd9094cb1ce13fe39ecbe  bursts-40hz/manifest.json
e291d3560dba67c8006ce25548d75dc5a45de87fd8382461015d16efb85f3fcf  bursts-40hz/assessment-calibration-current.json
dda4a80cae6d3d74a38ba9dc3b48bd40e309f03cfc28e7ca127f7d082f83b9e9  bursts-80hz/manifest.json
908173541b7dc7094b178c323fc0d39b4287f5aa759e632df629095e2c1f530b  bursts-80hz/assessment-calibration.json
0618ea0341e454575b201ed6d2f0f2d0cad1a61bdab3ec563eb2eb707632cb24  bursts-comparison-calibration-current-v6.json
```

Recreate the ignored corpora and reports with fresh output directories:

```bash
set -e

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) profile=sampled-rate-nonanalytic ;;
    piecewise) profile=sampled-rate-piecewise-linear ;;
    bursts) profile=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root=".validation-corpus/cadence-calibration-factorial-751340-342-20416140-144-${family}-${cadence}"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 5 \
      --noise-seed-start 20416140 --motion-seeds 751340 751341 751342 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$profile" \
      --rate-sensor-calibrations identity misalignment-15deg bias-p-150mrad-s \
        axis-scale-1p3-0p8-1p1 \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 32 \
      --checkpoint "$root/assessment-calibration-current.checkpoint.json" \
      --checkpoint-every 200 --output "$root/assessment-calibration-current.json"
  done
  root40=".validation-corpus/cadence-calibration-factorial-751340-342-20416140-144-${family}-40hz"
  root80=".validation-corpus/cadence-calibration-factorial-751340-342-20416140-144-${family}-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-calibration-current.json" \
    "$root80/assessment-calibration-current.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-calibration-current-v6.json"
done
```

### Paired cadence factorial with rate-sensor polarity and clipping defects

This follow-up crosses per-axis scale, a `p`-axis polarity reversal, and
absolute clipping in a balanced synthetic `2 × 2 × 2` design. The eight
profiles are `identity`; `polarity-p-flip`; `clip-0p3-rad-s`;
`polarity-p-flip-clip-0p3-rad-s`; `axis-scale-1p3-0p8-1p1`;
`axis-scale-1p3-0p8-1p1-polarity-p-flip`;
`axis-scale-1p3-0p8-1p1-clip-0p3-rad-s`; and
`axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s`. The factors are
identity versus scale `(1.3, 0.8, 1.1)`, polarity `(1, 1, 1)` versus
`(-1, 1, 1)`, and no clipping versus component-wise clipping to
`±0.3 rad/s`. The clipping value is a deliberately binding stress setting,
not a measured sensor limit or a recommended operating range. Calibration,
additive noise, and clipping follow the documented transform order; clipping
is applied after additive measurement noise.

The paired 40/80 Hz screen covers Fourier, piecewise-linear, and maneuver-burst
sampled-rate motion; three fresh motion seeds (`751343`–`751345`); five fresh
noise seeds (`20416145`–`20416149`); both truth rate frames; and the complete
192-encoding convention sweep. It uses four-second trajectories, moderate
stationary OU noise (`0.003 rad` orientation and `0.025 rad/s` rate, with
`50/100 ms` correlation times), and body-frame rate noise. Jitter, dropout,
and channel latency are zero. The paired cadences share continuous-time motion
and noise paths. Each family/cadence manifest contains 23,040 noisy cases plus
three controls: 8 calibration cells × 3 motion seeds × 5 noise seeds × 2 truth
frames × 96 convention encodings per stratum. Each family comparison checks
23,040 noisy pairs and 3,709,440 common-time measurement rows. There are 30
seed/truth-frame strata per calibration cell; the 96 encodings within a
stratum are paired variants, not independent replicates.

The manifests use schema 5, assessments schema 10, and cadence-comparison
reports schema 6. All six assessments share implementation fingerprint
`3a4067cd402bef00bfd1239715d42f342e8ceb614cf8512d693410658357d6b8`
(Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0,
`0.0.1a0.post1.dev25+gc2594b4a8.d20261101`). All three paired comparisons
passed implementation/configuration gates and had exact shared-input equality
(maximum absolute difference `0.0`).

#### Results

The recommendation column is the count among 2,880 noisy cases per
family/calibration at 40 Hz → 80 Hz. The margin column is q10 / median / q90
of the 30 seed-and-truth-frame stratum-level median candidate-minus-baseline
log-risk shifts, using linear quantiles. These are descriptive paired strata,
not confidence intervals.

| Motion family | Calibration | Recommendations, 40 → 80 Hz | Margin shift q10 / median / q90 | Negative strata |
| --- | --- | ---: | ---: | ---: |
| Fourier | identity | 1,920 → 1,920 | −0.854 / −0.824 / −0.407 | 30/30 |
| Fourier | polarity-p-flip | 552 → 480 | −0.165 / −0.011 / −0.002 | 29/30 |
| Fourier | clip-0p3-rad-s | 0 → 0 | −0.331 / −0.022 / −0.010 | 30/30 |
| Fourier | polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.162 / −0.008 / −0.001 | 29/30 |
| Fourier | axis-scale-1p3-0p8-1p1 | 1,320 → 480 | −0.205 / −0.188 / −0.126 | 30/30 |
| Fourier | axis-scale-1p3-0p8-1p1-polarity-p-flip | 456 → 312 | −0.092 / −0.026 / 0.001 | 24/30 |
| Fourier | axis-scale-1p3-0p8-1p1-clip-0p3-rad-s | 0 → 0 | −0.239 / −0.022 / −0.009 | 30/30 |
| Fourier | axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.128 / −0.008 / 0.001 | 25/30 |
| Piecewise | identity | 1,920 → 960 | −0.813 / −0.760 / −0.616 | 30/30 |
| Piecewise | polarity-p-flip | 720 → 720 | −0.092 / −0.032 / −0.001 | 27/30 |
| Piecewise | clip-0p3-rad-s | 0 → 0 | −0.138 / −0.039 / −0.018 | 30/30 |
| Piecewise | polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.045 / −0.016 / −0.004 | 30/30 |
| Piecewise | axis-scale-1p3-0p8-1p1 | 0 → 0 | −0.290 / −0.216 / −0.182 | 30/30 |
| Piecewise | axis-scale-1p3-0p8-1p1-polarity-p-flip | 600 → 480 | −0.057 / −0.024 / −0.014 | 30/30 |
| Piecewise | axis-scale-1p3-0p8-1p1-clip-0p3-rad-s | 0 → 0 | −0.127 / −0.040 / −0.017 | 30/30 |
| Piecewise | axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.066 / −0.017 / −0.003 | 30/30 |
| Maneuver bursts | identity | 0 → 0 | −0.163 / −0.081 / −0.044 | 30/30 |
| Maneuver bursts | polarity-p-flip | 72 → 0 | −0.318 / −0.284 / −0.226 | 30/30 |
| Maneuver bursts | clip-0p3-rad-s | 0 → 0 | −0.102 / −0.048 / −0.012 | 30/30 |
| Maneuver bursts | polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.305 / −0.252 / −0.156 | 30/30 |
| Maneuver bursts | axis-scale-1p3-0p8-1p1 | 0 → 0 | −0.151 / −0.074 / −0.015 | 30/30 |
| Maneuver bursts | axis-scale-1p3-0p8-1p1-polarity-p-flip | 0 → 0 | −0.256 / −0.210 / −0.104 | 30/30 |
| Maneuver bursts | axis-scale-1p3-0p8-1p1-clip-0p3-rad-s | 0 → 0 | −0.116 / −0.067 / −0.015 | 29/30 |
| Maneuver bursts | axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s | 0 → 0 | −0.273 / −0.222 / −0.084 | 30/30 |

Across all three families, recommendations changed from 7,560 to 5,352 of
69,120 noisy paired cases. Fourier changed 4,248 → 3,192, piecewise motion
3,240 → 2,160, and maneuver bursts 72 → 0. The clipped profiles produced no
recommendations at either cadence in any family. Recommendation misses
(incorrect recommendations) changed 1,008 → 792 for Fourier, 1,320 → 1,200
for piecewise motion, and 72 → 0 for bursts. Raw data-winner and operational
misses were 11,112 → 11,112, 11,280 → 11,280, and 11,208 → 11,256,
respectively. Evidence misses changed 3,696 → 3,120, 2,400 → 2,136, and
1,440 → 48. In this report, an “incorrect evidence-sufficient” outcome is an
evidence miss for which the policy nevertheless marked evidence sufficient;
its counts were equal to the evidence-miss counts in these cohorts.

The paired median risk-margin shift was negative in 703/720 strata. That
relative-fit movement did not imply better recovery: polarity-flip cells
contain large data-winner and evidence misses, and recommendation errors
remain in the Fourier and piecewise cohorts. Clipping at this intentionally
strong synthetic level removes recommendation coverage entirely. Do not infer
sensor tolerances, expected field performance, or a threshold-tuning direction
from this screen. It identifies an important productization gap: recovery
under unknown rate-channel polarity and saturation needs further designed
tests and independently labeled acquisition logs before making reliability
claims.

#### Reproducibility fingerprints

The artifact hashes refer to ignored local files, not checked-in corpora.
Paths are relative to
`.validation-corpus/cadence-polarity-clip-factorial-751343-345-20416145-149-`.
The implementation fingerprint above records the validator and package code
used for these assessments; source-file hashes below identify the study tools
as they existed when the assessments were generated.

```text
implementation_sha256  3a4067cd402bef00bfd1239715d42f342e8ceb614cf8512d693410658357d6b8

cdfc7650dad112df95ba2f7eecc38640c806e60540646338395ee2bb733fabe1  scripts/generate_validation_corpus.py
04779e221d3a5e157b3ed77e883c7f618b9b51995607eab3c07104eabdf26772  scripts/rate_sensor_calibration.py
2f84cd180cbd6654016398b46a665a5ba52709a27e273664eb7ff5e40195a922  scripts/validate_corpus.py
f10829534dd218506df4ab181b655cf59d2c39eedd4a9d5ea45af59bda87fb79  scripts/compare_cadence_assessments.py
41ae49ab4de6c662abe37c59d6eca624cf7e89a7113975e63127c7be88ffad4f  scripts/validation_noise.py

4293d9bde33de4b806e21c47c9c768f2e4b6daffb336679c9333fdee6ebdb519  fourier-40hz/manifest.json
4f8d7c3d3f9814a1718874fecbaa69a094a4ff9db03e9b23dc910d9429b1ee83  fourier-40hz/assessment-polarity-clipping-current.json
d089efa491e3407edd841a36858c0c94c3796b9e0f2e510e718af2f6817d8c3f  fourier-80hz/manifest.json
b9e1cad43756c6d42ffcb8e8d7e459a797c4ddd0c0616e2a8c5a3cfa95a1901e  fourier-80hz/assessment-polarity-clipping-current.json
57ce118d93f8ed62fb129bbeefa9d61cc6fb774b867360d0b710a1ec7e9773ed  fourier-40hz/comparison-polarity-clipping-current-v6.json
8b32852e659ddc4d1da1557cc7e0ccc9c87071a0bb591bc680aa36d5bb0f9f1f  piecewise-40hz/manifest.json
d975e3fb97a6bd46fad0b4e5d7797d02b94d6a8ce5eb2e3fdc8b564941d8bfa6  piecewise-40hz/assessment-polarity-clipping-current.json
7d37f2012a8dba584f2e59d3b78ea4fa82d1f297f943455f5960470509765f3c  piecewise-80hz/manifest.json
0695fdb17758f7f138887393958b6e56ef13a47c83dc2cbacb3c5923abb7ccae  piecewise-80hz/assessment-polarity-clipping-current.json
5d76460d7ceaff433c10a3bebf6f16a1c1c63b6f126ae4a3c951e460e22bc1cc  piecewise-40hz/comparison-polarity-clipping-current-v6.json
4e6e1b93936e7ffcce58fa37e2ea44f90e08652d2a0467a3f23a486318e2a8b9  bursts-40hz/manifest.json
8fc42093023bbd945233d03c78e0d2667603eacc1a27e6502d029bd8fef22899  bursts-40hz/assessment-polarity-clipping-current.json
b4fafb09929c4da1c59f22c5c3056354513927b30dfc676a521382c292120b80  bursts-80hz/manifest.json
af7030bd2d83864df0810d025ff80a302b473aaca18e5a69ee2dd6e824075cc9  bursts-80hz/assessment-polarity-clipping-current.json
dc86cd6f7f4843a69b4977a0675d356a5bf13cb2731d061f224666694c1d8ca9  bursts-40hz/comparison-polarity-clipping-current-v6.json
```

Regenerate the local corpora, assessments, and comparisons with:

```bash
set -euo pipefail
prefix=".validation-corpus/cadence-polarity-clip-factorial-751343-345-20416145-149"
profiles="identity polarity-p-flip clip-0p3-rad-s polarity-p-flip-clip-0p3-rad-s"
profiles="$profiles axis-scale-1p3-0p8-1p1"
profiles="$profiles axis-scale-1p3-0p8-1p1-polarity-p-flip"
profiles="$profiles axis-scale-1p3-0p8-1p1-clip-0p3-rad-s"
profiles="$profiles axis-scale-1p3-0p8-1p1-polarity-p-flip-clip-0p3-rad-s"

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) motion=sampled-rate-nonanalytic ;;
    piecewise) motion=sampled-rate-piecewise-linear ;;
    bursts) motion=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root="$prefix-$family-$cadence"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 5 \
      --noise-seed-start 20416145 --motion-seeds 751343 751344 751345 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$motion" \
      --rate-sensor-calibrations $profiles \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 16 \
      --checkpoint "$root/assessment-polarity-clipping-current.checkpoint.json" \
      --checkpoint-every 1000 \
      --output "$root/assessment-polarity-clipping-current.json"
  done
  root40="$prefix-$family-40hz"
  root80="$prefix-$family-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-polarity-clipping-current.json" \
    "$root80/assessment-polarity-clipping-current.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-polarity-clipping-current-v6.json"
done
```

### Paired cadence screen across rate-sensor polarity patterns and clipping severity

This follow-up broadens the calibration screen to 14 profiles: identity; all
seven non-identity p/q/r polarity combinations; clipping alone at `0.15`,
`0.3`, and `0.6 rad/s`; and a p-axis reversal crossed with each of those
three clipping levels. Clipping is component-wise and follows additive sensor
noise. The limits are deliberately synthetic stress settings, not measured
sensor specifications.

The 40/80 Hz study uses Fourier, piecewise-linear, and maneuver-burst sampled
rates; new motion seeds `751346`–`751348`; new moderate-noise seeds
`20416150`–`20416152`; both truth rate frames; and the complete 192-encoding
convention sweep. Trajectories are four seconds long, with stationary OU noise
of `0.003 rad` on orientation and `0.025 rad/s` on rates, correlation times of
`50/100 ms`, and body-frame rate noise. Jitter, dropout, and channel latency
are zero. Each of the six manifests contains 24,192 noisy cases plus three
controls: 14 profiles × 3 motion seeds × 3 noise seeds × 2 truth frames × 96
convention encodings. That is 1,728 cases and 18 paired motion/noise/truth
strata per profile. The 96 convention encodings in each stratum are paired
representations, not independent observations.

Manifests use schema 5, assessments schema 10, and cadence-comparison reports
schema 6. Each family comparison passed its configuration and implementation
gates, compared 24,192 noisy pairs and 3,894,912 common-time rows, and found
exactly equal shared inputs (maximum absolute difference `0.0`). All six
assessments used implementation fingerprint
`d6a445cd3a496b38a18d72b707bc8c18877043fb0958dd543dc700036a5e9dc3` on
Python 3.14.6, NumPy 2.5.2, SciPy 1.18.0, package
`0.0.1a0.post1.dev25+gc2594b4a8.d20261101`.

#### Results

Counts below aggregate all 14 profiles per motion family; each cadence has
24,192 noisy cases. A “wrong recommendation” is an emitted recommendation
that does not match the known synthetic truth. A data-winner miss means the
truth class was not the data-winning class; an evidence miss means truth was
not retained in the evidence set.

| Motion family | Recommendations, 40 → 80 Hz | Wrong recommendations, 40 → 80 Hz | Data-winner misses, 40 → 80 Hz | Evidence misses, 40 → 80 Hz |
| --- | ---: | ---: | ---: | ---: |
| Fourier | 7,632 → 6,096 | 5,040 → 3,792 | 16,584 → 16,608 | 7,272 → 7,248 |
| Piecewise-linear | 6,768 → 2,856 | 5,040 → 2,856 | 17,136 → 17,136 | 7,992 → 7,704 |
| Maneuver bursts | 768 → 0 | 768 → 0 | 17,712 → 17,712 | 9,984 → 7,560 |
| Total | 15,168 → 8,952 | 10,848 → 6,648 | 51,432 → 51,456 | 25,248 → 22,512 |

The 80 Hz minus 40 Hz stratum-level median log-risk shift was negative in
731/756 profile/seed/truth strata (252 per family). This relative-fit movement
is not accuracy: across these generated cases, every recommendation that was
made under a polarity-flipped profile was wrong. Clipping-only responses were
not monotonic with clipping threshold or motion family; for example, the
strongest clipping profile still produced correct Fourier recommendations,
while its p-flip interaction produced wrong recommendations. Treat these as
diagnostics of a synthetic model, not estimates of field error rates, sensor
tolerances, or a direction for threshold tuning. Broader independent seeds,
explicit unknown-polarity/saturation policy tests, and independently labeled
logger data remain necessary before making reliability claims.

A streaming audit of the saved per-case reports found 17,496 wrong
recommendations under polarity-flipped profiles across the two cadences. Every
one had at least one generic warning, but the existing “large residual” warning
appeared in none of the Fourier or piecewise cases and in only 216 of the 768
40 Hz burst cases; the 80 Hz burst cohort made no recommendations. The warnings
usually described exact convention equivalences or multiple near-tied
hypotheses, not the out-of-model per-axis polarity. Thus these generic warnings
did not prevent the false recommendations and cannot be treated as calibration
detection.

This motivated the opt-in
[`assess_rate_polarity_sensitivity`](../README.md#check-sensitivity-to-unknown-rate-channel-polarity)
helper. It reruns the convention fit under each of the eight hypothetical raw
rate-column sign patterns and reports relative data-risk shifts, convention
classes, evidence decisions, and warnings. On a clean, deterministic four-second
analytic trajectory sampled at 40 Hz (`rich_three_axis_rates`, scalar-last
body-to-world quaternion attitude), a synthetic p-channel sign reversal made
the p correction reduce floor-adjusted log risk by about `15.8` relative to
identity. Four different polarity patterns nevertheless fell within the
best-fit margin because they trade off with convention hypotheses. The
regression is exercised by
`tests/test_calibration_diagnostics.py::test_rate_polarity_sensitivity_compares_all_patterns_without_mutating_input`.
The helper is a sensitivity probe, not an automatic correction or
sensor-polarity estimate.

For known sensor limits, `inspect_rate_rails` reports exact or tolerance-bounded
sample counts and consecutive runs at caller-supplied lower/upper values for
each raw input column. Its input model requires an explicit rate unit and
per-column bounds. It does not infer an unknown limit or label a hit as
saturation. The deterministic contract test
`tests/test_calibration_diagnostics.py::test_rate_rail_inspection_counts_known_input_rails_and_runs`
checks counts, fractions, run lengths, and schema round-tripping; the separate
error-path test covers overlapping tolerance regions and missing rate data.
This is an observation utility only: clipping remains irreversible, and the
library does not yet select a safe policy when rails are hit.

#### Reproduction

Generated manifests, assessments, and comparisons are ignored local artifacts
under `.validation-corpus/`; they are not committed. Recreate them with:

```bash
set -euo pipefail
prefix=".validation-corpus/cadence-polarity-clip-levels-751346-348-20416150-152"
profiles="identity polarity-p-flip polarity-q-flip polarity-r-flip"
profiles="$profiles polarity-pq-flip polarity-pr-flip polarity-qr-flip polarity-pqr-flip"
profiles="$profiles clip-0p15-rad-s clip-0p3-rad-s clip-0p6-rad-s"
profiles="$profiles polarity-p-flip-clip-0p15-rad-s"
profiles="$profiles polarity-p-flip-clip-0p3-rad-s polarity-p-flip-clip-0p6-rad-s"

for family in fourier piecewise bursts; do
  case "$family" in
    fourier) motion=sampled-rate-nonanalytic ;;
    piecewise) motion=sampled-rate-piecewise-linear ;;
    bursts) motion=sampled-rate-maneuver-bursts ;;
  esac
  for cadence in 40hz 80hz; do
    case "$cadence" in
      40hz) period=0.025; count=161 ;;
      80hz) period=0.0125; count=321 ;;
    esac
    root="$prefix-$family-$cadence"
    python -m scripts.generate_validation_corpus \
      --outdir "$root" --noise-profile moderate --noise-replicates 3 \
      --noise-seed-start 20416150 --motion-seeds 751346 751347 751348 \
      --noise-sample-counts "$count" --sample-period-s "$period" \
      --truth-convention-sweep --motion-profile "$motion" \
      --rate-sensor-calibrations $profiles \
      --orientation-noise-correlation-time-s 0.05 \
      --rate-noise-correlation-time-s 0.1 --rate-noise-frame body
    python -m scripts.validate_corpus "$root/manifest.json" --workers 16 \
      --checkpoint "$root/assessment-polarity-levels.checkpoint.json" \
      --checkpoint-every 1000 \
      --output "$root/assessment-polarity-levels.json"
  done
  root40="$prefix-$family-40hz"
  root80="$prefix-$family-80hz"
  python -m scripts.compare_cadence_assessments \
    "$root40/assessment-polarity-levels.json" \
    "$root80/assessment-polarity-levels.json" \
    --baseline-manifest "$root40/manifest.json" \
    --candidate-manifest "$root80/manifest.json" \
    --output "$root40/comparison-polarity-levels-v7.json"
done
```

The following hashes identify the local output artifacts relative to the
prefix above, and the source files used to generate and assess them:

```text
implementation_sha256  d6a445cd3a496b38a18d72b707bc8c18877043fb0958dd543dc700036a5e9dc3

cdfc7650dad112df95ba2f7eecc38640c806e60540646338395ee2bb733fabe1  scripts/generate_validation_corpus.py
5d9808747154edb150226e332b35479797d739ae33baae45f0dcf07ce609ace9  scripts/rate_sensor_calibration.py
767ff0c01ee1b17395514c7986313e1f479c54159aa72ed33f6b04df869da0c6  scripts/validate_corpus.py
f10829534dd218506df4ab181b655cf59d2c39eedd4a9d5ea45af59bda87fb79  scripts/compare_cadence_assessments.py
41ae49ab4de6c662abe37c59d6eca624cf7e89a7113975e63127c7be88ffad4f  scripts/validation_noise.py

c994af756eab1b1e4ee788df484327edc3d753556a0b3031969d33db3e06a96d  fourier-40hz/manifest.json
5f7852b5863a0c7a6707a260ed31d40ab60975485a8c4cc0beb95ada1dedebfe  fourier-40hz/assessment-polarity-levels.json
83e69f200a5250b42da977cc0ad491ea9cec0d26790bad30f86d77426b85e993  fourier-80hz/manifest.json
203ad4dfe0bb412d4a45383271b168c420ca7e511d96cb367bb29e62f2636af4  fourier-80hz/assessment-polarity-levels.json
f1b4b32720d2df98f264369fb26cb074d556532131f587d2adce2a27991bb6d4  fourier-40hz/comparison-polarity-levels-v7.json
aa4531453b87735b669d71b5cd376da4e20ac37c1875fa5fa507b73e3da2f23d  piecewise-40hz/manifest.json
34dca8512e19b5e66eb9b88d0f6ffb74652c6c1bdd63117e5add11fb96d7e337  piecewise-40hz/assessment-polarity-levels.json
3a4281ca53f41c3444d66854e732b16bd2ec69c01dd7d77eb163cf8e421edd4a  piecewise-80hz/manifest.json
38910e2fe0db12308f108a6454370f3f8fbc123afbe00f8678edeb8fd11dc2b6  piecewise-80hz/assessment-polarity-levels.json
826404178970376833b03dc61f2531b1fb8b00fd061b261161fe39b4f63299da  piecewise-40hz/comparison-polarity-levels-v7.json
5af4836738e44fb1aafcd4c1102bc1c3d3afc5f33fc84abe2018815931764782  bursts-40hz/manifest.json
7a0c513f8403417bee55fd6377cf38750d1dde4051bf0e90651fd99c59a1b541  bursts-40hz/assessment-polarity-levels.json
ef26a0d28eb9567dd11d87a1f95c769bbe080423ebc201b83be4b05f0a8c245f  bursts-80hz/manifest.json
0a4fe9f3791aeb76708942fd50f97f252ed64eb07adf0de54d31b5665fd9cb8f  bursts-80hz/assessment-polarity-levels.json
86386180ca955023a9b769c0c9026ca6bff515511af33fd3bae1d14ba97fbeb3  bursts-40hz/comparison-polarity-levels-v7.json
```

## Sparse orientation anchors with dense rates

The repeatable stress runner is `scripts/validate_sparse_anchors.py`. It samples
the three independently labeled rate-motion families for 16 seconds, keeps the
rate stream at 100 Hz, and evaluates nominal orientation-anchor cadences of
0.2, 0.4, 0.8, 1.6, and 3.2 seconds (81 down to 6 before dropped anchors).
Five motion seeds are paired across the Fourier, piecewise-linear, and
maneuver-burst families. Each motion is evaluated with clean and moderate
independent per-axis Gaussian measurement noise (orientation 0.003 rad; rates
0.025 rad/s). Attitude noise is left-multiplicative in the world frame; rate
noise is additive in measured body-rate axes.

Six acquisition conditions cover exact regular observations, 20% random anchor
dropout, independent uniform timestamp jitter bounded by ±10 ms for anchors and
±2 ms for rates, dropout plus jitter, relative channel-value latency, and shared
clock-rate scale error. Dropout retains endpoints and at least three anchors.
For timestamp jitter, measurements are sampled at nominal physical times and
their reported timestamps are perturbed. For relative latency, orientation
values are sampled 10 ms earlier and rates 25 ms earlier than their nominal
timestamps. The clock-scale condition scales both timestamp streams by +1000
ppm while leaving sampled values unchanged. Noise and timing realizations are
paired across common timestamps, cadences, and noise tiers. Jitter is zero at
both endpoints so rate coverage stays valid. Default inference settings and
decision thresholds are unchanged.

The 900 paired evaluations (15 motion seeds across families, repeated over
cadence/noise/acquisition conditions) retained the known convention in both the
data-winning classes and evidence contenders in all 900. At the finest nominal
cadence (0.2 s), 174 cases recommended correctly and 6 abstained. Each of the
six acquisition conditions had 29 correct recommendations and one abstention
at this cadence. All 720 cases at 0.4 s or coarser abstained. There were no
wrong recommendations. Thus these irregular schedules and selected timing
perturbations did not dislodge truth in this cohort, but denser rates did not
make coarse pose evidence sufficient. The 900 evaluations are paired repeats
over 15 generated trajectories, not 900 independent trajectories.

This remains synthetic sensitivity evidence—not field accuracy, a calibrated
error rate, or a basis for threshold tuning. The single relative-latency pair
and positive clock-scale point do not establish response across a timing-error
range, and the study does not replace independently labeled real logs. Run and
save the complete
typed report (class IDs, seeds, noise scales, and timing-condition definitions)
with:

```bash
python -m scripts.validate_sparse_anchors \
  --output .validation-corpus/sparse-anchor-study-v3.json
```

The output directory is ignored and the report is reproducible from its stored
config, seeds, noise scales, timing conditions, and package version. Follow-up
work should sweep relative latency and positive/negative clock-scale errors,
then validate on independently labeled multi-system trajectories before making
broader claims.
