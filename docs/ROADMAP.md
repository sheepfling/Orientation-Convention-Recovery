# Productization roadmap

The `0.0.1a0` tag is an alpha checkpoint. A green synthetic test suite establishes
implementation consistency, not recovery accuracy on outside data.

## Current engineering baseline

- Before the coverage gate was added, check-only `scripts/ci.py` passed on
  Windows/Python 3.11.15, 3.12.13, 3.13.14, and 3.14.6 with 403 tests per version.
  The current pipeline measures statement and branch coverage and enforces an
  85% combined floor. A previous 423-test suite passed on Windows/Python 3.11.9,
  3.12.13, 3.13.14, and 3.14.6. On the current worktree, full `scripts/ci.py`
  runs pass on Windows/Python 3.11.15 (`--fix`) and 3.12.13, 3.13.14, and
  3.14.6 (check-only): each passes mdrepo, rumdl, generated-artifact checks,
  Ruff, strict Pyright, scope markers, all 462 tests, and alpha public-contract
  compatibility, with 89.45% combined coverage. `scripts/check_distribution.py`
  also passes on all four versions, including sdist/wheel builds, Twine checks,
  and an installed-wheel smoke test that runs sparse inference through both
  public entry points. `scripts/ci.py` checks for `pytest-cov` before running
  checks or applying fixes and prints the `.[dev]` install command when it is
  missing.
- `OverallDiagnostics.confidence` is documented in the generated Pydantic schema
  as a bounded, uncalibrated heuristic score; runtime validation enforces its
  deterministic score band, and it remains independent of the recommendation
  decision. The fresh noisy screen demonstrates this distinction but cannot
  calibrate the score.
- The current working-tree GitHub Actions cleanliness and distribution matrix is
  configured for Python 3.11–3.14 on Ubuntu and Windows. Current local Windows
  evidence for this worktree covers all four versions; hosted verification
  remains open. Both available hosted runs are historical:
  they ran one cleanliness job on commit `60a080a` and failed because the
  generated `examples/synthetic_quaternion.csv` was stale ([run 1](https://github.com/sheepfling/Orientation-Convention-Recovery/actions/runs/31921873300),
  [run 2](https://github.com/sheepfling/Orientation-Convention-Recovery/actions/runs/31921826162)).
  No hosted run is associated with current local `HEAD` `c2594b4`, so the
  expanded 8-cell matrix remains unverified in GitHub Actions.
- The current source passes the isolated sdist-to-wheel build, Twine check,
  target install, and installed-wheel smoke on Windows/Python 3.11.9, 3.12.13,
  3.13.14, and 3.14.6 using `scripts/check_distribution.py`. This verifies the
  build backend can be provisioned in isolation even when it is absent from the
  project virtualenv. Both artifacts are checked for direct-URL dependencies;
  the latest Python 3.14.6 wheel smoke also runs sparse inference through both
  public entry points from the installed artifact. Hosted verification of the
  current matrix remains open.
- `TrajectoryData.time_s` now documents seconds and permits an absolute clock
  origin. A regression case confirms the same convention and decision on
  Unix-epoch seconds, with interval RMSE within `1e-6 rad`; it passes in the
  current full check-only runs on Python 3.11.9, 3.12.13, 3.13.14, and 3.14.6.
  For high-resolution integer epoch
  clocks, subtract the origin before float conversion to avoid losing interval
  precision.
- The configurable sample period and integration-cache behavior in the
  full-inference benchmark are covered by the current check-only runs on
  Windows/Python 3.11.9, 3.12.13, 3.13.14, and 3.14.6, including propagation to
  fresh workers and cached/uncached report fingerprint parity.
- The validation corpus has labeled recovery/ambiguity/miss evaluation across
  analytic and numerically integrated seeded-rate trajectories, with paired noise,
  duration, timestamp-jitter, constant timestamp-clock-scale, whole-row-dropout,
  and analytic channel-latency sweeps.
  Standard CI also recovers all 192 independently encoded quaternion/rate convention
  combinations plus three fixed controls, and all 192 native-Euler convention
  combinations, with zero misses on clean independently generated regression corpora.
  These controlled studies are not evidence of accuracy on independent systems;
  no independently labeled external logs are available yet.
- `SparseOrientationRateData` supports timestamped quaternion or Euler pose anchors
  at a lower cadence than the rate stream. CI verifies both representations against
  independently generated motion, including 81 anchors with 1,601 rate samples,
  irregular anchor boundaries, and sequential body/world rate-increment composition.
  Sparse evidence deliberately uses non-overlapping anchor intervals and can abstain
  when the number of independent pose changes is insufficient. This validates the
  implementation on synthetic data, not recovery accuracy on real IMU logs.
- Full-convention corpus generation now reuses each prepared motion and acquisition
  realization across its 192 encoded conventions. A call-count regression protects
  the reuse, and optimized reruns matched every CSV and manifest byte-for-byte in
  the compared 20 and 80 Hz cells. Local measured generation time fell from about
  39.5/69.5 seconds to 0.86/1.48 seconds, respectively; this tooling improvement
  does not alter inference or close the external-data evidence gate. See the
  [validation record](VALIDATION.md#convention-sweep-generator-reuse).
- The independent generator now models a shared constant timestamp-clock scale error.
  In a paired, noiseless 4/10-second pilot (one motion, one truth convention, seven
  ppm values), the true class remained the sole data/evidence winner in all 14 cases.
  Recommendations remained correct through ±1,000 ppm; at ±10,000 and ±50,000 ppm
  the existing residual-dependence guard abstained, with no false recommendations.
  A deterministic CI regression now checks all 14 winner and decision outcomes.
  This does not identify clock error as the cause or establish a ppm threshold; see
  the [clock-scale pilot](VALIDATION.md#shared-timestamp-clock-scale-pilot).
- A paired, noisy clock-scale screen expanded this to all seven generated motion
  profiles, three noise tiers, three sampled-rate motion seeds, two durations, and
  seven clock scales (1,638 noisy cases). It produced 522 correct recommendations
  and 1,116 truth-retaining abstentions, with no evidence misses or false
  recommendations. Data winners and operational classes were invariant in paired
  comparisons, while some evidence sets changed and 17 baseline recommendations
  became abstentions at −50,000, −10,000, and +50,000 ppm. These generator-specific
  results do not establish field accuracy or a clock-error threshold; no policy
  thresholds changed. See the [paired noisy clock-scale screen](VALIDATION.md#paired-noisy-timestamp-clock-scale-screen-across-motion-profiles).
- A fresh-seed, 1,560-case noisy screen crossed all seven motion profiles with
  three noise levels, paired jitter/dropout conditions, and OU-correlated noise.
  It produced 526 correct recommendations, 224 raw data-winner misses, and no
  evidence misses or incorrect recommendations; truth remained available for
  abstention in every case. The results are paired, fixed to one truth convention,
  and generator-specific, so they do not estimate field accuracy or calibrate
  confidence. No decision thresholds changed. See the [fresh noisy screen](VALIDATION.md#fresh-seed-noisy-decision-screen-across-all-motion-profiles).
- An out-of-model 90-degree sensor-to-body rate rotation exposed a unique false
  recommendation under near-persistent residuals. Evidence sufficiency now abstains
  when absolute residual lag-one correlation is at least `0.95` and the estimated
  residual scale exceeds the configured floor. The fixed-seed regression passes, but
  this is a narrow guard, not calibration or general sensor-mount recovery.
- A bounded TUM-VI room-sequence adapter now writes local canonical CSV/manifest
  pairs and records typed citation, license, convention basis, timing caveat, source
  file digests, exact requested/selected time ranges, and transformations in the
  assessment. It checks that source hashes remain stable during preparation, and
  content-addresses case IDs. Unmeasured noise, latency, jitter, dropout, and
  timestamp-clock scale metadata are explicit nulls, not synthetic zero defaults.
  No source data is bundled. It requires explicit EuRoC-style nanosecond timestamps, named body-rate
  and quaternion columns, and declared units; ambiguous aliases or incompatible
  units are rejected instead of being assigned a convention by inference.
  A local ingestion smoke test has now exercised four gap-free windows from a
  pinned third-party room1 CSV re-export: three correct recommendations and one
  truth-retaining abstention, with no winner or operational misses. The 0–20 s
  window is nested in the 0–40 s window; all excerpts are one sequence and do not
  count as independent accuracy trials. The mirror's extraction/conversion was not
  independently audited, and the official EuRoC/DSO export has not been run locally.
  The repeatable sparse-stream runner now preserves native pose and gyro timestamps,
  fingerprints both direct inputs, verifies optional lineage manifests, and keeps
  upstream provenance separate from the transformations applied by sparse validation.
  On the same local sequence it tested 13–201 native pose anchors against 7,658–7,977
  dense rate samples; the mapped convention ranked first and was uniquely
  recommended at all five cadences. This is still a same-sequence compatibility smoke
  test, not independent accuracy evidence.
  TUM-VI's calibrated axis and gyro/pose time alignment make it unsuitable as a
  blind raw-axis or time-offset test; it does not satisfy the two-system gate. See
  the [bounded smoke-test notes](VALIDATION.md#optional-tum-vi-real-data-preparation).
- The sampled-rate synthetic attitude integrators are cross-checked against a
  separate DOP853 ODE solution using the same continuous rate signal. The
  standalone sample-step integrator converges at second order and stays below
  `2e-4 rad` at 25 ms sampling across three seeds. Corpus generation uses a
  query-independent 2.5 ms grid anchored at `t = 0`; its jittered and delayed
  queries stay below `1e-5 rad` error across three seeds. This verifies generator
  accuracy only.
- The four-seed convention sweep across low, moderate, and high noise covers
  9,216 paired synthetic cases. Truth remained in the evidence contenders and
  the policy abstained on every noisy case; these runs measure neither production
  miss rates nor confidence calibration. See [validation results](VALIDATION.md).
- A paired 35% row-dropout stress covered the same 3,072 low-noise conventions
  across four motions and seeds. It produced 576 correct synthetic recommendations
  in rich motion while confidence labels remained indeterminate; aggregate raw
  winner misses increased. This remains generator-specific, not calibration evidence.
- A four-seed sampled-rate pilot covered 16 low-noise cases across paired timestamp
  jitter and row dropout; all 16 recommendations were correct. It is one synthetic
  motion family and one seed block, not an accuracy or confidence-calibration rate.
- A sampled-rate convention sweep covered the full 192-value convention matrix on
  three seeded rate processes (576 paired encodings). All data winners and
  operational classes matched truth; 384 cases received correct recommendations,
  while 192 abstained with truth retained among ambiguous contenders. Confidence
  labels were indeterminate throughout. This is synthetic evidence from one motion
  model, not external-source validation or calibrated accuracy.
- A native Euler sweep covered 12 orders, both Euler modes, four angle units, and
  both attitude directions across three seeded Fourier paths (576 cases). Direct
  axis matrices and analytic derivatives avoid inference/SciPy generation code.
  The data winner matched truth in 573/576 cases; 366 recommendations were correct,
  210 cases abstained with truth retained, and no recommendation was false. The
  rate convention was pinned, and all confidence labels remained indeterminate.
  See [validation results](VALIDATION.md).
- A matched four-second native-Euler cadence sensitivity sweep covered 10, 40, and
  100 Hz across the same three seeds (1,728 cases, same implementation/runtime).
  Correct recommendations fell from 507/576 at 10 Hz to 366/576 at 40 Hz and
  132/576 at 100 Hz; all recommendations were correct and truth remained in the
  evidence set. At 100 Hz, 54 data-winner/operational misses resulted in abstention,
  not false recommendations. The experiment held per-sample noise fixed, so this
  does not isolate cadence from time-domain noise density. See [validation
  results](VALIDATION.md).
- A paired cadence follow-up samples one shared OU measurement-noise path at 10,
  40, and 100 Hz, preserving a fixed real-time covariance across 1,728 cases.
  Correct recommendations were 504/576, 382/576, and 238/576 respectively; all
  recommendations remained correct, while the 100 Hz cohort had 16 data-winner
  misses that abstained with truth retained. This is one synthetic motion/noise
  model and the same seeds, not held-out evidence or a threshold-tuning basis.
  See [validation results](VALIDATION.md).
- A fresh motion/noise seed block (730304--730306, noise seed 20301032) adds another
  1,728 OU cases at the same cadences. Across both blocks, coverage was 1,050/1,152
  at 10 Hz, 744/1,152 at 40 Hz, and 466/1,152 at 100 Hz. All recommendations
  remained correct; 21 high-rate winner misses abstained with truth retained. The
  six motion seeds still use one Euler-Fourier family, so this does not establish
  robustness to other dynamics or external logs. See [validation results](VALIDATION.md).
- A follow-up compared OU time-constant pairs (12.5/25 ms and 200/400 ms) with the
  50/100 ms reference at all three cadences on matched motion seed 730301. At 100 Hz,
  recommendations shifted from 34/192 to 0/192 and 96/192, respectively; all were
  either correct recommendations or truth-retaining abstentions. Expanding the
  short pair to three motion seeds gave 168/576 recommendations at 100 Hz, with all
  47 raw winner misses concentrated in one seed. The long pair remains single-seed,
  so this is a sensitivity screen, not a balanced estimate or a threshold-tuning
  basis. This Euler-specific screen did not include non-Euler motion; subsequent
  sampled-rate studies are recorded below. See [validation results](VALIDATION.md).
- The generic sampled-rate generator now supports continuous-time OU noise and
  configurable cadence using the shared 1 ms noise process. A one-seed, low-noise
  three-time-constant sweep recommended all 192 convention encodings at each
  cadence. Moderate noise did the same; high noise yielded 192/192 recommendations
  at 10/40 Hz and conservative abstention at 100 Hz. That abstention repeated across
  three motion seeds with no winner/evidence misses or false recommendations. Noise
  seeds and motions remain too limited for a performance rate; see
  [validation results](VALIDATION.md).
- A paired sampled-rate sweep adds three independently generated motion seeds and
  20 shared noise seeds at each of three noise levels (180 cases). All data winners
  matched truth; all 100 recommendations were correct, with motion-dependent
  coverage at moderate noise and abstention across all high-noise cases. This is
  synthetic robustness evidence only; held-out real loggers remain the accuracy gate.
- A balanced sampled-rate OU follow-up crosses three motion seeds and 20 paired
  noise seeds with three correlation-time pairs, three noise levels, and 10/40/100 Hz
  sampling (1,620 noisy cases). It produced 930 correct recommendations and 690
  conservative abstentions, with no data-winner/evidence misses or incorrect
  recommendations. Coverage differed substantially by motion seed, so this is a
  sensitivity result—not a pooled performance rate or basis for threshold changes.
  The study still uses one Fourier-rate motion family and one convention; see
  [validation results](VALIDATION.md).
- A separate piecewise-linear knot-rate family now has an independent DOP853
  integration cross-check, a three-seed sweep over all 192 quaternion/rate
  conventions (576 noisy cases), and the same balanced OU/noise/cadence design
  (1,620 noisy cases). The convention sweep had 384 correct recommendations and
  192 abstentions; the OU matrix had 896 correct recommendations and 724 abstentions.
  Both had zero data-winner/evidence misses and zero incorrect recommendations, but
  coverage varied strongly by motion seed. This broadens synthetic generator
  diversity; it is not external validation or calibration evidence.
- A third, smooth per-axis maneuver-burst family now passes the same fixed-grid
  DOP853 and query-independence checks. Across three seeds and all 192 conventions,
  it abstained on all 576 low-noise cases for both four- and eight-second records,
  with no winner/evidence/operational misses. The same exact-data sweep recommended
  all 576 correctly. This sharp noise-to-confidence gap is a diagnostic, not a
  threshold-tuning target or external accuracy estimate. A paired 180-case
  single-convention noise ladder also had zero wrong winners/recommendations and
  zero recommendations: low/moderate cases passed the stable-evidence check but
  retained multiple contenders, while 48/60 high-noise cases had insufficient
  evidence. Residual lag-one autocorrelation was about -0.46 across levels, so the
  i.i.d. evidence scale remains approximate. No threshold changes were made.
- A matched 180-case-per-family noise ladder now compares Fourier, piecewise-linear,
  and maneuver-burst rates at the same truth convention, cadence, motion-seed IDs,
  and noise-seed IDs. It produced 108/180, 85/180, and 0/180 correct
  recommendations respectively, with no false recommendations or winner misses.
  The first-order colored reference produced mean LLRs 33.5--41.0x the package's
  i.i.d. scores; even-seed-fitted covariance blocks did not consistently improve
  odd-seed prediction over the analytic reference. These diagnostics are synthetic
  and do not justify a resolver or threshold change.
- A matched full sweep now covers 192 truth conventions, three motion seeds, and
  three noise levels (1,728 cases per family). Fourier recommended 1,104/1,728,
  piecewise-linear 768/1,728, and bursts 0/1,728; all 5,184 noisy cases retained
  the correct data/evidence/operational winner and none had a false recommendation.
  Coverage varied sharply by motion seed. An exploratory Fourier body/world rate-
  frame recommendation gap appeared only at moderate noise (144/288 vs 192/288);
  with one noise realization per cell, it needs replication before interpretation.
  No inference thresholds or confidence labels were changed.
- Five additional moderate-noise realizations per Fourier motion seed now replicate
  that frame comparison; combined with the original seed, the six-seed paired sample
  has 1,008/1,728 body and 1,152/1,728 world recommendations, with no wrong
  recommendations or winner misses. All 144 discordant pairs favor world-frame only
  on motion seed 732501; the other two seeds have no frame gap (one all-recommend,
  one all-abstain). This is more consistent with seed-specific ambiguity than a
  general frame bias, but remains synthetic and needs broader motion/cadence checks.
- A previously unrecorded burst OU/cadence screen is now documented: 1,620 cases
  across three cadences, three OU time-constant pairs, three motion seeds, 20 noise
  seeds, and three noise levels. It produced no recommendations, 64 data-winner/
  operational misses, and 649 evidence-insufficient abstentions; truth remained in
  the evidence set in every case. This was one truth convention; the later full
  convention and timing follow-ups are recorded below.
- The reference OU condition has now been swept across all 192 burst truth
  conventions, three motion seeds, three noise levels, and 10/40/100 Hz (5,184
  noisy cases). It produced 192 correct recommendations at 10 Hz/low noise and no
  recommendations elsewhere; 384 cases had raw-winner/operational misses, with no
  evidence misses or false recommendations. At moderate noise, 96 misses at 10 Hz
  disappeared at denser cadences; at high noise, 96 misses persisted at all three.
  Outcomes were strongly seed-dependent and based on one noise realization per
  cell, so replication across motion seeds remains open. No thresholds changed;
  see [validation results](VALIDATION.md).
- A burst timing-artifact screen adds 3 ms jitter, 10%/35% whole-row dropout, and
  13/27 ms channel delays at 40 Hz across 192 conventions, three motion seeds, and
  three noise levels (15,552 noisy cases). Jitter and latency did not change public
  decisions or raw-winner correctness, though latency often expanded evidence
  contenders. At 35% dropout, low-noise recommendations appeared only on motion
  seed 732403. A three-noise-seed follow-up repeated 480/1,728 correct
  recommendations on that same seed, with no false recommendations or winner/
  evidence misses; the other motion seeds still had no recommendations. This is a
  repeatable, seed-specific synthetic sensitivity, not a claim that dropout helps.
- A fresh-seed extension used three new burst motions, three noise replicates, and
  20,736 paired noisy cases across clean, 3 ms jitter, 35% dropout, and 13/27 ms
  channel-delay conditions. Jitter and latency again left public decisions and
  winner correctness unchanged, while evidence contenders changed in 216/5,184
  and 816/5,184 cases respectively. Dropout produced 576 correct low-noise
  recommendations on a different motion seed (732405), with no false
  recommendations, winner misses, or evidence misses; high-noise insufficiency
  fell on two of the three motions. This replicates a seed-dependent sensitivity,
  not a general benefit of dropout. The screen remains synthetic and within one
  motion family; broader motion families, additional timing levels, and independent
  logger data remain open. No inference thresholds changed; see
  [fresh-seed validation results](VALIDATION.md#fresh-burst-seed-replication).
- A low-noise cross-family dropout replication covered fresh Fourier-rate and
  piecewise-linear-rate motions, with three noise replicates per convention. On
  Fourier, 35% dropout resolved 192 additional cases (all on motion seed 742402),
  raising correct recommendations from 1,536/1,728 to 1,728/1,728. Piecewise-rate
  already recommended correctly in all 1,728 cases and was unchanged. Both had no
  false recommendations or winner/evidence misses. This supports a synthetic,
  ambiguity-dependent effect rather than a general dropout benefit. A paired
  timing follow-up found 3 ms jitter invariant in both families; 13/27 ms channel
  latency added 192 Fourier and 96 piecewise abstentions on motion seed 742402,
  without winner/evidence misses. See [cross-family results](VALIDATION.md#cross-family-dropout-replication)
  and [timing results](VALIDATION.md#cross-family-jitter-and-latency-replication).
- A fresh moderate/high-noise latency holdout used three new motion seeds across
  Fourier-rate and piecewise-linear-rate trajectories (6,912 paired noisy cases).
  Moderate Fourier decisions were unchanged; moderate piecewise ambiguity expanded
  on one already-abstaining seed. At high noise, latency added 48 correct Fourier
  recommendations on one seed while expanding piecewise ambiguity on another, with
  no winner/evidence misses or false recommendations. This non-monotonic,
  seed-dependent response is synthetic evidence that timing convention belongs in
  the acquisition contract; it is not a basis for threshold tuning. An isolated
  channel follow-up found the 27 ms rate delay removed 192 moderate- and 192
  high-noise piecewise recommendations, while high-noise orientation delay added
  48 Fourier recommendations and removed 72 piecewise recommendations. All 13,824
  cases retained the correct winner and evidence; these synthetic results do not
  justify tuning thresholds. See [combined latency results](VALIDATION.md#moderatehigh-noise-latency-holdout)
  and [isolated-channel results](VALIDATION.md#isolated-orientation-and-rate-latency-holdout).
- Long corpus assessments support atomic checkpoint/resume, bound to the exact
  manifest, source CSV contents, inference/validator implementation, and runtime.
  The streaming writer was measured on a 23,043-case checkpoint: the 309.6 MiB
  input serialized to 261.2 MiB in 2.4 s. After loading, process RSS was 675.5 MiB;
  sampled peak RSS during serialization was 676.2 MiB (+0.7 MiB). This measures
  serializer overhead after the case objects are resident, not whole-run or
  32-worker memory limits. The input checkpoint SHA-256 is
  `c1770697a8524b7ed818619c51ed6b152d108390133552e86d61b0db55eae881`, and the
  measured writer source SHA-256 is
  `767ff0c01ee1b17395514c7986313e1f479c54159aa72ed33f6b04df869da0c6`.
- Exact-class grouping and evidence prefixes avoid known quadratic paths.
- Exact symmetry grouping, ranking, operational selection, evidence summaries,
  dimension assessments, diagnostics, and report assembly now have separate
  modules; `inference.py` orchestrates them.
- Shared model primitives, search contracts, policy configuration, convention
  hypotheses, and report contracts now have separate modules with stable
  `models.py` re-exports.
- A repeatable alpha-tag audit checks public exports, function signatures,
  class constructor shapes, enum members, and generated model schemas without
  tracking schema files.
- [Performance baselines](PERFORMANCE.md) include 10,001-sample compact and
  48-hypothesis quaternion searches, plus 576-hypothesis Euler searches through
  10,001 samples with isolated-process peak memory and cold-call timing. A fresh
  Euler matrix on Python 3.11/3.12/3.14 measured 101, 501, and 2,001 samples; its
  analytic-motion timings varied by less than 3% on this host. A one-repeat
  sampled-rate Euler matrix also covers those sizes and runtimes; it showed
  similar timings for one synthetic seed, not a general scaling guarantee.
- Euler range/branch diagnostics are now cached per order/unit during inference.
  A paired 3.11 synthetic benchmark preserved identical reports and measured
  1.51×–1.76× speedups at 101–2,001 samples; host timing instability still makes
  this a diagnostic, not a performance guarantee.
- Observed interval/window rotations are reused per decoded attitude, rate frame,
  and window size, including missing-timestamp period refinement. A paired 3.14
  timestamped synthetic Euler benchmark preserved identical reports and measured
  1.19×–1.24× speedups at 101–501 samples; traced peak memory rose by about
  0.02–0.05 MB. A small missing-time synthetic probe preserved reports and was
  1.30× faster at 31 samples/two hypotheses; this is only a regression probe.
  Matched fresh-process comparisons on the same 10,001-sample/576-hypothesis
  sampled-rate input at 25 ms preserved identical reports on Python 3.11, 3.12,
  and 3.14. Median peak working set increased 1.6–2.7%; timing showed about
  1.25× speedup on 3.12/3.14 but was inconclusive on 3.11 due to large run-to-run
  variation. Paired 50 ms and 100 ms follow-ups on Python 3.14 also preserved
  identical reports within each cadence; ratios were 1.22× and 1.26×, with
  1.8–2.7% higher median peak working set. These two-repeat, single-host
  synthetic measurements are not a scale guarantee.
- Predicted rate integrations are reused across attitude conventions only when
  the decoded rate array and resolved interval durations match exactly. A full
  report-parity test checks the cached and bypassed paths. The current 3.11
  sampled-rate Euler run measured 1.64 s at 501 samples and 6.17 s at 2,001,
  with 6.32/14.45 MB traced peaks; these are one-run local baselines, not a
  performance guarantee. A separate 10,001-sample cold-process run took 29.78 s
  with a 160.13 MB peak working set. See [performance details](PERFORMANCE.md).
  See [performance details](PERFORMANCE.md).

## Next evidence gates

1. Assemble a permission-cleared corpus from at least two external systems.
   Record the exact logger/API version, independent convention label, sensor
   timing contract, and source citation for each case. Hold out entire sources
   or recording sessions for evaluation.
2. Measure class-level miss rate, abstention rate, and false diagnostic evidence
   sufficiency across motion, noise, timing, and sampling conditions. Tune
   thresholds only on training sources and publish held-out results with the
   configuration used.
3. Calibrate any probability or confidence claim against held-out outcomes and
   temporal dependence. Until then, keep the current heuristic and weight-only
   labels explicit.

## Next engineering gates

1. Extend the convention comparison under noise to more motion seeds and cadences.
   A fresh clean full-convention screen now crosses three sampled-rate families,
   three new motion seeds, and 20/40/80 Hz: all 5,184 convention cases retained the
   correct data and operational winner, with no evidence misses or false
   recommendations. Fourier 20 Hz conservatively abstained on 384/576 convention
   cases due to insufficient evidence; all other convention cases were recommended.
   The 17,496 shared-time input-field comparisons were exact. This does not close
   the noisy robustness gate; see the [fresh clean convention/cadence screen](VALIDATION.md#fresh-clean-full-convention-screen-across-sampled-rate-families-and-cadences).
   The moderate-noise body/world gap attenuated with five new noise seeds and was
   confined to one of three motion seeds; large seed-to-seed coverage differences
   remain. The burst family now also has
   full-convention cadence, timing-artifact, and fresh-motion dropout screens.
   The 732407--732409 dropout replication did not reproduce the earlier low-noise
   recommendation gain; all noisy cases abstained, and moderate-noise top-winner
   misses increased on one motion seed despite truth remaining in the evidence
   set. Treat dropout effects as seed-sensitive and make no threshold changes.
   See [fresh burst-dropout replication](VALIDATION.md#fresh-burst-dropout-replication).
   A fresh piecewise rate-latency replication on seeds 742510--742512 showed no
   moderate-noise effect and only 72 high-noise contender expansions on one seed,
   with no decision changes. The earlier recommendation losses did not repeat,
   underscoring seed dependence. See [fresh-seed rate-latency results](VALIDATION.md#fresh-seed-piecewise-rate-latency-replication).
   A 50 ms follow-up on motion seeds 742520--742522 likewise showed no paired
   rate-delay effect, but high-noise coverage ranged from 0/576 to 576/576
   recommendations by motion seed. Its noise seeds differ from the 25 ms run,
   so the cross-cadence coverage difference is confounded and cannot be credited
   to cadence alone. See [50 ms cadence results](VALIDATION.md#50-ms-cadence-rate-latency-replication).
   A matched-noise 25/50 ms baseline comparison reused the same OU paths and
   verified every shared input row exactly. Moderate outcomes were invariant;
   high-noise coverage increased by 24 correct recommendations at 50 ms, all on
   one motion seed, with no winner/evidence misses or false recommendations.
   In the paired 27 ms rate-delay conditions, 25 ms produced 576 moderate-noise
   and 72 high-noise truth-retaining abstentions on one seed each; 50 ms had no
   decision changes. Every recommendation remained correct, indicating a
   synthetic cadence/latency interaction rather than a stable delay response.
   See [shared-path baseline results](VALIDATION.md#shared-path-cross-cadence-baseline-comparison)
   and the [rate-delay interaction](VALIDATION.md#shared-path-rate-delay-comparison-across-cadences).
   A fresh piecewise 40/10 Hz screen on seeds 742540--742542 matched 141,696
   shared-time rows exactly. Moderate outcomes were unchanged; high-noise
   recommendations moved from 0/1,728 at 40 Hz to 1,728/1,728 correct at 10 Hz,
   equally across all three seeds and with no misses. This strong synthetic
   cadence/noise interaction is consistent with the need to model temporal
   dependence, but does not isolate that cause; do not tune thresholds from it.
   See the [fresh 40/10 Hz piecewise screen](VALIDATION.md#fresh-seed-piecewise-40-hz10-hz-cadence-screen).
   A same-path Fourier-rate screen on seeds 742530--742532 matched all 139,968
   shared-time rows per cadence comparison exactly. Recommendation coverage rose
   from 1,152 to 1,728 at moderate noise and 24 to 1,080 at high noise from 25 to
   50 ms; at 25 ms, high-noise 27 ms delay moved the 24 recommendations to
   abstention, while the 50 ms delay had no paired effect. Winners and operational
   classes stayed fixed, but the large seed/cadence sensitivity is a warning about
   evidence behavior, not a calibration result. Repeat with more Fourier seeds and
   evaluate temporal-dependence handling before changing thresholds. See the
   [Fourier-rate screen](VALIDATION.md#shared-path-fourier-rate-cadence-and-rate-delay-screen).
   The baseline first-order covariance diagnostic showed similar normalized
   residuals across cadence; fitting covariance on one even noise seed did not
   beat the known scalar OU reference on any of 192 convention groups held out on
   odd seeds. This is far too little independent noise data to justify integration
   into package decisions. See the same [Fourier-rate validation](VALIDATION.md#shared-path-fourier-rate-cadence-and-rate-delay-screen).
   A second fresh-seed replication expanded this Fourier screen to six motions:
   pooled 25→50 ms recommendations rose 2,304→3,168 at moderate noise and
   24→1,656 at high noise. Rate delay remained inert for moderate noise but
   shifted 24 high-noise 25 ms and 96 high-noise 50 ms decisions to abstention,
   in different seed batches. The covariance holdout also reversed between
   batches (fitted covariance better in 0/192 versus 192/192 truth groups),
   confirming that these small synthetic splits cannot guide calibration or
   package integration. See the [fresh-seed replication](VALIDATION.md#fresh-seed-fourier-rate-replication).
   A subsequent 20-noise-seed holdout across three fresh Fourier motions improved
   the covariance check to 30 training/30 held-out trajectories per noise tier.
   The known scalar OU reference beat an empirical 3x3 covariance fit on held-out
   RMS block error (0.059 vs 0.077); colored and i.i.d. references both favored
   the labeled convention in all 360 paired one-field comparisons per tier.
   Package recommendation coverage remained motion-seed-dependent (40/60 moderate,
   15/60 high), with no misses. This validates only the ideal synthetic covariance
   calculation; it does not calibrate decisions or explain cadence sensitivity.
   See the [fresh OU covariance holdout](VALIDATION.md#fresh-seed-ou-covariance-holdout-and-colored-evidence-screen).
   Do not tune the near-tie margin, evidence thresholds, or confidence labels from
   these synthetic results; keep them separate from held-out multi-logger field
   validation. A fresh moderate-OU full-convention replication across three new
   motion seeds, three sampled-rate families, and 20/40/100 Hz added 5,184 noisy
   cases plus 27 controls. It had no winner/evidence misses or false recommendations,
   but only 2,016/5,184 convention cases were recommended and coverage varied sharply
   by seed. A 100 Hz piecewise case exposed a body/world coverage split only under
   combined orientation and rate noise. The generator now supports explicitly
   body- or world-referenced rate-noise vectors and exact OU bridges on a 0.5 ms
   output grid; the default preserves historical per-truth-frame injection. A
   first canonical-body 80 Hz piecewise holdout had zero recommendations for 576
   cases across three motion seeds. A second noise block across all three motion
   families likewise had 0/1,728 convention recommendations and no misses; body
   and world outcomes were balanced. In a paired Fourier 40-to-80 Hz comparison,
   all 92,736 common-timestamp input rows were identical, yet all 576 evidence sets
   expanded and 384 recommendations became abstentions. A second block with new
   motion seeds and an independent OU noise seed also shifted every nearest
   alternative log-risk delta downward (median −0.819), but only 144/576 cases
   crossed the fixed margin of 2.0. In both blocks, sequential evidence remained
   sufficient, the nearest alternative class did not change, and common-time input
   rows matched exactly. Each 576-case cohort contains six motion-seed/truth-frame
   strata with 96 convention encodings each—not 576 independent replicates. This
   is replicated cadence-sensitive relative fit, not evidence growth; fixed
   sample-count windows also cover different physical durations at each rate. A
   fresh cross-family block repeated the downward margin shift for piecewise and
   maneuver-burst rates; piecewise lost 288/576 recommendations, while bursts had
   none at either cadence because the cohort was already ambiguous/insufficient.
   Thus decision impact depends on motion and starting margin. A fresh five-noise-seed
   block now crosses three new motion seeds in Fourier, piecewise, and burst families
   at 40/80 Hz: the paired risk-margin median fell in all 30 motion/noise/truth-frame
   strata per family. Recommendation counts fell from 1,728 to 576 for Fourier and
   1,920 to 1,056 for piecewise; bursts remained at zero. There were no winner or
   evidence misses. These are 15 motion/noise combinations per family, not 2,880
   independent trials; truth-frame and convention encodings are paired variants,
   and noise seeds are shared across families for comparison. The paired low/high
   extensions now cover all three noise tiers: every risk-margin stratum shifted
   downward, but recommendation loss varied by motion family and tier. A new
   40/80 Hz factorial crossed moderate noise with clean, jitter-only, dropout-only,
   and combined jitter/dropout conditions using nested shared acquisition paths.
   Recommendation counts and classes were unchanged across cadence, with no
   winner/evidence misses, although all margin strata fell and burst evidence sets
   expanded. Do not tune thresholds from these synthetic blocks. A fresh full-
   convention paired study then crossed four 40/80 Hz channel-latency combinations
   across the three sampled-rate families. A same-build follow-up crossed identity,
   15-degree misalignment, p-axis bias, and per-axis scale defects across those
   families. At 80 Hz, recommendations fell from 6,336 to 3,480 for Fourier and
   3,840 to 2,496 for piecewise motion; maneuver bursts remained recommendation-free.
   There were no incorrect evidence-sufficient outcomes or recommendation misses;
   burst data-winner/operational misses fell from 672 to 576. A subsequent balanced
   scale × polarity × clipping screen found many truth-winner/evidence misses under
   polarity reversal; the deliberately strong synthetic `0.3 rad/s` clipping level
   eliminated recommendations in all three sampled-rate families. These are
   controlled synthetic encodings, not field trials or evidence for retuning
   thresholds. The follow-up now covers all eight p/q/r polarity patterns, three
   clipping-only levels, and p-axis reversal crossed with each clipping level, on
   three fresh motion and three fresh noise seeds. Across the three families,
   731/756 paired calibration/seed/truth strata had a lower 80 Hz relative-risk
   margin, but this did not improve recovery: every recommendation emitted under
   a polarity-flipped profile was wrong in this selected cohort, and data-winner
   misses remained essentially unchanged. Clipping responses were not monotonic
   across levels or motion families. An opt-in typed API now compares fit
   sensitivity under all eight raw rate-column sign patterns without mutating
   inputs or auto-correcting data. This surfaces polarity/convention tradeoffs,
   but does not identify physical polarity. A second opt-in API counts samples
   at caller-declared per-column rate rails; it does not guess unknown limits or
   infer that rail hits came from saturation. Unknown-limit behavior and
   independently labeled acquisition logs remain open. No thresholds were
   tuned. See the [polarity sensitivity probe](../README.md#check-sensitivity-to-unknown-rate-channel-polarity),
   [known-rail inspection](../README.md#inspect-samples-at-known-rate-rails),
   and its [synthetic assessment](VALIDATION.md#paired-cadence-screen-across-rate-sensor-polarity-patterns-and-clipping-severity).
   The profile catalog and opt-in calibration diagnostics pass the current full
   local CI sequence (453 tests, 90.50% coverage, strict Pyright clean, and alpha
   contract compatibility). A large run also exposed
   checkpoint-write memory pressure:
   serializing the entire checkpoint as one formatted string failed under a
   32-worker run, while resuming with 16 workers completed. Checkpoint output now
   streams completed cases individually, but broader memory/worker scaling still
   needs measurement. See the
   [fresh moderate-noise convention/cadence screen](VALIDATION.md#fresh-moderate-noise-full-convention-screen-across-sampled-rate-families-and-cadences),
   [paired low/high-noise extension](VALIDATION.md#paired-lowhigh-noise-cadence-extension),
   and [paired cadence jitter/dropout factorial](VALIDATION.md#paired-cadence-factorial-with-jitter-and-dropout),
   [paired cadence channel-latency factorial](VALIDATION.md#paired-cadence-factorial-with-channel-latency),
   [paired cadence rate-sensor calibration factorial](VALIDATION.md#paired-cadence-factorial-with-rate-sensor-calibration-defects),
   [paired polarity/clipping factorial](VALIDATION.md#paired-cadence-factorial-with-rate-sensor-polarity-and-clipping-defects),
   and [expanded polarity/clipping severity screen](VALIDATION.md#paired-cadence-screen-across-rate-sensor-polarity-patterns-and-clipping-severity).
2. The repeatable out-of-model screen covers mount rotations, per-axis bias/scale/polarity,
   clipping, and three mixed channel-latency/calibration cases. Ten fresh Fourier seeds
   yielded 17/220 recommendations and 203 abstentions; an opt-in disabled-guard comparison
   yielded 32 recommendations, all correct in that cohort. A paired three-seed screen
   across Fourier, piecewise-linear, and maneuver-burst motion produced zero wrong
   recommendations; all 27 mixed cases abstained with truth retained. Keep the current
   cutoff as a conservative fail-safe, not a calibrated threshold: a separate synthetic
   regression shows it can prevent a false recommendation, while these pilots expose its
   coverage cost. A fresh five-motion/five-noise-seed replication adds noise-free acquisition
   controls and all three OU-noise/jitter/dropout tiers: 5,625 noisy trials produced 419
   recommendations (all correct in this selected cohort), 5,206 abstentions, and no wrong
   recommendations. Truth was among the data-winning classes in only 3,374/5,625
   cases, though it remained in evidence in 5,389/5,625. Recommendation coverage varied
   substantially by motion family, and the earlier small pilot's perfect truth retention on mixed
   defects did not replicate. A pinned axis-polarity case now verifies the existing large-residual
   and ambiguity warnings plus abstention: truth ranks 16th and falls outside the evidence set.
   All 236 evidence misses in this noisy cohort exceeded the generic residual-warning threshold,
   though four low-tier data-winner misses remained evidence contenders below that threshold.
   These paired trials are not field evidence or a false-recommendation rate. Do not change
   tie-breaks or thresholds from this synthetic cohort. Next decide whether per-axis calibration
   estimation belongs in scope and evaluate any extension against independently labeled logs. See
   the [sensor-mismatch validation](VALIDATION.md#fresh-seed-sensor-mismatch-screen),
   [three-tier pilot](VALIDATION.md#paired-three-tier-sensor-mismatch-screen-with-jitter-and-dropout),
   and [fresh-seed replication](VALIDATION.md#fresh-seed-replication-with-noise-free-acquisition-controls).
3. Repeat sampled-rate Euler and observed-cache timing/memory runs across more
   cadences and on a controlled, low-load host; the local five-repeat seed probe
   and Python 3.11 process comparison showed substantial timing instability.
   Matched 10,001-sample process-RSS runs now cover Python 3.11/3.12/3.14 at
   25 ms and Python 3.14 at 50/100 ms. Avoid promising long-log support until
   broader, independent measurements justify it. A current-worktree three-repeat
   25 ms rerun on Python 3.14.6 had a 1.40x median observed-cache speedup with
   1.6% higher median peak working set; timing ranges remained wide and all runs
   were on one synthetic profile and host. See the [three-repeat performance
   follow-up](PERFORMANCE.md#three-repeat-25-ms-long-recording-follow-up).
4. Confirm the expanded Python 3.11–3.14 Linux/Windows release matrix passes on
   GitHub, then test an actual `0.0.1` prerelease artifact on TestPyPI before a
   public package release.
5. Characterize sparse-anchor recovery across anchor cadence, independent
   orientation/rate noise, timing perturbations, and distinct motion families.
   The repeatable 900-evaluation screen now covers five seeds per family,
   regular and 20%-dropout anchors, bounded timestamp jitter, one relative
   channel-value latency pair, a +1000 ppm shared clock scale, and clean/moderate
   noise. Truth remained in both winner and evidence sets throughout; 174
   finest-cadence cases recommended correctly, six abstained, and every coarser
   case abstained, with no wrong recommendations. These are paired synthetic
   evaluations over 15 trajectories, not 900 independent trajectories. Next
   sweep timing-error ranges and validate with independently labeled multi-system
   data. Keep synthetic sensitivity separate from the external evidence gate
   and do not tune thresholds; see [sparse-anchor validation](VALIDATION.md#sparse-orientation-anchors-with-dense-rates).
