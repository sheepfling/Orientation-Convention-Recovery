# Convention-preference research worksheet

The inference engine can preserve a set-valued scientific result while an
application still needs one operational decoder. A preference profile is the
right place for that external policy. Research should therefore answer two
separate questions:

1. Which quotient class is plausible in this source context?
2. Which representative does the source explicitly use when several members
   are exactly observationally equivalent?

Do not answer either question from ecosystem popularity alone. A documented
schema or API contract is stronger evidence than a tutorial, a variable name,
or a convention that is merely common in another tool.

## Research order

For each source, inspect the following in order:

1. Versioned API and schema documentation for the exact artifact or logger.
2. Serialization code, generated schemas, and import/export tests.
3. Reference implementation code that constructs or consumes the rotations.
4. Artifact-role documentation: IMU, camera pose, engine transform, navigation
   state, telemetry, or offline analysis output.
5. Independent ecosystem surveys or representative datasets, used only as a
   weak prior when the first four sources are silent.

## Questions to record

| Dimension          | Research question                                                              | Typical mistake                                                |
|--------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------|
| Quaternion order   | Are components scalar-first or scalar-last in this exact artifact?             | Confusing display order with constructor order.                |
| Attitude direction | Does the rotation map body coordinates to world coordinates, or the reverse?   | Treating an inverse transform as a different physical pose.    |
| Rate frame         | Is angular velocity expressed in body or world coordinates?                    | Inferring the frame from the variable name `omega`.            |
| Rate order/sign    | Which physical axes map to each channel, and are any channels negated?         | Assuming `p,q,r` means the file columns are already canonical. |
| Euler order/mode   | What sequence, intrinsic/extrinsic interpretation, and branch are specified?   | Treating principal ranges as physical motion limits.           |
| Units              | Are angles and rates radians, degrees, turns, or another unit?                 | Using a wrapping cue as proof of clock scale.                  |
| Time               | Are timestamps authoritative, sampled at a fixed period, delayed, or rescaled? | Letting a preferred unit silently determine the clock.         |

## Evidence record

For every claim, record:

- source title, URL/path, version, and access date;
- artifact role and exact field names;
- the quoted or executable claim, paraphrased in plain language;
- whether it selects a quotient class or only a representative;
- whether the claim is hard schema information, a strong prior, or a weak
  heuristic;
- the conflicting sources considered and why one was preferred;
- reviewer and profile version.

The profile `source` should point to this record or its change-controlled
policy document. Relative weights are rankings, not calibrated probabilities.
Use zero only for an explicit exclusion; otherwise leave unsupported values at
neutral weight rather than pretending the source ruled them out.

## Mapping research into a profile

- Put class weights on distinct trajectory-supported quotient classes only when
  the source context supplies evidence about those classes.
- Put exact source conventions such as `scalar_last`, `body`, or a documented
  Euler sequence in `representative_preferences` when they choose a member of
  an exact class.
- Keep a class contender set and the selected representative in the report.
- Treat a profile with no positive support among the contenders as an abstention.
- Re-run the inference after every profile change and compare the data-only
  ranking, evidence contender set, operational class, and representative.

## Profile composition

Profiles may be combined into a new explicit policy with a normalized weighted
geometric mixture. The weights are blend coefficients, not probabilities. Missing
preference values contribute neutral weight `1.0`; an explicit zero class weight
remains an exclusion. A composed profile must retain its member profile IDs,
blend weights, sources, and composition method so the policy can be reproduced.

Composition changes only soft policy weights. It must not merge hard constraints,
manufacture evidence, split an exact quotient class, or be described as a calibrated
posterior unless the resulting policy has been calibrated independently.

## Suggested corpus fields

If we build a research-backed preference corpus, each row should contain:

```text
source_id, source_version, artifact_role, domain, field_name,
attitude_direction, quaternion_order, euler_order, euler_mode, angle_unit,
rate_frame, rate_order, rate_unit, rate_sign, timestamp_contract,
claim_kind, quotient_or_representative, evidence_strength,
source_url, accessed_at, reviewer, profile_version
```

The corpus should remain separate from trajectory evidence and should be
calibrated on held-out sources before its weights are described as probabilities.
