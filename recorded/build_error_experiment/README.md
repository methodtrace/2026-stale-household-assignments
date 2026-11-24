# Build-error experiment

Currency when the build itself is wrong (section 6.4): four build arms scored on the 2018 build.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `currency_by_build_accuracy.csv` | 8 | One row per (arm, s): build accuracy a, kV, kP, their difference and ratio with 99% county-bootstrap intervals, for the exact, noisy, stale and stale-and-noisy builds. | Table S22; section 6.4 | `other_states.build_error_summary` |
| `exact_arm_check_2018.csv` | 2 | The exact arm scored two ways at the 2018 build (n and exact share); the run stops unless they are equal. | Supplement L.3 | `other_states.exact_build_2018` |
