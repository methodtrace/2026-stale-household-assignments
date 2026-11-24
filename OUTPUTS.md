# Output columns: exits, reference sensitivities and the anchor diagnostics

This file defines the columns of the files below. Each is written to `results/development/` and `results/confirmation/` of the work directory and shipped in `recorded/nc_development_counties/` and `recorded/nc_confirmation_counties/` under the name in brackets (`recorded/README.md`). The development `mover_timing.csv` has cells under 10 clients and is checked by SHA-256 only.

- Everything here is at the 2016 build (t = 2016-11-08), lag 0, over the run's counties.
- k is the age of the evidence in years, 2, 4, 6 or 8, with s = t + k.
- A share is a plain fraction between 0 and 1, never a percent. Each share names its numerator and denominator below.
- Counts are numbers of clients unless stated otherwise.
- "Persons with truth" are clients registered at s, active or inactive.
- "Exact" means the returned set equals the true co-resident set at s, with the client excluded. When the returned set is empty, exact means the client lives alone at s. Returned members who are absent at s count as not co-resident.

## exit_counts.csv (exit_counts_by_reason.csv)

Counts clients by (policy, k, class, outcome, outcome_active). The rows cover the whole client file, so summing `n` over class and both outcome columns gives the file size.

| Column | Definition |
|---|---|
| policy | `last_known` is the person anchor: the client's own cluster at t. `never_rebuilt` is the address anchor: the t cluster at the address that the client's address at s matches. |
| class | `TRUTH` is a person with truth. An exit (a client with no registration at s) is classed by the reason on the client's last row at or before s: `RD` deceased, `RS` moved out of state, `RL` moved within the state (RL or RH), `RM` removed after inactivity, `DUP` duplicate (R2, RG, DU), `OTHER`, and `AI` for an exit whose last row was still active or inactive. |
| outcome | For TRUTH: `right` is answered and exact, `wrong` is answered and not exact, `none` is not answered (address anchor only). For exits: `none` is not answered. Every unmatched exit gets `none`, the deceased included. A matched exit is `wrong` in four cases: it is RD; it is RS and a returned member other than the client is registered at s; it is RL and a returned member is still at the returned address at s; or (any class) a returned member is recorded deceased at s, or the returned members sit at two or more addresses at s. Every other exit is `unknown`, or `unknown_alone` when the returned set holds only the client. |
| outcome_active | Same as `outcome`, except that the RS and RL checks count only returned members with an active row at s. For TRUTH rows it equals `outcome`. |
| n | Number of clients. |
| n_rm_mover_wrong | Among the counted rows, RM exits with outcome unknown that the RL rule would make wrong, i.e. if RM exits were treated as movers. |

## three_outcome.csv (right_wrong_unknown_bounds.csv)

One row per (policy, k). Every share has the whole client file `n_file` as its denominator, except the two `_rd_excluded` columns, whose denominator is `n_file` minus the deceased exits.

| Column | Numerator |
|---|---|
| right_lo | right (persons with truth only, since exits are never right) |
| right_hi | right + every unknown exit |
| right_hi_rm_movers | right + unknown − n_rm_mover_wrong |
| right_lo_rd_excluded | right, with denominator n_file − RD exits |
| right_hi_rd_excluded | right + unknown, with denominator n_file − RD exits |
| wrong_lo | wrong |
| none | not answered |
| unknown | unknown and unknown_alone exits |
| exits | all exits |
| wrong_lo_active | wrong under outcome_active |
| unknown_active | unknown under outcome_active |
| right_hi_active | right + unknown under outcome_active |
| rd_share | deceased exits (class RD) |

## ref_sensitivity.csv (reference_sensitivity.csv)

One row per k, person anchor, over persons with truth who have a single active-or-inactive registration row at s. Clients with two rows at s are left out, which is 13 clients at k = 2 in the confirmation counties.

| Column | Definition |
|---|---|
| n | Clients scored. |
| base | Exact share, with denominator n. |
| ghost | Exact share when inactive members are dropped from both sets. Denominator n. |
| n_s2, slag, base_on_s2 | Truth at the January 1 snapshot after s. slag and base_on_s2 have denominator n_s2. |
| n_v16, tlag | Clients who voted on 2016-11-08. tlag is base among them, with denominator n_v16. |
| n_relinked, exact_relinked, reissue | Clients without truth whose NCID was re-issued. reissue is (exact + relinked exact) / (n + n_relinked). |
| n_mixed, unitkey_mixed_dropped | Clients whose building is mixed (some registrants have a unit or suffix and some do not). unitkey_mixed_dropped is base without them, with denominator n − n_mixed. |
| n_agein, agein_dropped | agein_dropped is the exact share after removing, from both sets, members who were under 18 at t (age at s minus k is below 18). Denominator n. n_agein counts the clients whose true set lost at least one such member. |
| n_v16_all, tlag_all | Clients who voted on 2016-11-08 and whose every returned member did too. tlag_all is base among them, with denominator n_v16_all. |

## matcher_answer_diagnostics.csv (same name)

One row per (rule, link, k), over persons with truth.

- A stayer is a client whose address at s maps to the client's own t cluster (the address anchor's key equals the person anchor's key), as in sigma.
- eP is person-anchor exact.
- `rule` is either `probabilistic_matcher_w<theta>` or `build_alone`:
  - `probabilistic_matcher_w<theta>`: the probabilistic matcher answers when its weight is at least theta (`wnone` means any weight). V means the matcher's returned set is exact.
  - `build_alone`: answer only clients whose t cluster holds no one else. The returned set is the person anchor's.
- `link` takes three values:
  - `all`
  - `correct`: answered clients whose linked 2016 record is the client
  - `wrong`: answered clients linked to another record

  Unanswered clients appear in every link row, so n, n_unans_stay and kS_unans repeat across link rows.

| Column | Definition |
|---|---|
| n | Clients in the row. |
| n_ans | Answered clients. |
| n_ans_stay, n_ans_move | Answered stayers and answered movers. |
| eV, eV_stay, eV_move | Answered and exact (V), overall and among stayers and movers. |
| eP_ans_stay, eP_ans_move | Person anchor exact among answered stayers and among answered movers. |
| n_unans_stay, eP_unans_stay | Unanswered stayers, and person anchor exact among them. |
| answer_rate | n_ans / n |
| s_theta | n_ans_stay / n_ans, the stayer share among answers |
| V | eV / n_ans, validity |
| kS_ans, kS_unans | eP_ans_stay / n_ans_stay and eP_unans_stay / n_unans_stay. Their gap tests condition (i). |
| mu_ans | eP_ans_move / n_ans_move. Tests condition (ii). |
| V_stay, V_move | eV_stay / n_ans_stay and eV_move / n_ans_move. |

## mover_timing.csv (same name)

Person anchor, over persons with truth who are not stayers (movers, plus clients whose address at s matched no t cluster). The move snapshot is the first snapshot after t at which the client is registered and none of the client's unit keys equals the client's t cluster key. These are the keys sigma compares, so a county readdressing counts as a move here.

| Column | Definition |
|---|---|
| table | `by_time` gives one row per (k, years_since_move). `cohort` follows the movers at k = 2 (cohort_k = 2) to k = 2, 4, 6 and 8, among those with truth at each k. |
| years_since_move | Whole years from the move snapshot to s, at most k. Empty means no move snapshot was found: the client kept the t unit key at every snapshot, but the address-only match at s did not return the client's t cluster. |
| n | Clients. |
| exact | Person anchor exact. |
| exact_lone | Exact, and the client was alone at t and is alone at s. |
| exact_group | exact − exact_lone: the t household unchanged at a new address. |
| n_back | cohort only. Cohort members who are stayers again at that k. |
| mu | exact / n |

## readdress_ablation.csv (readdressing_ablation.csv)

One row per k, over persons with truth who have a single registration row at s.

- The `_tk` columns use the reference household key with the readdressing rule. A building that a county readdressed maps to its canonical building, as in every other output.
- The `_ak` columns drop the rule: co-residents are the people who share the registered unit key at s.
- sigma does not depend on the rule.

| Column | Definition |
|---|---|
| n, n_stay, n_matched | Clients scored, stayers, and clients the address anchor matched. |
| n_affected, affected_share | Clients whose true set differs between the two keys. The share has denominator n. |
| eP_*, kP_* | Person anchor exact. Denominator n. |
| eS_*, kS_* | Person anchor exact among stayers. Denominator n_stay. |
| eM_*, mu_* | Person anchor exact among non-stayers. Denominator n − n_stay. |
| eA_*, kA_* | Address anchor exact among matched clients. Denominator n_matched. |
| sigma | n_stay / n |

## mobility_calibration.csv (nc_statewide/mobility_vs_acs_by_age_band.csv)

One-year mobility, statewide, for adults.

- Year Y compares the January 1 snapshots of Y and Y + 1 with ACS year Y. ACS has no standard 2020 one-year release, so 2020 uses the mean of 2019 and 2021.
- The base is persons with an active row and a unit key on January 1 of Y who are aged 18 or over.
- Band is the ACS B07001 age band at Y. Recorded ages above 200 are left out.
- Register moves cluster in election years, because registrants update their address when they vote. Single years swing, so compare means over a four-year cycle. Two bands are derived: `18+` is all adults, and `std2016` weights each band's rates by the age mix of the run's 2016 clients at t.

| Column | Definition |
|---|---|
| n | Base persons. Empty for std2016. |
| moved | Base persons registered on January 1 of Y + 1 who share no reference household key (tk) with Y. A county readdressing is not a move. |
| exited | Base persons with no active or inactive row on January 1 of Y + 1. |
| reg_moved, reg_exited | moved / n and exited / n. |
| reg_moved_or_exited | Their sum. |
| acs_moved | ACS adults in a different house one year earlier, divided by ACS adults, for the band (B07001: total minus same house, over total). |
| gap | acs_moved − reg_moved_or_exited. A positive gap estimates the one-year share of adults whose move the register does not show within the year. |
