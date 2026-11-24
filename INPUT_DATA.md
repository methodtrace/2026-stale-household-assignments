# Input data

The pipeline reads 41 voter registration snapshots published by the North Carolina State Board of Elections (NCSBE): the
40 registered snapshots below and one later file. They are public records. This repository contains no records and no
record-level file; it contains only the code, the expected file checksums, and aggregate results.

## Where the files come from

Each snapshot is one ZIP file, published at

    https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_YYYYMMDD.zip

How you obtain them does not matter. Put the 41 ZIP files, with their published names, in one directory and pass it as
`--snapshots`. `inputs/snapshots.tsv` and `inputs/later_file.tsv` list every file with its size and SHA-256.
`reproduce_all.py` checks each file against that list before it is used and stops on a mismatch; `--allow-changed-inputs`
runs anyway, and the outputs are then not expected to match the recorded ones.

| Snapshot dates used | Count |
|---|---|
| Election Days: 2008-11-04, 2010-11-02, 2012-11-06, 2014-11-04, 2016-11-08, 2018-11-06, 2020-11-03, 2022-11-08, 2024-11-05 | 9 |
| January 1 of every year, 2009 to 2026 | 18 |
| Odd-year Novembers: 2009-11-03, 2013-11-05, 2015-11-03, 2017-11-07, 2019-11-05, 2021-11-02, 2023-11-07, 2025-11-04 (no November 2011 file is published) | 8 |
| 2019 list-maintenance series: 2019-04-30, 2019-05-14, 2019-07-09, 2019-09-10, 2019-10-08 | 5 |

The exact list is `inputs/snapshots.tsv` (and `stale_assignments/store.py`, `SNAPSHOTS`).

The later file, `VR_Snapshot_20261103.zip` (2026-11-03), is listed with its size and SHA-256 in `inputs/later_file.tsv`.
It is read with the same layout and rules as the other files, and it scores the builds of the registered snapshots at a later
date (paper sections 6.4 and 8.1). Put it in the `--snapshots` directory with the
other 40.

## File layout the code expects

Each ZIP holds exactly one member: a text file encoded as UTF-16 (little-endian with a byte-order mark), lines ending in
CR LF, fields separated by tabs, one header line, then one line per registration record. The code reads the member as a
stream; it is never fully decompressed to disk.

Columns are read by position, not by header name. The expected order is the 90 columns below; a line with 91 or 92 fields
is treated as misaligned and dropped, and a line with more is rejected, as are lines with a broken quoted field.

    snapshot_dt county_id county_desc voter_reg_num ncid status_cd voter_status_desc reason_cd voter_status_reason_desc
    absent_ind name_prefx_cd last_name first_name midl_name name_sufx_cd house_num half_code street_dir street_name
    street_type_cd street_sufx_cd unit_designator unit_num res_city_desc state_cd zip_code mail_addr1 mail_addr2 mail_addr3
    mail_addr4 mail_city mail_state mail_zipcode area_cd phone_num race_code race_desc ethnic_code ethnic_desc party_cd
    party_desc sex_code sex age birth_place registr_dt precinct_abbrv precinct_desc municipality_abbrv municipality_desc
    ward_abbrv ward_desc cong_dist_abbrv cong_dist_desc super_court_abbrv super_court_desc judic_dist_abbrv judic_dist_desc
    NC_senate_abbrv NC_senate_desc NC_house_abbrv NC_house_desc county_commiss_abbrv county_commiss_desc township_abbrv
    township_desc school_dist_abbrv school_dist_desc fire_dist_abbrv fire_dist_desc water_dist_abbrv water_dist_desc
    sewer_dist_abbrv sewer_dist_desc sanit_dist_abbrv sanit_dist_desc rescue_dist_abbrv rescue_dist_desc munic_dist_abbrv
    munic_dist_desc dist_1_abbrv dist_1_desc dist_2_abbrv dist_2_desc confidential_ind cancellation_dt vtd_abbrv vtd_desc
    load_dt age_group

Only these 19 columns are used:

| Column | Use |
|---|---|
| snapshot_dt | snapshot date (first 10 characters, YYYY-MM-DD) |
| county_id | county of residence |
| ncid | the state's person identifier; used only to build the reference households and to score, never by a simulated matcher |
| status_cd, reason_cd | active or inactive status; removal reason (death, moved out of state) for exit classes |
| last_name, first_name, midl_name, name_sufx_cd | kept only as FarmHash fingerprints of the uppercased, normalized names |
| house_num, half_code, street_dir, street_name, street_type_cd, street_sufx_cd, unit_num, zip_code | residential address; building and unit keys |
| age | age; registrants flagged as minors at the build are never clients (they remain co-members) |
| registr_dt | registration date (tie-break among a person's rows) |

No column describing race, ethnicity, party, sex, phone or mailing address is read.

## Parsing rules

These reproduce the warehouse load the analysis was specified with:

- A field that is exactly empty is null. Every used text field has control characters and double quotes removed, is
  trimmed and is uppercased before any comparison.
- Quoting. In every file except the five Election Day files of 2016 to 2024, a field that starts with a double quote is a
  quoted field: it runs to the next lone quote (two quotes stand for one) and may contain tabs. The closing quote may be
  followed by spaces (the 2012-11-06 file pads fields to a fixed width), then must be followed by a tab or the end of the
  line; otherwise the line is a bad record and is dropped. Any other quote character is literal. In the Election Day files
  of 2016, 2018, 2020, 2022 and 2024, every tab separates fields and a quote is an ordinary character, because the warehouse
  loaded those five files in a separate batch without quote handling (`snapshots.UNQUOTED`). The two rules give different
  results on one line in all 40 files: a 2024-11-05 record whose mailing address opens with a quoted word followed by more
  text. Lines cannot span newlines.
- A line with one or two fields more than the 90 is misaligned and dropped; a line with more is rejected. A line whose
  snapshot date does not parse is dropped.
- Across the 40 files these rules drop 0 to 28 lines per file and 51 in all, out of 556,508,644: two bad quoted
  lines in each file from 2008-11-04 to 2013-01-01 and one in each of the last three; 20 lines with too many fields and 8
  misaligned lines in 2013-11-05; one misaligned line in 2024-11-05 (and one line whose date does not parse). The counts for
  each file are in `results/prep_manifest.json` (`stage1_lines`).

## Other inputs

### Voter history (one column of ref_sensitivity.csv)

`tlag` in `ref_sensitivity.csv` is the person-anchor exact share among clients who voted in the 2016 general election.
It reads the NCSBE statewide voter history file, `ncvhis_Statewide.zip`, from
`https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvhis_Statewide.zip`. The ZIP holds one tab-separated text file, every field
in double quotes, one row per voter and election (about 5.8 GB unpacked). The code reads two columns, `election_lbl`
(MM/DD/YYYY) and `ncid`, and keeps the distinct NCIDs with a row for 11/08/2016 (about 4.77 million). Nothing else is
read or kept.

NCSBE replaces this file every week. Two weekly copies disagree on a few hundred of the NCIDs recorded for 2016-11-08, so
the results depend on the copy. Pass the copy with `--voter-history FILE`; the run records its size and SHA-256 in
`results/prep_manifest.json` (`voter_history`), and when `inputs/voter_history.tsv` names a copy, the run stops if the file
differs (unless `--allow-changed-inputs`). Without the option, `tlag` and `n_v16` are left blank.

### North Carolina address points (unitkey_county.csv, unitkey_statewide.csv)

The unit-key audit compares the voter building and unit keys with the NC OneMap AddressNC address points
(`https://services.nconemap.gov/secure/rest/services/AddressNC/NC1Map_Addresses/FeatureServer/0`, about 6.3 million
points). `fetch_address_points.py OUT_DIR` downloads their attributes (no geometry) into gzip CSV files of 50,000 object
IDs and writes `OUT_DIR/SHA256SUMS`. The code reads county FIPS, house number and suffix, the street name parts, ZIP code,
building and unit; it keeps per-county counts only.

The service is live and has no versions. Download once and copy the directory to every machine; pass it with
`--address-points DIR`. The run checks the files against `SHA256SUMS`, records the SHA-256 of `SHA256SUMS` in
`results/prep_manifest.json` (`address_points`), and stops if `inputs/address_points.tsv` names a different copy. The audit
reads the full address tables of 2016-11-08, 2018-11-06, 2020-11-03, 2022-11-08, 2024-11-05 and 2026-01-01 before stage
2 slims them, so the option must be given on the first run in a work directory. Without it the two unitkey files are not
written.

### American Community Survey mover rates (acs_movers.csv)

`inputs/acs_nc_movers.tsv` holds published Census Bureau aggregates for North Carolina (state FIPS 37) from the ACS 1-year
estimates, 2008 to 2019 and 2021 (2020 has no standard 1-year release): persons one year and over living in a different
house one year ago, same city and different city, and the total population. They were read from the BigQuery public
dataset `bigquery-public-data.census_bureau_acs` (tables `state_YYYY_1yr`, columns `different_house_year_ago_same_city`,
`different_house_year_ago_different_city`, `total_pop`). `acs_movers.csv` divides the movers by the total population.

### Ohio and Washington voter lists (other_states/, household_change/transfer/)

Section 9 and the household-change transfer read five statewide lists, named with their sizes and SHA-256 in
`inputs/other_states.tsv`: Ohio as of 2017-07-18 (one file) and 2026-10-23 (four files by county range), and Washington as
of 2017-06-30, 2018-06-30 and 2026-10-01 (one file each). They come from the Ohio Secretary of State and the Washington
Secretary of State under each state's published voter-file terms and are not redistributed here. The states publish the
current list only, so earlier dates come from copies kept when they were current. Put the files under one directory at
the relative paths in `inputs/other_states.tsv` and pass it with `--other-states DIR`; the run stops if a file's size or
SHA-256 differs (unless `--allow-changed-inputs`). Without the option the other-state results are not written.

Each file is gzip text with a header line. Ohio: comma separated, every field in double quotes, 104 columns (2017) or 140
(2026). Washington: no quoting; pipe separated with 41 columns (2017), tab separated with 40 (2018), pipe separated with
36 (2026). A line that does not parse or has more fields than its layout is skipped and counted in the run log and in
`results/prep_manifest.json` (`other_states_read`). The code reads by column position: voter ID, county, last, first and
middle name, birth date or year, status, and the residential address (Ohio: one street line, unit and ZIP; Washington:
house number and fraction, street name, type, directions, unit type, unit and ZIP). Columns and positions are in
`stale_assignments/other_registers.py` (`LAYOUTS`). Active and inactive registrants are kept (Ohio `ACTIVE` and
`CONFIRMATION`; Washington `A`, `I`, `Active`, `Inactive`). The normalized tables stay under `WORK/tables/other_states`.

## What the pipeline keeps

Stage 1 writes one Parquet table per snapshot with 28 typed columns (dates, county, NCID, status, age, normalized address
parts, building and unit keys, name fingerprints). Stage 2 and later stages write only keys, fingerprints, counts and
scores. All of it stays in the working directory you choose (`--work`); delete it when you are done. The released outputs
are statewide or multi-county aggregates.

Disk: the 40 ZIP files take 39 GB; the working directory peaks at about 25 GB (`--discard-inputs` deletes each ZIP once its
table is built, so only one is on disk at a time). Memory: the development counties peaked at 11.7 GB on a 4-core Linux
machine; the 60 confirmation counties need more.


### American Community Survey mobility by age (mobility_calibration.csv)

`inputs/acs_nc_mobility_by_age.tsv` holds published Census Bureau estimates for North Carolina from ACS 1-year table B07001 ("Geographic mobility in the past year by age"), survey years 2010 to 2019 and 2021 to 2024. There is no standard 2020 one-year release.

Columns:

- `year`
- `age_band`: the 13 adult bands of B07001, 18-19 to 75+
- `population`: the B07001 total for the band (variables 004E to 016E)
- `same_house_1yr_ago`: the band's "same house 1 year ago" count (variables 020E to 032E)

The values were read from the Census Data API (`https://api.census.gov/data/<year>/acs/acs1?get=group(B07001)&for=state:37`) on 2026-11-03. They are published aggregates, not microdata. `mobility_calibration.csv` compares the register's one-year move and exit rates with `1 - same_house_1yr_ago / population` (OUTPUTS.md).
