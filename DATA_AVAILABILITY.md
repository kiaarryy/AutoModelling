# Data availability

## Public Site D

The end-to-end public case uses the LBNL *Fault Detection and Diagnostics*
chiller-plant dataset, DOI **10.25984/1881324**. Download the fault-free
`ChillerPlant.csv` archive from the dataset landing page and keep it outside the
repository. The source dataset is licensed CC BY 4.0 by its provider.

- Dataset DOI: https://doi.org/10.25984/1881324
- U.S. Open Energy Data Initiative record:
  https://data.openei.org/submissions/5763

The published archive has the outdoor dry-bulb and wet-bulb columns exchanged.
`scripts/preprocess_lbnl_swap.py` performs only that row-wise correction and,
in a separate invocation, the 30-minute subsampling used in the paper.

## Commercial Sites A-C

The raw BMS exports cannot be redistributed because they are covered by site
confidentiality agreements. To make the reported claims auditable without
releasing identifiable plant histories, `evidence/` contains:

- per-device selection and independent-test metrics;
- status, refusal reason, excitation, and observability ledgers;
- aggregate attrition and cross-site tables;
- the representative measured/simulated series used in the validation figure;
- figure source tables and time-and-motion logs.

Device and site identifiers are anonymised. No raw BMS export, credential,
access token, proprietary FMU binary, or machine-specific data path is included.

## Integrity boundary

The manuscript reports 112 attempted devices: 58 reach an independent test
score and 54 do not. “Scored” is deliberately distinct from “accepted”; the
released ledgers preserve the error, skill, evidence grade, and refusal reason
needed to interpret each outcome.

