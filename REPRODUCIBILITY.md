# Reproducibility guide

## Levels of reproduction

1. **Software verification:** run the synthetic and golden tests with no
   external data or FMUs.
2. **Evidence audit:** inspect/rebuild the tables and figures from the released
   CSV evidence.
3. **Public-data execution:** download Site D, apply the explicit temperature
   correction, and run the five pipeline stages with locally exported FMUs.

## Expected public-case boundary

The public plant contains three water-cooled chillers, three cooling-tower
cells, and two secondary chilled-water pumps in the released configuration.
Six of these eight configured devices reach an independent score; two are
refused because the retained evidence is insufficient. The output must retain
those refusal records rather than silently substituting measured values or a
passthrough model.

## Provenance outputs

Every run is written below `outputs/runs/<run-id>/`. The manifest records the
configuration hash, code revision, stage, input paths, and generated artefacts.
Later stages refuse to mix outputs produced from a different configuration or
revision.

Key outputs include:

- `attribute/modelability_report.csv`;
- `calibrate/selected_models.csv`;
- independent-test metric tables from `validate`;
- the run manifest and report artefacts.

Use a new `--run-id` after changing a configuration or the code.

## External dependencies

FMU execution requires `fmpy` and locally exported FMUs. Modelica source
interfaces are supplied under `modelica/components/`; compiled FMUs are omitted
because their redistribution and binary compatibility depend on the user's
Modelica library/toolchain. Set `AUTOFMU_FMU_ROOT` to the export root.

## Evidence regeneration

The scripts in `scripts/` rebuild the released cross-site tables and figures
from their declared input artefacts. Some cross-site scripts require the
commercial-site derived run outputs that are represented here by the released
tables; they do not make the confidential raw archives public.

