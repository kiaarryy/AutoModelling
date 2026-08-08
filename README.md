# AutoModelling

Code, configuration templates, derived evidence, and the public-data
reproduction path for:

> **AutoModelling: From BMS Operational Data to Validated Modelica Models for
> HVAC Equipment**

AutoModelling converts building-management-system (BMS) archives into
device-specific Modelica/FMU assets under five explicit contracts:

1. semantic mapping;
2. data readiness;
3. sensor observability;
4. model-type selection; and
5. artefact provenance.

The repository covers chillers, cooling towers, pumps, and heat exchangers.
The cooling-tower, pump, and heat-exchanger formulations are introduced in the
associated article. The chiller equations and identification procedure were
published separately in *Energy*; this repository applies that established
chiller interface inside the same four-family workflow.

## What the reported result means

Across four cooling plants, 58 of 112 devices reach an independent test score;
54 are refused with a recorded reason. A scored device is not automatically an
accepted model: the evidence tables also report test error, skill against a
mean predictor, excitation, observability level, and acceptance status.

The commercial-site raw archives cannot be redistributed. This repository
therefore releases anonymised derived metrics and representative validation
series for those sites, plus an end-to-end path based on the public LBNL
chiller-plant dataset (Site D). See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Repository map

- `src/autofmu/`: ingestion, attribution, observability gates, calibration,
  validation, and reporting.
- `modelica/components/`: table-free component interfaces for all four
  equipment families.
- `configs/fmu/`: external FMU contracts and tunable-parameter mappings.
- `configs/lbnl/`: public Site D adapters and project configuration.
- `evidence/`: manuscript/SI result tables, figures, representative traces,
  and time-and-motion logs.
- `scripts/`: preprocessing and evidence-generation utilities.
- `tests/`: synthetic, golden, contract, and optional external-FMU tests.

## Install and test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

Tests that require proprietary exported FMUs skip when
`AUTOFMU_FMU_ROOT` is unset. The pure-Python and synthetic tests require no
commercial data or Modelica licence.

## Reproduce the public-data case

1. Download the fault-free `ChillerPlant.csv` archive identified in
   [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
2. Apply the documented dry-/wet-bulb correction and 30-minute sampling:

   ```bash
   python scripts/preprocess_lbnl_swap.py ChillerPlant.csv ChillerPlant_fixed.csv
   python scripts/preprocess_lbnl_swap.py --resample 30 ChillerPlant_fixed.csv ChillerPlant_30min.csv
   ```

3. Set the external roots and run the five stages:

   ```bash
   export AUTOFMU_DATA_ROOT=/absolute/path/to/preprocessed
   export AUTOFMU_FMU_ROOT=/absolute/path/to/exported_fmus

   autofmu ingest --config configs/lbnl/project_fleet.yaml --run-id public-demo
   autofmu attribute --config configs/lbnl/project_fleet.yaml --run-id public-demo
   autofmu calibrate --config configs/lbnl/project_fleet.yaml --run-id public-demo
   autofmu validate --config configs/lbnl/project_fleet.yaml --run-id public-demo
   autofmu report --config configs/lbnl/project_fleet.yaml --run-id public-demo
   ```

The external FMUs are not stored in Git because they depend on the Modelica
Buildings Library and the user's export toolchain. Their interfaces and source
components are versioned here. Detailed expected outputs and limitations are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Licence and citation

The software is released under the MIT License. Dataset licences remain with
their respective providers. Cite the associated article and the software
release; citation metadata are provided in [CITATION.cff](CITATION.cff).

