# Public Site D configuration

This directory maps the public LBNL fault-detection chiller-plant archive onto
the same canonical signal and FMU contracts used for the three commercial
sites. The full fleet configuration includes three chillers, three
cooling-tower cells, and two secondary chilled-water pumps.

## Prepare the external data

Download the fault-free `ChillerPlant.csv` file identified in
`../../DATA_AVAILABILITY.md`. The published outdoor dry-bulb and wet-bulb
columns are exchanged. Correct them explicitly, then retain every 30th row:

```bash
python scripts/preprocess_lbnl_swap.py ChillerPlant.csv ChillerPlant_fixed.csv
python scripts/preprocess_lbnl_swap.py --resample 30 ChillerPlant_fixed.csv ChillerPlant_30min.csv
```

Place `ChillerPlant_30min.csv` in a directory outside the repository and set:

```bash
export AUTOFMU_DATA_ROOT=/absolute/path/to/preprocessed
export AUTOFMU_FMU_ROOT=/absolute/path/to/exported_fmus
```

On PowerShell:

```powershell
$env:AUTOFMU_DATA_ROOT = 'C:\absolute\path\to\preprocessed'
$env:AUTOFMU_FMU_ROOT = 'C:\absolute\path\to\exported_fmus'
```

## Run the five stages

```bash
autofmu ingest --config configs/lbnl/project_fleet.yaml --run-id public-demo
autofmu attribute --config configs/lbnl/project_fleet.yaml --run-id public-demo
autofmu calibrate --config configs/lbnl/project_fleet.yaml --run-id public-demo
autofmu validate --config configs/lbnl/project_fleet.yaml --run-id public-demo
autofmu report --config configs/lbnl/project_fleet.yaml --run-id public-demo
```

Expected boundary: six configured devices reach an independent test score and
two are refused for insufficient retained evidence. A score is not synonymous
with acceptance; inspect error, skill, excitation, and evidence grade in the
generated reports.

The dataset's condenser-side energy balance does not close perfectly. The
pipeline does not silently repair that limitation; reconstructed quantities
remain a lower evidence grade than directly measured targets.
