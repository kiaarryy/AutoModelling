# System Component Modelica Sources

This directory holds table-free, signal-level Modelica sources intended for
generated system-level assembly. Each component receives calibrated parameters
from autofmu and explicit operating inputs from a generated top-level model.
None of these components reads `CombiTimeTable` data.

These are system assembly components, not the original calibration FMU wrappers.
They are designed as stable "puzzle pieces" for an agent-generated loop model.
Fluid-port adapters can be added later around the same parameter contracts.

## Components

| Device | Type | File |
| --- | --- | --- |
| chiller | EIR | `chiller/EIRComponent.mo` |
| chiller | EEIR | `chiller/EEIRComponent.mo` |
| chiller | Carnot | `chiller/CarnotComponent.mo` |
| cooling tower | Merkel | `cooling_tower/MerkelComponent.mo` |
| cooling tower | YorkCalc 27 | `cooling_tower/YorkCalc27Component.mo` |
| heat exchanger | ConstantEffectiveness | `heat_exchanger/ConstantEffectivenessComponent.mo` |
| heat exchanger | PlateEffectivenessNTU | `heat_exchanger/PlateEffectivenessNTUComponent.mo` |
| pump | empirical_power | `pump/PumpEmpiricalPowerComponent.mo` |
| pump | mover | `pump/PumpMoverComponent.mo` |

## Interface Pattern

- chiller components use `TEvaEnt_C`, `TEvaLvgSet_C`, `TConEnt_C`,
  `mEva_flow_kg_s`, `mCon_flow_kg_s`, and `y` inputs; they output leaving
  temperatures, cooling, power, COP, and PLR.
- cooling tower components use `Tin_C`, `Twb_C`, `m_flow_kg_s`, `y`, and
  `TRan_C` inputs; they output outlet temperature, heat rejection, fan power,
  and approach diagnostics.
- heat exchanger components use side-1/side-2 inlet temperatures and mass flows;
  they output side outlet temperatures, heat flow, and effectiveness.
- pump components use speed and, for the empirical power type, measured or
  upstream-provided mass flow; they output power and hydraulic diagnostics.
