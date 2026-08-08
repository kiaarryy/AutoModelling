"""Device model engines (L3). To be migrated from FMU_Modelica/scripts Site A
one-off calibrators, de-hardcoded into config-driven engines:
chiller (ElectricReformulatedEIR), cooling_tower (YorkCalc/fan affinity),
pump (SpeedControlled_Nrpm/speed-poly), heat_exchanger (effectiveness-NTU).
Each consumes the L2 modelability gate to pick full_physical vs nominal_only.
"""
