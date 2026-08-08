"""Removed in FMU-5.

The heat exchanger is modelled by driving the Buildings ConstantEffectiveness /
PlateHeatExchangerEffectivenessNTU FMUs (see ``heat_exchanger_fmu.py``). The
former Python effectiveness-NTU surrogate (``predict_hex`` / ``calibrate_hex``)
re-implemented the device physics in Python and was removed -- device thermal
physics must come from the FMU. This module is intentionally left empty.
"""
