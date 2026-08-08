"""Declarative model-type layer: turn a device's candidate FMU *types* into data.

Each device class (chiller, cooling tower, ...) can model its physics with more
than one Buildings library component -- e.g. a cooling tower as ``Merkel``,
``YorkCalc`` or ``FixedApproach``. autofmu fits every *enabled* candidate
FMU-in-the-loop and selects the best. Historically the per-type behaviour
(start-value names, parameter grids) was hard-coded with ``if model == "..."``
branches in each device module, so adding a type meant editing the engine.

This module moves that behaviour into the candidate's YAML so the engine consumes
it generically. A candidate may declare:

``static_start_values``
    FMU parameters that are fixed per run from the estimated ``base`` nominals or
    a constant, e.g.::

        static_start_values:
          m_flow_nominal: {base: m_flow_nominal}   # base[...] (see device base_values)
          nFan:           {const: 1.0}

``grid``
    the parameter search space, as a product of named ``axes``. Each axis carries
    a list of ``values`` and an ``apply`` map describing how an axis value sets one
    or more FMU parameters::

        grid:
          axes:
            - name: tapp                              # YorkCalc approach scale
              values: [0.25, 0.5, 0.75, 1.0, 1.25]    # scale factors of a nominal
              apply: {TApp_nominal: {base: TApp_nominal, scale: axis, floor: 0.2}}
            - name: cwat                              # Merkel: one axis -> 3 params
              values: [0.75, 1.0, 1.25, 1.5]
              apply:
                "merkel.UACor.cWatFra[1]": {default: 0.1082, scale: axis}
                "merkel.UACor.cWatFra[2]": {default: 1.667,  scale: axis}

An ``apply`` entry is one of:
  * ``{value: axis}``            -> the axis value itself
  * ``{value: <number>}``       -> a constant
  * ``{base: KEY, scale: axis}``-> ``base[KEY] * axis_value`` (``floor``/``ceil`` optional)
  * ``{default: D, scale: axis}``-> ``D * axis_value`` (``floor``/``ceil`` optional)

Adding a new type therefore needs only a YAML candidate (+ an exported FMU and,
for an unusual search shape, a small grid spec) -- no change to the device engine.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Optional, Sequence


def _apply_axis(apply_map: Mapping, axis_value: float, base: Mapping) -> Dict[str, float]:
    """Resolve one axis value into ``{fmu_param: value}`` via its ``apply`` map."""
    out: Dict[str, float] = {}
    for param, spec in apply_map.items():
        if "value" in spec:
            v = spec["value"]
            out[param] = float(axis_value) if v == "axis" else float(v)
            continue
        if "scale" in spec:
            factor = float(axis_value) if spec["scale"] == "axis" else float(spec["scale"])
            if "default" in spec:
                val = float(spec["default"]) * factor
            elif "base" in spec:
                val = float(base[spec["base"]]) * factor
            else:
                val = factor
            if "floor" in spec:
                val = max(float(spec["floor"]), val)
            if "ceil" in spec:
                val = min(float(spec["ceil"]), val)
            out[param] = val
            continue
        raise ValueError(f"grid apply spec for {param!r} needs 'value' or 'scale': {spec!r}")
    return out


def expand_grid(grid_spec: Optional[Mapping], base: Mapping) -> List[Dict[str, float]]:
    """Cartesian product of the declared axes -> list of FMU parameter dicts.

    Returns ``[{}]`` (one empty candidate) when no grid is declared, so callers
    can iterate uniformly.
    """
    axes = list((grid_spec or {}).get("axes", []))
    if not axes:
        return [{}]
    if (grid_spec or {}).get("mode") == "coordinate":
        out: List[Dict[str, float]] = []
        for axis in axes:
            for axis_value in axis["values"]:
                out.append(_apply_axis(axis["apply"], axis_value, base))
        return out
    value_lists = [axis["values"] for axis in axes]
    out: List[Dict[str, float]] = []
    for combo in itertools.product(*value_lists):
        params: Dict[str, float] = {}
        for axis, axis_value in zip(axes, combo):
            params.update(_apply_axis(axis["apply"], axis_value, base))
        out.append(params)
    return out


def assemble_start_values(static_spec: Optional[Mapping], base: Mapping,
                          params: Optional[Mapping] = None) -> Dict[str, float]:
    """Build the static FMU start values (from ``base`` nominals / constants) and
    overlay the per-candidate ``params`` (grid + fitted values)."""
    out: Dict[str, float] = {}
    for name, spec in (static_spec or {}).items():
        if "const" in spec:
            out[name] = float(spec["const"])
        elif "base" in spec:
            out[name] = float(base[spec["base"]])
        else:
            raise ValueError(f"static_start_values[{name!r}] needs 'base' or 'const': {spec!r}")
    if params:
        out.update({k: v for k, v in params.items() if not str(k).startswith("_")})
    return out


# --------------------------------------------------------------------------- #
# Candidate selection ("which types compete")
# --------------------------------------------------------------------------- #
def resolve_enabled(fmu_cfg: Mapping, project_cfg: Optional[Mapping],
                    device_type: str) -> Optional[List[str]]:
    """Effective enabled-candidate name list, or ``None`` meaning *all*.

    Precedence: project config ``fmu_candidates: {<type>: [...]}`` overrides the
    contract's ``enabled_candidates``; absent both, every candidate competes.
    """
    proj = (project_cfg or {}).get("fmu_candidates") or {}
    if device_type in proj and proj[device_type]:
        return [str(n) for n in proj[device_type]]
    enabled = fmu_cfg.get("enabled_candidates")
    return [str(n) for n in enabled] if enabled else None


def select_candidates(candidates: Sequence[Mapping],
                      enabled: Optional[Sequence[str]]) -> List[Mapping]:
    """Filter candidate dicts by ``enabled`` (preserving the requested order).

    ``enabled=None`` returns all candidates in config order. Names not present in
    the contract are ignored (so a typo or a not-yet-exported type is skipped, not
    fatal); the caller can compare lengths to warn.
    """
    if enabled is None:
        return list(candidates)
    by_name = {c["name"]: c for c in candidates}
    return [by_name[n] for n in enabled if n in by_name]
