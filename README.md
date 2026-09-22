# superhot-wellbore

Code accompanying:

> Scott, S.W. (2026), Thermo-hydraulic drivers of superhot geothermal well performance, *Geothermics*, 141, 103784. https://doi.org/10.1016/j.geothermics.2026.103784

## Description

Python framework coupling three components for modeling single-well power output from superhot geothermal systems:

1. **Reservoir model** — Steady-state radial Darcy flow computing pressure drawdown and mass flow rate for a given reservoir pressure, temperature, and transmissivity.
2. **Wellbore model** — Thermohydraulic simulator integrating pressure and enthalpy gradients from bottomhole to wellhead, including frictional losses, gravitational head, kinetic energy, and conductive heat loss to the formation.
3. **Power cycle model** — Binary (water-based Rankine) or flash steam cycle analysis based on the predicted wellhead fluid state.

The framework is applied across the full range of superhot conditions observed globally (375–600 °C, 15–45 MPa) to quantify how reservoir pressure, temperature, and transmissivity jointly control deliverable power.

## Requirements

- Python 3.8+
- [CoolProp](http://www.coolprop.org/) — IAPWS-95 equation of state for water
- [iapws](https://github.com/jjgomez/iapws) — IAPWS-97 backward equations for near-critical routing
- NumPy
- SciPy
- Matplotlib (figure scripts only)

## Installation

```bash
pip install .
```

For development, install in editable mode with the optional extras:

```bash
pip install -e ".[dev]"
```

The extras are:

| Extra | Adds |
|-------|------|
| `figures` | Matplotlib, needed to run the scripts in `examples/` |
| `test` | pytest, needed to run the test suite |
| `dev` | Both of the above |

## Project layout

```
superhot-wellbore/
├── src/superhot_wellbore/       # the installable package
│   ├── wellbore_physics.py
│   ├── reservoir.py
│   ├── power_cycle.py
│   └── client/                  # production-history interface used by GEOPHIRES
├── examples/                    # figure scripts from the manuscript
├── tests/                       # pytest suite
└── pyproject.toml
```

## Contents

### Core modules

| Module | Description |
|--------|-------------|
| `superhot_wellbore.wellbore_physics` | Wellbore pressure and enthalpy gradient integration (Eqs. 4–5 in manuscript) |
| `superhot_wellbore.reservoir` | Radial Darcy flow model, depth–pressure scaling, and reservoir–wellbore coupling via bisection |
| `superhot_wellbore.power_cycle` | Binary and flash power cycle analysis with Baumann wet-stage efficiency |
| `superhot_wellbore.client` | Stable interface exposing the coupled model as a production history; used by the GEOPHIRES superhot production wellbore model |

### Figure scripts

| File | Figure | Description |
|------|--------|-------------|
| `examples/figure4_sensitivity_U_roughness.py` | Fig. 4 | Sensitivity to casing roughness and heat-loss coefficient |
| `examples/figure5_calibrate_iddp1.py` | Fig. 5 | IDDP-1 deliverability curve calibration |
| `examples/figure6_iddp1_profiles.py` | Fig. 6 | Downhole pressure, temperature, and enthalpy profiles |
| `examples/figure7_pressure_parametric.py` | Fig. 7 | Pressure parametric analysis (450, 475, 500 °C) |
| `examples/figure8_temperature_parametric.py` | Fig. 8 | Temperature parametric analysis (20, 30, 40 MPa) |
| `examples/figure9_transmissivity_parametric_analysis.py` | Fig. 9 | Transmissivity parametric analysis |
| `examples/figure11_whpsweep.py` | Fig. 11 | Wellhead pressure sweep for selected scenarios |
| `examples/figure12_velocity_diameter.py` | Fig. 12 | Wellhead velocity and mass flow for two casing sizes |

## Usage

### As a library

```python
from superhot_wellbore.reservoir import solve_flow_for_whp
from superhot_wellbore.power_cycle import power_cycle_analysis
```

### Figure scripts

Each figure script can be run independently. For example:

```bash
python examples/figure7_pressure_parametric.py
```

Results are cached as `.pkl` files to avoid rerunning simulations. Delete the cache file to force a fresh run.

### GEOPHIRES

[GEOPHIRES](https://github.com/NREL/GEOPHIRES-X) includes a *Superhot Production
Wellbore Model* (`Superhot Production Wellbore Model, True`) that runs the coupled
inflow–wellbore model through the `superhot_wellbore.client` package alongside any
GEOPHIRES thermal reservoir model, and then applies its own surface plant and
economics. Install this package alongside GEOPHIRES and set the flag in the input
file; see `example_SHR-4.txt` in the GEOPHIRES examples and the
`geophires_x.SuperhotWellBores` docstring for the parameter mapping.

```bash
pip install geophires-x
pip install git+https://github.com/softwareengineerprogrammer/superhot-wellbore.git
```

The client also runs the package's power cycle (`power_cycle.py`) at every
solved state, configured through the request's `power_cycle` section; the
GEOPHIRES *Superhot Power Cycle* surface plant (`Power Plant Type, 10`) uses it
for electricity generation.

#### Production pumping

The core modules describe a self-flowing well. When a prescribed flow rate
cannot be lifted to the surface by the reservoir pressure alone (a 200 °C
reservoir at 5 km with an 8 MPa drawdown, for instance), the client adds a
downhole production pump on top of the unmodified core
(`superhot_wellbore.client.pump`):

1. The unpumped march decides whether the well *self-flows*: it reaches the
   surface **and** the wellhead pressure is at least
   `pump.min_self_flow_whp_MPa` (or the caller's `pump.target_whp_MPa`).
   A self-flowing well, two-phase or not, is returned exactly as before.
2. Otherwise the pump intake is the shallowest depth on the unpumped profile
   from which the column is single-phase liquid with
   `P ≥ P_sat(T) + pump.npsh_margin_MPa` all the way down to the feedzone
   (the march is upstream-independent, so the unpumped profile *is* the
   lower segment).
3. The pump raises the pressure by `dP` with `h₂ = h₁ + dP / (ρ₁ η)`, and the
   upper segment is marched from the intake to the surface. `dP` is solved so
   that the wellhead pressure meets `pump.target_whp_MPa`, or
   `P_sat(T_intake) + npsh_margin` when no target is given (the stream then
   stays liquid to the wellhead). Pump power per well is `ṁ dP / (ρ₁ η)`.
4. The intake is compared with an ESP envelope (`pump.max_depth_m`,
   `pump.max_intake_temperature_C`). Under `pump.envelope = 'flag'` (default)
   a pump outside it is modelled and reported in `pump_flags`; under
   `'enforce'` the timestep fails with the reason in its message.

The `pump` request section (`PumpConfig`):

| Key | Default | Meaning |
|-----|---------|---------|
| `mode` | `auto` | `never` (fail as before), `auto` (pump only when not self-flowing), `always` |
| `efficiency` | `0.80` | pump efficiency η |
| `npsh_margin_MPa` | `0.3447` | intake margin above `P_sat(T)`; also sets the default pumped WHP |
| `min_self_flow_whp_MPa` | `1.0` | self-flow floor on the unpumped wellhead pressure |
| `max_depth_m` | `1500` | ESP setting-depth limit |
| `max_intake_temperature_C` | `250` | ESP intake-temperature limit |
| `envelope` | `flag` | `flag` or `enforce` |
| `target_whp_MPa` | `None` | pumped wellhead pressure; `None` selects `P_sat(T_intake) + margin` |
| `tolerance_MPa` | `0.01` | convergence tolerance on the pumped WHP |
| `max_dP_MPa` | `60` | upper bound of the pump pressure rise |

Every `TimestepResult` then carries `pumped`, `pump_depth_m`,
`P_pump_intake_MPa`, `T_pump_intake_C`, `dP_pump_MPa`, `pump_power_MWe`,
`self_flow_whp_MPa`, `self_flowing`, `wellhead_phase`, `wellhead_quality`,
`dry_steam_work_MJkg` and `pump_flags` (a subset of
`temperature_limit`, `depth_limit`, `self_flow_below_floor`,
`two_phase_at_sandface`, `pump_outside_envelope`, `no_liquid_intake`, and
`choked_flow` for a prescribed-flow march that reached the local sound
speed); the wellhead values of a pumped timestep are those of the pumped
upper segment.
`power_cycle.dry_steam_specific_work(P)` is the gross specific turbine work of
saturated steam expanded from `P` (the steam share of a two-phase wellhead
stream). The pump stage applies to prescribed-flow solves only: a wellhead
pressure solve (`operating.control = 'whp'`) finds the flow the well delivers
by itself.

In GEOPHIRES these map onto `Superhot Production Pump` (mode),
`Superhot Production Pump Envelope`, `Superhot Production Pump Maximum Depth`,
`Superhot Production Pump Maximum Intake Temperature` and
`Superhot Minimum Self-Flow Wellhead Pressure`, with `Circulation Pump
Efficiency` as the efficiency and `Production Wellhead Pressure` (when
provided) as the target; GEOPHIRES prices the pumps through its own
production-pump cost correlation from the reported pump power and depth.

#### Prescribed inflow

An external reservoir simulator can bypass the Darcy inflow model:
`reservoir.inflow = 'prescribed'` (the transmissivity is then optional)
with `decline.feedzone_profile = [[time_yr, P_feedzone_MPa, h_feedzone_MJkg],
...]` and, optionally, `decline.mass_flow_profile = [[time_yr, kg/s], ...]`.
Both tables are interpolated linearly in time and require
`operating.control = 'flow'`; the wellbore and pump stage run unchanged from
the tabulated sandface state.

The client can also be driven directly from Python:

```python
from superhot_wellbore.client import CoupledWellboreRequest, CoupledWellboreClient

request = CoupledWellboreRequest.from_dict({
    'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0},
    'operating': {'target_whp_MPa': 10.0},
})
profile = CoupledWellboreClient(request).solve_profile()
print(profile.summary())
```

For GEOPHIRES versions without the built-in model, the `superhot-geophires`
command writes a temperature profile and an input deck fragment that GEOPHIRES
reads with its *User-Provided Temperature Profile* reservoir model:

```bash
superhot-geophires template > case.json
superhot-geophires run --request case.json --output-dir out
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests that solve the coupled reservoir–wellbore model are marked `slow`
and take noticeably longer:

```bash
pytest -m "not slow"   # fast checks only
pytest -m slow         # coupled-model solves
```

## Citation

If you use this code, please cite:

```bibtex
@article{Scott2026superhot,
  author  = {Scott, Samuel W.},
  title   = {Thermo-hydraulic drivers of superhot geothermal well performance},
  journal = {Geothermics},
  year    = {2026},
  volume  = {141},
  pages   = {103784},
  doi     = {10.1016/j.geothermics.2026.103784}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
