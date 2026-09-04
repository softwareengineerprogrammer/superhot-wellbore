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
| `superhot_wellbore.client` | Stable interface exposing the coupled model as a production history; used by the GEOPHIRES Superhot Wellbore reservoir model |

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

[GEOPHIRES](https://github.com/NREL/GEOPHIRES-X) includes a *Superhot Wellbore*
reservoir model (`Reservoir Model, 9`) that runs the coupled reservoir–wellbore
model through the `superhot_wellbore.client` package and then applies its own
surface plant and economics. Install this package alongside GEOPHIRES and select
the model in the input file; see `example_superhot-wellbore.txt` in the GEOPHIRES
examples and the `geophires_x.SuperhotWellboreReservoir` docstring for the
parameter mapping.

```bash
pip install geophires-x
pip install git+https://github.com/softwareengineerprogrammer/superhot-wellbore.git
```

The client can also be driven directly from Python:

```python
from superhot_wellbore.client import SuperhotRequest, SuperhotWellboreClient

request = SuperhotRequest.from_dict({
    'reservoir': {'P_reservoir_MPa': 30.0, 'T_reservoir_C': 450.0},
    'operating': {'target_whp_MPa': 10.0},
})
profile = SuperhotWellboreClient(request).solve_profile()
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
