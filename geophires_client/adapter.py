# -*- coding: utf-8 -*-
"""
GEOPHIRES-X Reservoir Model Adapter
====================================

In-process integration of the superhot-wellbore model into
GEOPHIRES-X. SuperhotWellboreReservoir is a drop-in replacement for a
built-in GEOPHIRES reservoir model: it reads its parameters from the
ordinary GEOPHIRES input file, runs the coupled superhot
reservoir-wellbore model over the GEOPHIRES time vector, and fills in
both the reservoir output temperature and the production well state.

How it plugs in
---------------
GEOPHIRES-X instantiates its reservoir model from the 'Reservoir
Model' input parameter, so an external model cannot be selected from
an input file alone. Use install() right after building the model and
before calculating it::

    from geophires_x.Model import Model
    from geophires_client.adapter import install

    model = Model(input_file='my_superhot_case.txt')
    install(model)
    model.read_parameters()
    model.Calculate()

Keep a built-in reservoir model number in the input file (for example
'Reservoir Model, 1') as a placeholder; install() replaces the object
before any calculation happens.

What it sets
------------
Calculate() fills the reservoir output temperature
(Reservoir.Tresoutput) from the superhot solution and, because the
superhot model simulates the production well itself, also overrides
the results GEOPHIRES' own wellbore model would produce:

    WellBores.ProducedTemperature   wellhead temperature [C]
    WellBores.prodwellflowrate      produced mass flow rate [kg/s]
    WellBores.Pprodwellhead         wellhead pressure [kPa]
    WellBores.ProdTempDrop          feedzone-to-wellhead drop [C]
    WellBores.PumpingPower          zero, the well flows unaided
    WellBores.DPReserv              reservoir drawdown [kPa]

The override is applied by wrapping WellBores.Calculate, so it runs
after GEOPHIRES' own wellbore calculation and before the surface
plant. Electricity generation and economics stay entirely with
GEOPHIRES.

Author: superhot-wellbore GEOPHIRES client
"""

import numpy as np

from . import units
from .client import SuperhotWellboreClient
from .config import (DeclineConfig, OperatingConfig, ReservoirConfig,
                     RockTemperatureConfig, SolverConfig, SuperhotRequest,
                     TimeConfig, WellConfig)
from .results import GEOPHIRES_TMAX_LIMITS_C, PROFILE_TEMPERATURES

# ====================================================================
# OPTIONAL GEOPHIRES IMPORT
# ====================================================================

try:
    from geophires_x.Parameter import (floatParameter, intParameter,
                                       strParameter)
    from geophires_x.Reservoir import Reservoir as _ReservoirBase
    from geophires_x.Units import (LengthUnit, PressureUnit, Units,
                                   TemperatureUnit)

    GEOPHIRES_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - GEOPHIRES not installed
    GEOPHIRES_AVAILABLE = False
    _IMPORT_ERROR = exc
    _ReservoirBase = object


def _require_geophires():
    """Raise a helpful ImportError when GEOPHIRES-X is missing."""
    if not GEOPHIRES_AVAILABLE:
        raise ImportError(
            'GEOPHIRES-X is not importable, so the in-process adapter '
            'cannot be used. Install it with "pip install '
            'geophires-x", or use the file-based integration instead '
            '(python -m geophires_client run ...), which needs no '
            'GEOPHIRES import.') from _IMPORT_ERROR


# ====================================================================
# WELLBORE OVERRIDE
# ====================================================================

def apply_profile_to_wellbores(model, profile):
    """
    Write a superhot production history into GEOPHIRES' WellBores.

    Call this after WellBores.Calculate() so that the superhot values
    survive; install_wellbores_override() arranges exactly that.

    Parameters
    ----------
    model : geophires_x.Model.Model
        The GEOPHIRES model being calculated.
    profile : ProductionProfile
        Result of SuperhotWellboreClient.solve_profile(), solved on
        the GEOPHIRES time vector.
    """
    wellbores = model.wellbores
    n = len(profile.timesteps)

    production_temperature = np.array(profile.production_temperature_C,
                                      dtype=float)
    wellbores.ProducedTemperature.value = production_temperature

    first = profile.initial
    if first is not None:
        wellbores.prodwellflowrate.value = float(first.mass_flow_kgs)
        wellhead_pressure_kPa = units.mpa_to_kpa(float(first.whp_MPa))
        wellbores.Pprodwellhead.value = wellhead_pressure_kPa
        wellbores.ppwellhead.value = wellhead_pressure_kPa

    drop = profile.mean_wellbore_temperature_drop_C
    if np.isfinite(drop):
        wellbores.ProdTempDrop.value = float(drop)

    # A superhot well flows unaided: there is no production pump, and
    # the injection side is outside the scope of this model.
    wellbores.rameyoptionprod.value = False
    wellbores.productionwellpumping.value = False
    wellbores.PumpingPower.value = np.zeros(n)
    wellbores.PumpingPowerProd.value = np.zeros(n)
    wellbores.PumpingPowerInj.value = np.zeros(n)
    wellbores.DPProdWell.value = np.zeros(n)
    wellbores.DPInjWell.value = np.zeros(n)
    wellbores.DPBouyancy.value = np.zeros(n)

    drawdown_kPa = np.array(
        [units.mpa_to_kpa(value) for value in profile.dP_reservoir_MPa],
        dtype=float)
    wellbores.DPReserv.value = drawdown_kPa
    wellbores.DPOverall.value = drawdown_kPa

    model.logger.info(
        f'Superhot wellbore results applied: '
        f'flow rate {wellbores.prodwellflowrate.value:.2f} kg/s, '
        f'wellhead temperature '
        f'{production_temperature[0]:.1f} to '
        f'{production_temperature[-1]:.1f} C')


def install_wellbores_override(model, profile):
    """
    Make GEOPHIRES' WellBores report the superhot production state.

    WellBores.Calculate is wrapped rather than replaced, so
    GEOPHIRES' own wellbore results are still computed (and remain
    available for comparison) before being overridden.

    Parameters
    ----------
    model : geophires_x.Model.Model
    profile : ProductionProfile

    Returns
    -------
    geophires_x.WellBores.WellBores
        The patched wellbores object.
    """
    wellbores = model.wellbores
    wellbores.superhot_profile = profile

    if getattr(wellbores, 'superhot_override_installed', False):
        return wellbores

    original_calculate = wellbores.Calculate

    def calculate_with_superhot_results(model_argument):
        original_calculate(model_argument)
        apply_profile_to_wellbores(model_argument,
                                   wellbores.superhot_profile)

    wellbores.Calculate = calculate_with_superhot_results
    wellbores.superhot_override_installed = True
    return wellbores


# ====================================================================
# RESERVOIR MODEL
# ====================================================================

class SuperhotWellboreReservoir(_ReservoirBase):
    """
    GEOPHIRES reservoir model backed by the superhot-wellbore code.

    Parameters
    ----------
    model : geophires_x.Model.Model
        The GEOPHIRES model this reservoir belongs to.

    Attributes
    ----------
    superhot_request : SuperhotRequest
        Scenario assembled from the GEOPHIRES input parameters,
        available after read_parameters().
    superhot_profile : ProductionProfile
        Production history, available after Calculate().
    """

    def __init__(self, model):
        _require_geophires()
        super().__init__(model)

        self.superhot_request = None
        self.superhot_profile = None
        self.superhot_client = None

        sprefix = 'Superhot '

        self.superhot_reservoir_pressure = self._add_float(
            sprefix + 'Reservoir Pressure',
            DefaultValue=units.mpa_to_kpa(30.0),
            Min=1e2, Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ToolTipText='Far-field pressure of the superhot reservoir '
                        'at the start of production.')
        self.superhot_reservoir_temperature = self._add_float(
            sprefix + 'Reservoir Temperature',
            DefaultValue=450.0, Min=100.0, Max=600.0,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            ToolTipText='Far-field temperature of the superhot '
                        'reservoir at the start of production.')
        self.superhot_transmissivity = self._add_float(
            sprefix + 'Transmissivity',
            DefaultValue=1000.0, Min=1.0, Max=1e6,
            UnitType=Units.NONE,
            ToolTipText='Permeability-thickness product k*b of the '
                        'feedzone in md.m.')
        self.superhot_drainage_radius = self._add_float(
            sprefix + 'Drainage Radius',
            DefaultValue=500.0, Min=1.0, Max=1e5,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
            ToolTipText='Distance to the constant-pressure outer '
                        'boundary of the radial Darcy model.')

        self.superhot_well_depth = self._add_float(
            sprefix + 'Well Depth',
            DefaultValue=-1.0, Min=-1.0, Max=15000.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
            ToolTipText='Feedzone depth of the superhot well. Leave '
                        'at -1 to derive it from the reservoir '
                        'pressure.')
        self.superhot_well_diameter = self._add_float(
            sprefix + 'Well Diameter',
            DefaultValue=0.217, Min=0.01, Max=1.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
            ToolTipText='Casing inner diameter of the superhot well. '
                        'Note that this is in metres, unlike the '
                        'GEOPHIRES well diameters.')
        self.superhot_step_length = self._add_float(
            sprefix + 'Wellbore Step Length',
            DefaultValue=10.0, Min=1.0, Max=100.0,
            UnitType=Units.LENGTH,
            PreferredUnits=LengthUnit.METERS,
            CurrentUnits=LengthUnit.METERS,
            ToolTipText='Vertical integration step of the wellbore '
                        'model. Must be a whole number of metres.')
        self.superhot_roughness = self._add_float(
            sprefix + 'Casing Roughness',
            DefaultValue=4.6e-5, Min=0.0, Max=1e-2,
            UnitType=Units.NONE,
            ToolTipText='Absolute casing roughness in metres.')
        self.superhot_heat_loss = self._add_float(
            sprefix + 'Heat Loss Coefficient',
            DefaultValue=2.5, Min=0.0, Max=100.0,
            UnitType=Units.NONE,
            ToolTipText='Overall wellbore heat transfer coefficient U '
                        'in W/m/K.')
        self.superhot_rock_temperature_mode = self._add_str(
            sprefix + 'Rock Temperature Mode',
            DefaultValue='linear',
            ToolTipText="Formation temperature profile for wellbore "
                        "heat loss: 'linear' or 'boiling'.")

        self.superhot_flow_control = self._add_str(
            sprefix + 'Flow Control',
            DefaultValue='whp',
            ToolTipText="'whp' solves for the flow rate that delivers "
                        "the target wellhead pressure; 'flow' "
                        "prescribes the flow rate.")
        self.superhot_target_whp = self._add_float(
            sprefix + 'Target Wellhead Pressure',
            DefaultValue=units.mpa_to_kpa(10.0),
            Min=1e2, Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ToolTipText='Wellhead pressure the well is operated at.')
        self.superhot_flow_rate = self._add_float(
            sprefix + 'Production Flow Rate',
            DefaultValue=-1.0, Min=-1.0, Max=1000.0,
            UnitType=Units.NONE,
            ToolTipText='Prescribed mass flow rate in kg/s, used when '
                        "the flow control is 'flow'.")

        self.superhot_temperature_decline_mode = self._add_str(
            sprefix + 'Temperature Decline Mode',
            DefaultValue='none',
            ToolTipText="Reservoir temperature decline law: 'none', "
                        "'linear_percent', 'linear_absolute' or "
                        "'exponential'.")
        self.superhot_temperature_decline_rate = self._add_float(
            sprefix + 'Temperature Decline Rate',
            DefaultValue=0.0, Min=0.0, Max=100.0,
            UnitType=Units.NONE,
            ToolTipText='Temperature decline rate, in percent of the '
                        'initial temperature per year, C per year or '
                        '1/year depending on the decline mode.')
        self.superhot_minimum_temperature = self._add_float(
            sprefix + 'Minimum Reservoir Temperature',
            DefaultValue=200.0, Min=1.0, Max=600.0,
            UnitType=Units.TEMPERATURE,
            PreferredUnits=TemperatureUnit.CELSIUS,
            CurrentUnits=TemperatureUnit.CELSIUS,
            ToolTipText='Floor applied to the declining reservoir '
                        'temperature.')
        self.superhot_pressure_decline_mode = self._add_str(
            sprefix + 'Pressure Decline Mode',
            DefaultValue='none',
            ToolTipText='Reservoir pressure decline law, same options '
                        'as the temperature decline mode.')
        self.superhot_pressure_decline_rate = self._add_float(
            sprefix + 'Pressure Decline Rate',
            DefaultValue=0.0, Min=0.0, Max=100.0,
            UnitType=Units.NONE,
            ToolTipText='Pressure decline rate, in percent of the '
                        'initial pressure per year, MPa per year or '
                        '1/year depending on the decline mode.')
        self.superhot_minimum_pressure = self._add_float(
            sprefix + 'Minimum Reservoir Pressure',
            DefaultValue=units.mpa_to_kpa(5.0),
            Min=1e2, Max=1e5,
            UnitType=Units.PRESSURE,
            PreferredUnits=PressureUnit.KPASCAL,
            CurrentUnits=PressureUnit.KPASCAL,
            ToolTipText='Floor applied to the declining reservoir '
                        'pressure.')

        self.superhot_max_solve_points = self._add_int(
            sprefix + 'Max Solve Points',
            DefaultValue=8,
            AllowableRange=list(range(0, 401)),
            ToolTipText='Maximum number of coupled-model solves. '
                        'Timesteps in between are interpolated. Use 0 '
                        'to solve at every GEOPHIRES timestep.')
        self.superhot_reservoir_output_temperature = self._add_str(
            sprefix + 'Reservoir Output Temperature',
            DefaultValue='feedzone',
            ToolTipText="Which temperature is reported as the "
                        "reservoir output temperature: 'feedzone' "
                        "(after drawdown and expansion) or "
                        "'wellhead'.")

        model.logger.info(f'Initialised {self.__class__.__name__}')

    # ----------------------------------------------------------------
    # Parameter declaration helpers
    # ----------------------------------------------------------------

    def _add_float(self, name, **kwargs):
        """Declare a float parameter on this reservoir model."""
        parameter = floatParameter(name, **kwargs)
        self.ParameterDict[parameter.Name] = parameter
        return parameter

    def _add_int(self, name, **kwargs):
        """Declare an int parameter on this reservoir model."""
        parameter = intParameter(name, **kwargs)
        self.ParameterDict[parameter.Name] = parameter
        return parameter

    def _add_str(self, name, **kwargs):
        """Declare a string parameter on this reservoir model."""
        parameter = strParameter(name, **kwargs)
        self.ParameterDict[parameter.Name] = parameter
        return parameter

    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------

    def read_parameters(self, model):
        """
        Read the GEOPHIRES input parameters and build the request.

        Besides the superhot parameters, the reservoir depth,
        geothermal gradient and maximum temperature of the GEOPHIRES
        model are aligned with the superhot well so that the rest of
        GEOPHIRES (drilling cost in particular) sees a consistent
        geometry.
        """
        super().read_parameters(model)

        self.superhot_request = self._build_request(model)
        self.superhot_client = SuperhotWellboreClient(
            self.superhot_request)
        self._align_geophires_geometry(model)

    def _build_request(self, model):
        """Assemble a SuperhotRequest from the GEOPHIRES parameters."""
        depth_m = self.superhot_well_depth.value
        if depth_m is not None and depth_m <= 0:
            depth_m = None

        flow_rate = self.superhot_flow_rate.value
        if flow_rate is not None and flow_rate <= 0:
            flow_rate = None

        request = SuperhotRequest(
            name=str(getattr(model, 'description', None)
                     or 'superhot').strip() or 'superhot',
            reservoir=ReservoirConfig(
                P_reservoir_MPa=units.kpa_to_mpa(
                    self.superhot_reservoir_pressure.value),
                T_reservoir_C=self.superhot_reservoir_temperature.value,
                transmissivity_md_m=self.superhot_transmissivity.value,
                drainage_radius_m=self.superhot_drainage_radius.value),
            well=WellConfig(
                depth_m=depth_m,
                diameter_m=self.superhot_well_diameter.value,
                delta_z_m=self.superhot_step_length.value,
                roughness_m=self.superhot_roughness.value,
                heat_loss_factor=self.superhot_heat_loss.value),
            rock_temperature=RockTemperatureConfig(
                mode=self.superhot_rock_temperature_mode.value,
                T_surface_C=self.Tsurf.value),
            operating=OperatingConfig(
                control=self.superhot_flow_control.value,
                target_whp_MPa=units.kpa_to_mpa(
                    self.superhot_target_whp.value),
                mass_flow_kgs=flow_rate),
            decline=DeclineConfig(
                temperature_mode=(
                    self.superhot_temperature_decline_mode.value),
                temperature_rate_per_year=(
                    self.superhot_temperature_decline_rate.value),
                min_temperature_C=(
                    self.superhot_minimum_temperature.value),
                pressure_mode=self.superhot_pressure_decline_mode.value,
                pressure_rate_per_year=(
                    self.superhot_pressure_decline_rate.value),
                min_pressure_MPa=units.kpa_to_mpa(
                    self.superhot_minimum_pressure.value)),
            time=self._time_config(model),
            solver=SolverConfig(
                max_solve_points=self.superhot_max_solve_points.value))

        output_temperature = (
            self.superhot_reservoir_output_temperature.value)
        if output_temperature not in PROFILE_TEMPERATURES:
            raise ValueError(
                f'{self.superhot_reservoir_output_temperature.Name} '
                f'must be one of '
                f'{", ".join(PROFILE_TEMPERATURES)}, got '
                f'{output_temperature!r}')

        return request.validate()

    def _time_config(self, model):
        """
        Time discretisation taken from the GEOPHIRES model.

        The values are only used when the client builds its own time
        vector; Calculate() always hands GEOPHIRES' own time vector to
        the client. They are refreshed there because the surface plant
        and economics parameters may not have been read yet when this
        reservoir model reads its own.
        """
        config = TimeConfig()
        try:
            config.plant_lifetime_yr = (
                model.surfaceplant.plant_lifetime.value)
            config.timesteps_per_year = (
                model.economics.timestepsperyear.value)
        except AttributeError:
            pass
        return config

    def _align_geophires_geometry(self, model):
        """
        Make the GEOPHIRES geometry agree with the superhot well.

        The reservoir depth and the first geothermal gradient are set
        from the superhot well, and the maximum temperature is raised
        if necessary: GEOPHIRES caps the drilling depth at the depth
        where the maximum temperature is reached, which would
        otherwise truncate a superhot well.
        """
        depth_m = self.superhot_client.depth_m
        T_reservoir_C = self.superhot_request.reservoir.T_reservoir_C

        self.depth.value = units.m_to_km(depth_m)
        gradient = units.gradient_C_per_km(T_reservoir_C,
                                           self.Tsurf.value, depth_m)
        if gradient is not None:
            self.gradient.value[0] = gradient
            self.numseg.value = 1

        required_Tmax = min(T_reservoir_C + 1.0,
                            GEOPHIRES_TMAX_LIMITS_C[1])
        if self.Tmax.value < required_Tmax:
            model.logger.warning(
                f'Raising {self.Tmax.Name} from {self.Tmax.value} to '
                f'{required_Tmax} C so that the superhot reservoir '
                f'temperature of {T_reservoir_C} C is admissible')
            self.Tmax.value = required_Tmax

    # ----------------------------------------------------------------
    # Calculation
    # ----------------------------------------------------------------

    def Calculate(self, model):
        """
        Run the superhot model over the GEOPHIRES time vector.

        The base class calculation runs first so that the time vector,
        rock temperature and water properties are available; the
        superhot solution then replaces the reservoir output
        temperature and the production well state.
        """
        super().Calculate(model)

        if self.superhot_client is None:
            raise RuntimeError('read_parameters() must run before '
                               'Calculate()')

        self.superhot_request.time = self._time_config(model)

        profile = self.superhot_client.solve_profile(
            time_yr=self.timevector.value)
        self.superhot_profile = profile

        for note in profile.notes:
            model.logger.info(f'superhot-wellbore: {note}')

        if profile.initial is None:
            raise RuntimeError(
                'The superhot reservoir-wellbore model produced no '
                'usable solution; see the log for details. Check the '
                'reservoir pressure, temperature, transmissivity and '
                'target wellhead pressure.')

        if (self.superhot_reservoir_output_temperature.value
                == 'wellhead'):
            temperatures = profile.production_temperature_C
        else:
            temperatures = profile.reservoir_output_temperature_C
        self.Tresoutput.value = np.array(temperatures, dtype=float)

        install_wellbores_override(model, profile)


# ====================================================================
# INSTALLATION
# ====================================================================

def install(model, request=None):
    """
    Replace a GEOPHIRES model's reservoir with the superhot model.

    Parameters
    ----------
    model : geophires_x.Model.Model
        A freshly constructed GEOPHIRES model.
    request : SuperhotRequest or dict or None
        Optional scenario that overrides the parameters read from the
        GEOPHIRES input file. Handy for parameter studies driven from
        Python.

    Returns
    -------
    SuperhotWellboreReservoir
        The installed reservoir model.
    """
    _require_geophires()

    reservoir = SuperhotWellboreReservoir(model)
    model.reserv = reservoir
    reservoir.read_parameters(model)

    if request is not None:
        reservoir.superhot_request = SuperhotRequest.from_dict(
            request.to_dict() if hasattr(request, 'to_dict')
            else request).validate()
        reservoir.superhot_client = SuperhotWellboreClient(
            reservoir.superhot_request)
        reservoir._align_geophires_geometry(model)

    return reservoir
