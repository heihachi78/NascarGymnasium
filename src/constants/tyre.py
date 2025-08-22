# Tyre System Constants

# Tyre Temperature Constants
TYRE_START_TEMPERATURE = 80.0  # °C
TYRE_IDEAL_TEMPERATURE_MIN = 85.0  # °C
TYRE_IDEAL_TEMPERATURE_MAX = 105.0  # °C
TYRE_OPTIMAL_TEMPERATURE = (TYRE_IDEAL_TEMPERATURE_MIN + TYRE_IDEAL_TEMPERATURE_MAX) / 2.0

# Tyre Grip Constants
TYRE_MAX_GRIP_COEFFICIENT = 1.4  # 1.4 maximum grip coefficient at ideal temperature (optimal baseline)
                                    # NOTE: This is the single source of truth for tire grip in the simulation
TYRE_MIN_GRIP_COEFFICIENT = 0.8  # 0.8 minimum grip coefficient when far from ideal (realistic degraded performance)
TYRE_GRIP_FALLOFF_RATE = 0.02  # grip reduction per degree outside ideal range

# Tyre Wear Constants
TYRE_MAX_WEAR = 100.0  # maximum wear percentage (100% = completely worn)
TYRE_FRICTION_WEAR_COEFFICIENT = 0.00001  # wear rate per Newton of friction force per second
TYRE_GRIP_WEAR_FACTOR = 0.5  # grip reduction factor when tyre is worn

# Speed-Dependent Wear Constants
TYRE_SPEED_WEAR_REFERENCE_SPEED = 400.0  # km/h - reference speed for calculation  
TYRE_SPEED_WEAR_MULTIPLIER = 3.0  # multiplier factor for speed-based wear
TYRE_SPEED_WEAR_MAX_MULTIPLIER = 3.0  # maximum speed wear multiplier cap

# Cornering-Intensity Wear Constants
TYRE_CORNERING_WEAR_BASE_G = 2.0  # m/s² - lateral G above which cornering wear increases  
TYRE_CORNERING_WEAR_MULTIPLIER = 1.5  # multiplier for cornering-based wear
TYRE_CORNERING_WEAR_MAX_MULTIPLIER = 2.5  # maximum cornering wear multiplier

# Slip-Angle Wear Constants (proportional scaling)
TYRE_SLIP_WEAR_MAX_ANGLE = 45.0  # degrees - slip angle for maximum wear effect
TYRE_SLIP_WEAR_MAX_MULTIPLIER = 3.0  # maximum wear multiplier at max slip angle

# Tyre Thermal Constants
TYRE_HEATING_RATE_FRICTION = 0.040  # temperature increase per unit of friction work (moderate increase)
TYRE_COOLING_RATE_AMBIENT = 0.01  # temperature decrease rate in ambient air at low speeds
TYRE_HIGH_SPEED_COOLING_REDUCTION = 0.5  # cooling reduction factor at high speeds (aerodynamic heating effect)
TYRE_MIN_COOLING_SPEED_THRESHOLD = 50.0  # m/s - speed above which cooling is reduced
TYRE_AERODYNAMIC_HEATING_FACTOR = 0.0002  # additional heating from aerodynamic friction (reduced)
AMBIENT_TEMPERATURE = 25.0  # °C ambient air temperature
TYRE_THERMAL_MASS = 125.0  # thermal mass factor for temperature changes

# Tyre Pressure Constants
TYRE_OPTIMAL_PRESSURE_PSI = 32.0  # PSI (pounds per square inch)
TYRE_MIN_PRESSURE_PSI = 20.0  # PSI minimum safe pressure
TYRE_MAX_PRESSURE_PSI = 50.0  # PSI maximum safe pressure
TYRE_PRESSURE_TEMPERATURE_FACTOR = 0.015  # Pressure change per degree Celsius (reduced)
TYRE_PRESSURE_LOAD_FACTOR = 0.00005  # Pressure change per Newton of load (halved)
MAX_TYRE_PRESSURE_INCREASE = 8.0  # Maximum pressure increase from base (PSI)

# Cornering Heating Constants
CORNERING_OUTER_TYRE_HEATING_FACTOR = 1.5  # Outer tyres heat more during cornering
CORNERING_INNER_TYRE_HEATING_FACTOR = 0.5  # Inner tyres heat less during cornering

# Slip Angle Detection and Heating
SLIP_ANGLE_THRESHOLD_DEGREES = 5.0  # Degrees - minimum slip angle to consider for heating
SLIP_ANGLE_HEATING_BASE_MULTIPLIER = 2.2  # Base heating multiplier for slip angle
SLIP_ANGLE_HEATING_EXPONENTIAL_FACTOR = 0.04  # Exponential factor for slip angle heating (per degree)
SLIP_ANGLE_MAX_HEATING_MULTIPLIER = 8.0  # Maximum heating multiplier from slip angle
SLIP_ANGLE_SPEED_THRESHOLD = 2.0  # m/s - minimum speed for slip angle heating

# Lateral Force Heating Constants
LATERAL_FORCE_HEATING_FACTOR = 0.05  # Heating factor per Newton of lateral force (reduced for realistic temps)
LATERAL_FORCE_DISTRIBUTION_FRONT = 0.5  # 50% of lateral heating goes to front tyres (balanced)
LATERAL_FORCE_DISTRIBUTION_REAR = 0.5  # 50% of lateral heating goes to rear tyres (balanced)
MAX_LATERAL_HEATING_PER_TYRE = 25.0  # Maximum additional heating per tyre from lateral forces (°C/s) - increased

# Cornering Load Transfer Heating
CORNERING_LOAD_HEATING_FACTOR = 0.001  # Additional heating factor for loaded outer tyres during cornering (increased 25x)
CORNERING_SPEED_THRESHOLD = 3.0  # m/s - minimum speed for cornering load heating effects