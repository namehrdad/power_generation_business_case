# params.py - Business Case Parameters

# Project Configuration
PLANT_SIZE_MW = 100  # Plant size in MW
NUMBER_OF_GENERATORS = 10
GENERATOR_MODEL = "20V31SG"
ANALYSIS_PERIOD_YEARS = 25

# Unit conversions (we'll use these with pint later)
MILLION = 1_000_000

# Cost Categories (in millions of dollars)
EQUIPMENT_COST = 150.0        # Total equipment cost in million $
INSTALLATION_FACTOR = 3.0     # Installation cost = equipment cost * this factor
LAND_COST = 10.0             # Land cost in million $

# Operational Parameters
CAPACITY_FACTOR = 0.85  # How often the plant runs (85%)
HOURS_PER_YEAR = 8760
EFFICIENCY = 0.45 # Efficiency of the generators (45%)

# Operating Costs (annual, in base year dollars)
OM_COST_ANNUAL = 5.0           # O&M cost in million $ per year

# Natural Gas Costs
NATURAL_GAS_PRICE = 4.26       # $ per GJ
# Natural gas consumption will be calculated based on plant operation

# Fuel Oil (Backup) Costs
FUEL_OIL_BACKUP_PERIOD = 48    # Hours per year running on fuel oil
FUEL_OIL_PRICE = 1.35          # $ per Liter
FUEL_OIL_ENERGY_CONTENT = 0.04 # GJ per Liter

# Major Maintenance Schedule
MAJOR_MAINTENANCE_YEARS = [3, 6, 9, 12, 15, 18]  # Years when major maintenance occurs
MAJOR_MAINTENANCE_COST_PERCENT = 0.10  # 10% of equipment cost

# Grid Electricity (baseline comparison)
GRID_ELECTRICITY_RATE = 0.12   # $ per kWh from grid
GRID_DEMAND_CHARGE = 10.0      # $ per kW per month

# Financial Parameters
INFLATION_RATE = 0.03    # 3% annual inflation affecting all future costs
DISCOUNT_RATE = 0.05     # 8% discount rate for NPV calculations
