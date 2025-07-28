# main.py - Core Business Case Calculations

import pint
import params
import numpy as np
import pandas as pd

# Set up units (no USD - we'll use regular numbers for currency)
ureg = pint.UnitRegistry()
MW = ureg.megawatt
MWh = ureg.megawatt_hour
kW = ureg.kilowatt
kWh = ureg.kilowatt_hour
GJ = ureg.gigajoule
liter = ureg.liter
hour = ureg.hour

class BusinessCase:
    def __init__(self):
        """Initialize the business case with parameters from params.py"""
        self.plant_size = params.PLANT_SIZE_MW * MW
        self.equipment_cost = params.EQUIPMENT_COST  # Million USD
        self.analysis_period = params.ANALYSIS_PERIOD_YEARS
        self.inflation_rate = params.INFLATION_RATE
        self.discount_rate = params.DISCOUNT_RATE
        
        # CAPEX components (all in million USD)
        self.installation_cost = self.equipment_cost * params.INSTALLATION_FACTOR
        self.land_cost = params.LAND_COST
        
        # Operating costs (annual, base year, million USD)
        self.om_cost_annual = params.OM_COST_ANNUAL
        
        # Fuel costs
        self.ng_price = params.NATURAL_GAS_PRICE  # USD per GJ
        self.fuel_oil_price = params.FUEL_OIL_PRICE  # USD per liter
        self.fuel_oil_energy = params.FUEL_OIL_ENERGY_CONTENT  # GJ per liter
        
        # Grid electricity rate for comparison
        self.grid_rate = params.GRID_ELECTRICITY_RATE  # USD per kWh
        self.grid_demand_charge = params.GRID_DEMAND_CHARGE  # USD per kW per month
        
    def calculate_annual_generation(self):
        """Calculate how much electricity the plant generates per year"""
        annual_generation = (self.plant_size * 
                           params.CAPACITY_FACTOR * 
                           params.HOURS_PER_YEAR * hour)
        return annual_generation.to(MWh)
    
    def calculate_total_capex(self):
        """Calculate total CAPEX: equipment + installation + land (Million USD)"""
        total_capex = self.equipment_cost + self.installation_cost + self.land_cost
        return total_capex
    
    def calculate_capital_cost_per_mw(self):
        """Calculate CAPEX per MW of capacity (Million USD per MW)"""
        total_capex = self.calculate_total_capex()
        plant_size_mw = self.plant_size.to(MW).magnitude
        return total_capex / plant_size_mw
    
    def calculate_annual_ng_cost(self):
        """Calculate annual natural gas cost based on plant operation (Million USD)"""
        # Convert annual generation to GJ
        annual_gen_gj = self.calculate_annual_generation().to(GJ, 'energy').magnitude
        # Use efficiency from params
        annual_ng_consumption_gj = annual_gen_gj / params.EFFICIENCY
        annual_ng_cost_usd = annual_ng_consumption_gj * self.ng_price
        return annual_ng_cost_usd / 1_000_000  # Convert to million USD
    
    def calculate_annual_fuel_oil_cost(self):
        """Calculate annual fuel oil cost for backup operation (Million USD)"""
        # Plant runs on fuel oil for specified backup hours
        backup_hours = params.FUEL_OIL_BACKUP_PERIOD
        backup_generation_mwh = self.plant_size.to(MW).magnitude * backup_hours
        backup_gen_gj = backup_generation_mwh * 3.6  # 1 MWh = 3.6 GJ
        
        # Convert to fuel consumption (using same efficiency as NG)
        fuel_oil_gj_needed = backup_gen_gj / params.EFFICIENCY
        
        # Convert GJ to liters
        fuel_oil_liters = fuel_oil_gj_needed / self.fuel_oil_energy
        
        annual_fuel_oil_cost_usd = fuel_oil_liters * self.fuel_oil_price
        return annual_fuel_oil_cost_usd / 1_000_000  # Convert to million USD
    
    def calculate_annual_grid_cost(self):
        """Calculate what it would cost to buy same electricity from grid (Million USD)"""
        annual_gen = self.calculate_annual_generation()
        # Convert MWh to kWh for rate calculation
        annual_gen_kwh = annual_gen.to(kWh).magnitude
        energy_cost_usd = annual_gen_kwh * self.grid_rate
        
        # Add demand charges (based on plant capacity)
        plant_size_kw = self.plant_size.to(kW).magnitude
        monthly_demand_charge_usd = plant_size_kw * self.grid_demand_charge
        annual_demand_charge_usd = monthly_demand_charge_usd * 12
        
        total_grid_cost_usd = energy_cost_usd + annual_demand_charge_usd
        return total_grid_cost_usd / 1_000_000  # Convert to million USD
    
    def calculate_major_maintenance_cost(self, year_number):
        """Calculate major maintenance cost for a given year (Million USD)"""
        if year_number in params.MAJOR_MAINTENANCE_YEARS:
            return self.equipment_cost * params.MAJOR_MAINTENANCE_COST_PERCENT
        else:
            return 0.0
    
    def calculate_annual_operating_cost(self, year_number):
        """Calculate total annual operating cost for a given year (Million USD)"""
        inflation_factor = (1 + self.inflation_rate) ** year_number
        
        # Regular operating costs (O&M, NG, Fuel Oil)
        regular_costs = (self.om_cost_annual + 
                        self.calculate_annual_ng_cost() + 
                        self.calculate_annual_fuel_oil_cost()) * inflation_factor
        
        # Major maintenance (if applicable this year)
        maintenance_cost = self.calculate_major_maintenance_cost(year_number) * inflation_factor
        
        return regular_costs + maintenance_cost
    
    def calculate_annual_savings(self, year_number):
        """Calculate annual savings vs grid electricity for a given year (Million USD)"""
        # Grid cost also inflates
        inflation_factor = (1 + self.inflation_rate) ** year_number
        grid_cost = self.calculate_annual_grid_cost() * inflation_factor
        operating_cost = self.calculate_annual_operating_cost(year_number)
        
        return grid_cost - operating_cost
    
    def calculate_npv_grid_option(self):
        """Calculate NPV for grid electricity option (Million USD)"""
        npv_total = 0.0
        for year_num in range(1, self.analysis_period + 1):
            inflation_factor = (1 + self.inflation_rate) ** year_num
            annual_grid_cost = self.calculate_annual_grid_cost() * inflation_factor
            discount_factor = (1 + self.discount_rate) ** year_num
            npv_total += annual_grid_cost / discount_factor
        return npv_total
    
    def calculate_npv_proposed_option(self):
        """Calculate NPV for proposed equipment option (Million USD)"""
        # Initial CAPEX investment
        initial_capex = self.calculate_total_capex()
        
        # NPV of operating costs over analysis period
        npv_operating = 0.0
        for year_num in range(1, self.analysis_period + 1):
            annual_operating_cost = self.calculate_annual_operating_cost(year_num)
            discount_factor = (1 + self.discount_rate) ** year_num
            npv_operating += annual_operating_cost / discount_factor
        
        total_npv = initial_capex + npv_operating
        return total_npv
    
    def calculate_npv_savings(self):
        """Calculate NPV savings (Grid NPV - Proposed NPV) (Million USD)"""
        grid_npv = self.calculate_npv_grid_option()
        proposed_npv = self.calculate_npv_proposed_option()
        return grid_npv - proposed_npv
    
    def get_summary(self):
        """Return a summary of key metrics"""
        annual_gen = self.calculate_annual_generation()
        total_capex = self.calculate_total_capex()
        cost_per_mw = self.calculate_capital_cost_per_mw()
        annual_grid_cost = self.calculate_annual_grid_cost()
        annual_ng_cost = self.calculate_annual_ng_cost()
        annual_fuel_oil_cost = self.calculate_annual_fuel_oil_cost()
        annual_operating_cost = self.calculate_annual_operating_cost(1)  # Year 1
        annual_savings = self.calculate_annual_savings(1)  # Year 1
        
        # NPV calculations for both options
        npv_grid = self.calculate_npv_grid_option()
        npv_proposed = self.calculate_npv_proposed_option()
        npv_savings = self.calculate_npv_savings()
        
        return {
            'plant_size_mw': self.plant_size.to(MW).magnitude,
            'annual_generation_mwh': annual_gen.magnitude,
            'total_capex_musd': total_capex,
            'capex_per_mw_musd': cost_per_mw,
            'annual_grid_cost_musd': annual_grid_cost,
            'annual_ng_cost_musd': annual_ng_cost,
            'annual_fuel_oil_cost_musd': annual_fuel_oil_cost,
            'annual_operating_cost_musd': annual_operating_cost,
            'annual_savings_year1_musd': annual_savings,
            'npv_grid_option_musd': npv_grid,
            'npv_proposed_option_musd': npv_proposed,
            'npv_savings_musd': npv_savings,
            'analysis_period_years': self.analysis_period
        }
    
    def get_annual_cashflows(self):
        """Return year-by-year cashflows for both options (useful for plotting)"""
        years = list(range(0, self.analysis_period + 1))
        grid_costs = []
        proposed_costs = []
        
        # Year 0 - Initial investment
        grid_costs.append(0)  # No upfront cost for grid
        proposed_costs.append(self.calculate_total_capex())
        
        # Years 1 to analysis period
        for year_num in range(1, self.analysis_period + 1):
            inflation_factor = (1 + self.inflation_rate) ** year_num
            
            # Grid option annual cost
            grid_annual = self.calculate_annual_grid_cost() * inflation_factor
            grid_costs.append(grid_annual)
            
            # Proposed option annual cost  
            proposed_annual = self.calculate_annual_operating_cost(year_num)
            proposed_costs.append(proposed_annual)
        
        return {
            'years': years,
            'grid_annual_costs': grid_costs,
            'proposed_annual_costs': proposed_costs
        }

class MonteCarloAnalysis:
    def __init__(self, base_params=None):
        """Initialize Monte Carlo analysis with base parameters"""
        self.base_params = base_params or {}
        
    def run_simulation(self, n_simulations=10000, mc_params=None):
        """
        Run Monte Carlo simulation with uncertain parameters
        
        mc_params should contain:
        - inflation_rate_mean, inflation_rate_std
        - grid_rate_mean, grid_rate_std  
        - ng_rate_mean, ng_rate_std
        - maintenance_cost_mean, maintenance_cost_std (as % of equipment cost)
        - om_cost_mean, om_cost_std (Million USD)
        """
        if mc_params is None:
            mc_params = self.get_default_mc_params()
            
        results = []
        
        # Store original params
        original_params = {
            'INFLATION_RATE': params.INFLATION_RATE,
            'GRID_ELECTRICITY_RATE': params.GRID_ELECTRICITY_RATE,
            'NATURAL_GAS_PRICE': params.NATURAL_GAS_PRICE,
            'MAJOR_MAINTENANCE_COST_PERCENT': params.MAJOR_MAINTENANCE_COST_PERCENT,
            'OM_COST_ANNUAL': params.OM_COST_ANNUAL
        }
        
        for i in range(n_simulations):
            # Sample uncertain parameters
            sampled_params = self.sample_parameters(mc_params)
            
            # Update params module temporarily
            params.INFLATION_RATE = sampled_params['inflation_rate']
            params.GRID_ELECTRICITY_RATE = sampled_params['grid_rate']
            params.NATURAL_GAS_PRICE = sampled_params['ng_rate']
            params.MAJOR_MAINTENANCE_COST_PERCENT = sampled_params['maintenance_cost_rate']
            params.OM_COST_ANNUAL = sampled_params['om_cost']
            
            # Apply any base parameter overrides
            for key, value in self.base_params.items():
                setattr(params, key, value)
            
            # Run business case
            bc = BusinessCase()
            summary = bc.get_summary()
            
            # Store results with sampled parameters
            result = {
                'simulation': i + 1,
                'npv_grid': summary['npv_grid_option_musd'],
                'npv_proposed': summary['npv_proposed_option_musd'],
                'npv_savings': summary['npv_savings_musd'],
                'total_capex': summary['total_capex_musd'],
                'annual_operating_cost': summary['annual_operating_cost_musd'],
                'inflation_rate': sampled_params['inflation_rate'],
                'grid_rate': sampled_params['grid_rate'],
                'ng_rate': sampled_params['ng_rate'],
                'maintenance_cost_rate': sampled_params['maintenance_cost_rate'],
                'om_cost': sampled_params['om_cost']
            }
            results.append(result)
        
        # Restore original params
        for key, value in original_params.items():
            setattr(params, key, value)
        
        return pd.DataFrame(results)
    
    def sample_parameters(self, mc_params):
        """Sample one set of uncertain parameters"""
        return {
            'inflation_rate': np.random.normal(
                mc_params['inflation_rate_mean'], 
                mc_params['inflation_rate_std']
            ),
            'grid_rate': np.random.normal(
                mc_params['grid_rate_mean'], 
                mc_params['grid_rate_std']
            ),
            'ng_rate': np.random.normal(
                mc_params['ng_rate_mean'], 
                mc_params['ng_rate_std']
            ),
            'maintenance_cost_rate': np.random.normal(
                mc_params['maintenance_cost_mean'], 
                mc_params['maintenance_cost_std']
            ),
            'om_cost': np.random.normal(
                mc_params['om_cost_mean'], 
                mc_params['om_cost_std']
            )
        }
    
    def get_default_mc_params(self):
        """Default Monte Carlo parameters"""
        return {
            'inflation_rate_mean': 0.02,     # 2% inflation
            'inflation_rate_std': 0.01,      # ±1% std
            'grid_rate_mean': params.GRID_ELECTRICITY_RATE,
            'grid_rate_std': 0.02,           # ±2¢/kWh std
            'ng_rate_mean': params.NATURAL_GAS_PRICE,
            'ng_rate_std': 1.0,              # ±$1/GJ std
            'maintenance_cost_mean': params.MAJOR_MAINTENANCE_COST_PERCENT,
            'maintenance_cost_std': 0.02,    # ±2% std
            'om_cost_mean': params.OM_COST_ANNUAL,
            'om_cost_std': 1.0               # ±$1M std
        }
    
    def analyze_results(self, results_df):
        """Analyze Monte Carlo results and return summary statistics"""
        analysis = {}
        
        # NPV Savings Analysis
        npv_savings = results_df['npv_savings']
        analysis['npv_savings'] = {
            'mean': npv_savings.mean(),
            'std': npv_savings.std(),
            'min': npv_savings.min(),
            'max': npv_savings.max(),
            'p10': npv_savings.quantile(0.10),
            'p25': npv_savings.quantile(0.25),
            'p50': npv_savings.quantile(0.50),
            'p75': npv_savings.quantile(0.75),
            'p90': npv_savings.quantile(0.90),
            'prob_positive': (npv_savings > 0).mean() * 100
        }
        
        # Grid NPV Analysis
        npv_grid = results_df['npv_grid']
        analysis['npv_grid'] = {
            'mean': npv_grid.mean(),
            'std': npv_grid.std(),
            'p10': npv_grid.quantile(0.10),
            'p50': npv_grid.quantile(0.50),
            'p90': npv_grid.quantile(0.90)
        }
        
        # Proposed NPV Analysis
        npv_proposed = results_df['npv_proposed']
        analysis['npv_proposed'] = {
            'mean': npv_proposed.mean(),
            'std': npv_proposed.std(),
            'p10': npv_proposed.quantile(0.10),
            'p50': npv_proposed.quantile(0.50),
            'p90': npv_proposed.quantile(0.90)
        }
        
        # Sensitivity Analysis (correlation with NPV savings)
        correlations = results_df[['npv_savings', 'inflation_rate', 'grid_rate', 
                                 'ng_rate', 'maintenance_cost_rate', 'om_cost']].corr()['npv_savings']
        
        analysis['sensitivity'] = {
            'inflation_rate': correlations['inflation_rate'],
            'grid_rate': correlations['grid_rate'],
            'ng_rate': correlations['ng_rate'],
            'maintenance_cost_rate': correlations['maintenance_cost_rate'],
            'om_cost': correlations['om_cost']
        }
        
        return analysis

# Simple test function
if __name__ == "__main__":
    bc = BusinessCase()
    summary = bc.get_summary()
    
    print("Business Case Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")