# app.py - Streamlit Dashboard for Business Case

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import main
import params

# Page configuration
st.set_page_config(
    page_title="Power Generation Business Case",
    page_icon="⚡",
    layout="wide"
)

# Title and introduction
st.title("⚡ Power Generation Business Case Dashboard")
st.markdown("Compare grid electricity vs. on-site power generation with natural gas and fuel oil backup")

# Sidebar for parameters
st.sidebar.header("📊 Adjust Parameters")

# Key parameters that users might want to adjust
plant_size = st.sidebar.slider("Plant Size (MW)", 50, 200, params.PLANT_SIZE_MW)
equipment_cost = st.sidebar.slider("Equipment Cost (Million $)", 100, 300, int(params.EQUIPMENT_COST))
land_cost = st.sidebar.slider("Land Cost (Million $)", 5, 25, int(params.LAND_COST))
grid_rate = st.sidebar.slider("Grid Electricity Rate ($/kWh)", 0.03, 0.20, params.GRID_ELECTRICITY_RATE, 0.01)
ng_price = st.sidebar.slider("Natural Gas Price ($/GJ)", 2.0, 8.0, params.NATURAL_GAS_PRICE, 0.1)
fuel_oil_rate = st.sidebar.slider("Fuel Oil Price ($/Liter)", 1.0, 2.0, params.FUEL_OIL_PRICE, 0.05)
discount_rate = st.sidebar.slider("Discount Rate (%)", 3, 15, int(params.DISCOUNT_RATE * 100)) / 100
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 1, 10, int(params.INFLATION_RATE * 100)) / 100

# Update params temporarily for this run
params.PLANT_SIZE_MW = plant_size
params.EQUIPMENT_COST = equipment_cost
params.LAND_COST = land_cost
params.GRID_ELECTRICITY_RATE = grid_rate
params.NATURAL_GAS_PRICE = ng_price
params.FUEL_OIL_PRICE = fuel_oil_rate
params.DISCOUNT_RATE = discount_rate
params.INFLATION_RATE = inflation_rate
# Create business case instance
bc = main.BusinessCase()
summary = bc.get_summary()
cashflows = bc.get_annual_cashflows()

# Main dashboard layout
col1, col2, col3 = st.columns(3)

# Key metrics cards
with col1:
    st.metric(
        "Plant Size", 
        f"{summary['plant_size_mw']:.0f} MW",
        f"{summary['annual_generation_mwh']:.0f} MWh/year"
    )

with col2:
    st.metric(
        "Total CAPEX", 
        f"${summary['total_capex_musd']:.1f}M",
        f"${summary['capex_per_mw_musd']:.2f}M/MW"
    )

with col3:
    npv_savings = summary['npv_savings_musd']
    st.metric(
        "NPV Savings", 
        f"${npv_savings:.1f}M",
        "Grid vs Proposed" if npv_savings > 0 else "Proposed vs Grid"
    )

# NPV Comparison Chart
st.subheader(f"📊 NPV Comparison ({params.ANALYSIS_PERIOD_YEARS}-Year Analysis)")
npv_data = {
    'Option': ['Grid Electricity', 'Proposed (NG + Fuel Oil)'],
    'NPV (Million $)': [summary['npv_grid_option_musd'], summary['npv_proposed_option_musd']]
}
npv_df = pd.DataFrame(npv_data)

fig_npv = px.bar(
    npv_df, 
    x='Option', 
    y='NPV (Million $)', 
    title="Net Present Value Comparison",
    color='Option',
    color_discrete_map={
        'Grid Electricity': '#FF6B6B',
        'Proposed (NG + Fuel Oil)': '#4ECDC4'
    }
)
fig_npv.update_layout(showlegend=False, height=400)
st.plotly_chart(fig_npv, use_container_width=True)

# Annual Cashflow Charts
st.subheader("📈 Annual Costs Over Time")

cashflow_df = pd.DataFrame({
        'Year': cashflows['years'],
        'Grid Option': cashflows['grid_annual_costs'],
        'Proposed Option': cashflows['proposed_annual_costs'],
        'Costs Difference': cashflows['difference_annual_costs']
    })

col1, col2 = st.columns(2)

with col1:
    fig_cashflow = go.Figure()
    fig_cashflow.add_trace(go.Scatter(
        x=cashflow_df['Year'].iloc[1:], 
        y=cashflow_df['Grid Option'].iloc[1:],
        mode='lines+markers',
        name='Grid Electricity',
        line=dict(color='#FF6B6B', width=3)
    ))
    fig_cashflow.add_trace(go.Scatter(
        x=cashflow_df['Year'].iloc[1:], 
        y=cashflow_df['Proposed Option'].iloc[1:],
        mode='lines+markers',
        name='Proposed (NG + Fuel Oil)',
        line=dict(color='#4ECDC4', width=3)
    ))

    fig_cashflow.update_layout(
        title="Annual Costs (Excluding Initial CAPEX in Year 0)",
        xaxis_title="Year",
        yaxis_title="Annual Cost (Million $)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_cashflow, use_container_width=True)

with col2:
    
    fig_cashflow = go.Figure()
    fig_cashflow.add_trace(go.Scatter(
        x=cashflow_df['Year'], 
        y=cashflow_df['Costs Difference'],
        mode='lines+markers',
        name='Annual Costs Difference',
        line=dict(color="#DEFFA1", width=4)
    ))
    
    fig_cashflow.update_layout(
        title="Annual Costs Difference (Including Initial CAPEX in Year 0) = ACCUMULATED SAVINGS",
        xaxis_title="Year",
        yaxis_title="Annual Cost (Million $)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_cashflow, use_container_width=True)


# Cost Breakdown Section
st.subheader("🥧 Cost Breakdown Analysis")

col1, col2 = st.columns(2)

with col1:
    # Grid option breakdown
    grid_energy_cost = summary['annual_grid_cost_musd'] * (summary['annual_generation_mwh'] * grid_rate / 1000) / summary['annual_grid_cost_musd']
    grid_demand_cost = summary['annual_grid_cost_musd'] - grid_energy_cost
    
    grid_breakdown = pd.DataFrame({
        'Component': ['Energy Charges', 'Demand Charges'],
        'Cost': [grid_energy_cost, grid_demand_cost]
    })
    
    fig_grid = px.pie(
        grid_breakdown, 
        values='Cost', 
        names='Component',
        title="Grid Option - Annual Cost Breakdown",
        color_discrete_sequence=['#FF9999', '#FFCCCC']
    )
    st.plotly_chart(fig_grid, use_container_width=True)

with col2:
    # Proposed option breakdown (Year 1)
    proposed_breakdown = pd.DataFrame({
        'Component': ['O&M', 'Natural Gas', 'Fuel Oil (Backup)'],
        'Cost': [
            params.OM_COST_ANNUAL,
            summary['annual_ng_cost_musd'],
            summary['annual_fuel_oil_cost_musd']
        ]
    })
    
    fig_proposed = px.pie(
        proposed_breakdown, 
        values='Cost', 
        names='Component',
        title="Proposed Option - Annual Operating Costs",
        color_discrete_sequence=['#66B2FF', '#99CCFF', '#CCE5FF']
    )
    st.plotly_chart(fig_proposed, use_container_width=True)

# CAPEX Breakdown
st.subheader("💰 CAPEX Breakdown")
capex_breakdown = pd.DataFrame({
    'Component': ['Equipment', 'Installation', 'Land'],
    'Cost (Million $)': [
        params.EQUIPMENT_COST,
        params.EQUIPMENT_COST * params.INSTALLATION_FACTOR,
        params.LAND_COST
    ]
})

fig_capex = px.bar(
    capex_breakdown, 
    x='Component', 
    y='Cost (Million $)',
    title="Capital Expenditure Breakdown",
    color='Component',
    color_discrete_sequence=['#FFB347', '#FF8C69', '#FFA07A']
)
st.plotly_chart(fig_capex, use_container_width=True)

# Detailed Results Table
st.subheader("📋 Detailed Financial Results")
results_data = {
    'Metric': [
        'Plant Capacity',
        'Annual Generation',
        'Total CAPEX',
        'CAPEX per MW',
        'Annual Grid Cost (Year 1)',
        'Annual NG Cost (Year 1)',
        'Annual Fuel Oil Cost (Year 1)',
        'Annual Operating Cost (Year 1)',
        'Annual Savings (Year 1)',
        'Grid Option NPV',
        'Proposed Option NPV',
        'NPV Savings'
    ],
    'Value': [
        f"{summary['plant_size_mw']:.0f} MW",
        f"{summary['annual_generation_mwh']:.0f} MWh",
        f"${summary['total_capex_musd']:.1f}M",
        f"${summary['capex_per_mw_musd']:.2f}M/MW",
        f"${summary['annual_grid_cost_musd']:.2f}M",
        f"${summary['annual_ng_cost_musd']:.2f}M",
        f"${summary['annual_fuel_oil_cost_musd']:.3f}M",
        f"${summary['annual_operating_cost_musd']:.2f}M",
        f"${summary['annual_savings_year1_musd']:.2f}M",
        f"${summary['npv_grid_option_musd']:.1f}M",
        f"${summary['npv_proposed_option_musd']:.1f}M",
        f"${summary['npv_savings_musd']:.1f}M"
    ]
}

results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True, hide_index=True)

# Footer with assumptions
st.subheader("📝 Key Assumptions")
col1, col2 = st.columns(2)

with col1:
    st.write(f"• Plant Efficiency: {params.EFFICIENCY*100:.0f}%")
    st.write(f"• Capacity Factor: {params.CAPACITY_FACTOR*100:.0f}%")
    st.write(f"• Analysis Period: {params.ANALYSIS_PERIOD_YEARS} years")
    st.write(f"• Inflation Rate: {params.INFLATION_RATE*100:.1f}%")

with col2:
    st.write(f"• Backup Operation: {params.FUEL_OIL_BACKUP_PERIOD} hours/year")
    st.write(f"• Installation Factor: {params.INSTALLATION_FACTOR}x equipment cost")
    st.write(f"• Major Maintenance: {params.MAJOR_MAINTENANCE_COST_PERCENT*100:.0f}% of equipment cost")
    st.write(f"• Maintenance Years: {params.MAJOR_MAINTENANCE_YEARS}")

# =======================
# MONTE CARLO SIMULATION
# =======================
st.header("🎲 Monte Carlo Sensitivity and Risk Analysis")
st.markdown("Analyze uncertainty and sensitivity by varying multiple parameters simultaneously")

# Monte Carlo parameters in expandable section
with st.expander("⚙️ Monte Carlo Parameters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Distributions")
        
        # Inflation Rate
        inflation_mean = st.slider("Inflation Rate - Mean (%)", 1.0, 4.0, 2.0, 0.1) / 100
        inflation_std = st.slider("Inflation Rate - Std Dev (%)", 0.5, 2.0, 1.0, 0.1) / 100
        
        # Grid Rate  
        grid_rate_mean = st.slider("Grid Rate - Mean ($/kWh)", 0.02, 0.20, grid_rate, 0.01)
        grid_rate_std = st.slider("Grid Rate - Std Dev ($/kWh)", 0.01, 0.05, 0.02, 0.005)
        
        # Natural Gas Rate
        ng_rate_mean = st.slider("NG Rate - Mean ($/GJ)", 2.0, 8.0, ng_price, 0.1)
        ng_rate_std = st.slider("NG Rate - Std Dev ($/GJ)", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        st.subheader("Operating Parameters")
        
        # Maintenance Cost Rate
        maint_mean = st.slider("Maintenance Cost - Mean (%)", 5, 15, 10, 1) / 100
        maint_std = st.slider("Maintenance Cost - Std Dev (%)", 1, 5, 2, 1) / 100
        
        # O&M Cost
        om_mean = st.slider("O&M Cost - Mean (Million $)", 3, 8, 5, 1)
        om_std = st.slider("O&M Cost - Std Dev (Million $)", 0.5, 2.0, 1.0, 0.1)
        
        # Number of simulations
        n_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=2)

# Calculate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        # Prepare MC parameters
        mc_params = {
            'inflation_rate_mean': inflation_mean,
            'inflation_rate_std': inflation_std,
            'grid_rate_mean': grid_rate_mean,
            'grid_rate_std': grid_rate_std,
            'ng_rate_mean': ng_rate_mean,
            'ng_rate_std': ng_rate_std,
            'maintenance_cost_mean': maint_mean,
            'maintenance_cost_std': maint_std,
            'om_cost_mean': om_mean,
            'om_cost_std': om_std
        }
        
        # Base parameters from sidebar
        base_params = {
            'PLANT_SIZE_MW': plant_size,
            'EQUIPMENT_COST': equipment_cost,
            'LAND_COST': land_cost,
            'FUEL_OIL_PRICE': fuel_oil_rate,
            'DISCOUNT_RATE': discount_rate
        }
        
        # Run simulation
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
            mc = main.MonteCarloAnalysis(base_params)
            results_df = mc.run_simulation(n_sims, mc_params)
            analysis = mc.analyze_results(results_df)
        
        # Store results in session state
        st.session_state['mc_results'] = results_df
        st.session_state['mc_analysis'] = analysis
        st.success(f"✅ Completed {n_sims:,} simulations!")

# Display results if available
if 'mc_results' in st.session_state:
    results_df = st.session_state['mc_results']
    analysis = st.session_state['mc_analysis']
    
    # Summary Statistics
    st.subheader("📊 Monte Carlo Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean NPV Savings",
            f"${analysis['npv_savings']['mean']:.1f}M",
            f"±${analysis['npv_savings']['std']:.1f}M"
        )
    
    with col2:
        st.metric(
            "Probability of Success",
            f"{analysis['npv_savings']['prob_positive']:.1f}%",
            "NPV Savings > $0"
        )
    
    with col3:
        st.metric(
            "10th Percentile",
            f"${analysis['npv_savings']['p10']:.1f}M",
            "Worst 10% of cases"
        )
    
    with col4:
        st.metric(
            "90th Percentile", 
            f"${analysis['npv_savings']['p90']:.1f}M",
            "Best 10% of cases"
        )
    
    # NPV Distribution Histogram
    st.subheader("📈 NPV Distribution Analysis")
    
    # Create three columns for the three NPV distributions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # NPV Savings Distribution
        fig_savings = px.histogram(
            results_df, 
            x='npv_savings',
            nbins=30,
            title="NPV Savings Distribution",
            labels={'npv_savings': 'NPV Savings (Million $)', 'count': 'Frequency'},
            color_discrete_sequence=['#2E86AB']
        )
        
        # Add vertical lines for percentiles
        fig_savings.add_vline(x=analysis['npv_savings']['p10'], line_dash="dash", line_color="red", 
                             annotation_text="10%")
        fig_savings.add_vline(x=analysis['npv_savings']['p50'], line_dash="dash", line_color="green",
                             annotation_text="50%")
        fig_savings.add_vline(x=analysis['npv_savings']['p90'], line_dash="dash", line_color="red",
                             annotation_text="90%")
        fig_savings.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="Break-even")
        
        fig_savings.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_savings, use_container_width=True)
    
    with col2:
        # Grid Option NPV Distribution
        fig_grid = px.histogram(
            results_df, 
            x='npv_grid',
            nbins=30,
            title="Grid Option NPV Distribution",
            labels={'npv_grid': 'Grid NPV (Million $)', 'count': 'Frequency'},
            color_discrete_sequence=['#FF6B6B']
        )
        
        # Add percentile lines
        fig_grid.add_vline(x=analysis['npv_grid']['p10'], line_dash="dash", line_color="red",
                          annotation_text="10%")
        fig_grid.add_vline(x=analysis['npv_grid']['p50'], line_dash="dash", line_color="green",
                          annotation_text="50%")
        fig_grid.add_vline(x=analysis['npv_grid']['p90'], line_dash="dash", line_color="red",
                          annotation_text="90%")
        
        fig_grid.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_grid, use_container_width=True)
    
    with col3:
        # Proposed Option NPV Distribution
        fig_proposed = px.histogram(
            results_df, 
            x='npv_proposed',
            nbins=30,
            title="Proposed Option NPV Distribution",
            labels={'npv_proposed': 'Proposed NPV (Million $)', 'count': 'Frequency'},
            color_discrete_sequence=['#4ECDC4']
        )
        
        # Add percentile lines
        fig_proposed.add_vline(x=analysis['npv_proposed']['p10'], line_dash="dash", line_color="red",
                              annotation_text="10%")
        fig_proposed.add_vline(x=analysis['npv_proposed']['p50'], line_dash="dash", line_color="green",
                              annotation_text="50%")
        fig_proposed.add_vline(x=analysis['npv_proposed']['p90'], line_dash="dash", line_color="red",
                              annotation_text="90%")
        
        fig_proposed.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_proposed, use_container_width=True)
    
    # Summary statistics for all three NPVs
    st.subheader("📊 NPV Statistics Comparison")
    stats_df = pd.DataFrame({
        'Metric': ['Mean', '10th Percentile', 'Median (50th)', '90th Percentile', 'Std Deviation'],
        'NPV Savings': [
            f"${analysis['npv_savings']['mean']:.1f}M",
            f"${analysis['npv_savings']['p10']:.1f}M", 
            f"${analysis['npv_savings']['p50']:.1f}M",
            f"${analysis['npv_savings']['p90']:.1f}M",
            f"±${analysis['npv_savings']['std']:.1f}M"
        ],
        'Grid Option NPV': [
            f"${analysis['npv_grid']['mean']:.1f}M",
            f"${analysis['npv_grid']['p10']:.1f}M",
            f"${analysis['npv_grid']['p50']:.1f}M", 
            f"${analysis['npv_grid']['p90']:.1f}M",
            f"±${analysis['npv_grid']['std']:.1f}M"
        ],
        'Proposed Option NPV': [
            f"${analysis['npv_proposed']['mean']:.1f}M",
            f"${analysis['npv_proposed']['p10']:.1f}M",
            f"${analysis['npv_proposed']['p50']:.1f}M",
            f"${analysis['npv_proposed']['p90']:.1f}M", 
            f"±${analysis['npv_proposed']['std']:.1f}M"
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Sensitivity Analysis
    st.subheader("🎯 Sensitivity Analysis")
    sensitivity_data = pd.DataFrame({
        'Parameter': ['Grid Rate', 'NG Rate', 'Inflation Rate', 'Maintenance Cost', 'O&M Cost'],
        'Correlation': [
            analysis['sensitivity']['grid_rate'],
            analysis['sensitivity']['ng_rate'], 
            analysis['sensitivity']['inflation_rate'],
            analysis['sensitivity']['maintenance_cost_rate'],
            analysis['sensitivity']['om_cost']
        ]
    })
    
    fig_sens = px.bar(
        sensitivity_data,
        x='Correlation',
        y='Parameter',
        orientation='h',
        title="Correlation with NPV Savings",
        labels={'Correlation': 'Correlation Coefficient', 'Parameter': ''},
        color='Correlation',
        color_continuous_scale='RdBu'
    )
    fig_sens.update_layout(height=400)
    st.plotly_chart(fig_sens, use_container_width=True)
    
    # Scatter plot showing relationship between key variables
    st.subheader("🔍 Parameter Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter1 = px.scatter(
            results_df,
            x='grid_rate',
            y='npv_savings',
            title="NPV Savings vs Grid Rate",
            labels={'grid_rate': 'Grid Rate ($/kWh)', 'npv_savings': 'NPV Savings (Million $)'},
            opacity=0.6
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        fig_scatter2 = px.scatter(
            results_df,
            x='ng_rate', 
            y='npv_savings',
            title="NPV Savings vs Natural Gas Rate",
            labels={'ng_rate': 'NG Rate ($/GJ)', 'npv_savings': 'NPV Savings (Million $)'},
            opacity=0.6
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Risk Analysis Table
    st.subheader("📋 Risk Analysis Summary")
    risk_df = pd.DataFrame({
        'Metric': [
            'Best Case (90th percentile)',
            'Most Likely (Median)', 
            'Worst Case (10th percentile)',
            'Probability of Success',
            'Expected Value',
            'Standard Deviation',
            'Value at Risk (10%)'
        ],
        'NPV Savings (Million $)': [
            f"${analysis['npv_savings']['p90']:.1f}M",
            f"${analysis['npv_savings']['p50']:.1f}M",
            f"${analysis['npv_savings']['p10']:.1f}M",
            f"{analysis['npv_savings']['prob_positive']:.1f}%",
            f"${analysis['npv_savings']['mean']:.1f}M",
            f"±${analysis['npv_savings']['std']:.1f}M",
            f"${analysis['npv_savings']['p10']:.1f}M"
        ]
    })
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Click 'Run Monte Carlo Simulation' above to analyze uncertainty and risk in your business case")
