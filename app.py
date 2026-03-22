import streamlit as st
import pandas as pd
import numpy as np
from System import DynamicalSystem
from text import INTRO
import plotly.graph_objects as go

base_params = {
    'gamma_m': 5.0, 'gamma_f': 1.0, 'gamma_s': 1.0, 'gamma_e': 0.225,
    'gamma_p': 1.0, 'gamma_fp': 1.0, 'e_d': 1.0, 'e_sw': 1.0, 'e_sm': 1.0,
    'K': 1.0, 'F_threshold': 0.5, 'q': 0.07, 'r': 0.225, 'pw0': 1.0,
    'c0': 0.9, 'pw1': 0.81, 'c1': 0.153
}
init_state = {'S': np.float128(0.6), 'E': np.float128(0.3), 'F': np.float128(0.1), 'FP': np.float128(0.1)}

system = DynamicalSystem(
    params=base_params,
    state=init_state
)

# Start of page
st.set_page_config(
    layout="wide",
    page_title="Dynamics of Seafood, Fraudsters, and Buyers",
    page_icon=":fish:"
)

# INTRODUCTION
st.write(INTRO)
    
# TIME SERIES PLOT
st.header("Time Series")
st.write("Time series of the system with its default parameters")

@st.cache_data
def get_time_series_plot(_system: DynamicalSystem, time: int = 500) -> pd.DataFrame:
    simulation = _system.time_series_plot(time=time)
    keys = ['Seafood', 'Effort', 'Fraudsters', 'Perception of Fraud']
    df = pd.DataFrame(
        np.array(list(zip(*[simulation[k] for k in keys])), dtype="float64"),
        columns=keys
    )
    return df

st.line_chart(
    get_time_series_plot(system, time=500),
    x_label="Time",
    y_label="Levels"
)

# ELASTICITY OF DEMAND PLOT
if False:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Scenario: Market price elasticty of demand represents buyers' dependency on resource")
        res = system.ed_fp_demand(low_harvest=0.1, high_harvest=0.5)

        fig = go.Figure(data=[
            go.Surface(z=res['Low Harvest']),
            go.Surface(z=res['High Harvest']),
        ])
        st.plotly_chart(fig)
    
# System Dynamics: Varying gamma_m and pw1
if False:
    st.title("System Dynamics: Varying gamma_m and pw1")

    base_params = {
        'gamma_m': 20.0, 'gamma_f': 1.0, 'gamma_s': 1.0, 'gamma_e': 0.225,
        'gamma_p': 1.0, 'gamma_fp': 1.0, 'e_d': 1.0, 'e_sw': 1.0, 'e_sm': 1.0,
        'K': 1.0, 'F_threshold': 0.5, 'q': 0.07, 'r': 0.225, 'pw0': 1.0,
        'c0': 0.9, 'pw1': 0.81, 'c1': 0.153
    }
    init_state = {'S': np.float128(0.6), 'E': np.float128(0.3), 'F': np.float128(0.1), 'FP': np.float128(0.1)}

    system = DynamicalSystem(params=base_params, state=init_state)

    gamma_m_vals = [5.0, 7.5, 10.0, 12.5]
    pw1_vals = {'lower': 0.1, 'little_lower': 0.81, 'little_higher': 5.0, 'higher': 10.0}

    for g_m in gamma_m_vals:
        st.subheader(f"γ_m (gamma_m) = {g_m}")
        
        # Create 4 columns for this row
        cols = st.columns(4)
        
        # Iterate through the columns and pw1 values simultaneously
        for col, (pw1_label, pw1_val) in zip(cols, pw1_vals.items()):
            
            # Reset the state and update the parameters for this specific plot
            system.state = init_state.copy()
            system.params['gamma_m'] = g_m
            system.params['pw1'] = pw1_val
            
            # Run the simulation
            nut = system.time_series_plot(time=500) 
            
            # Convert dictionary data to a Pandas DataFrame for Streamlit
            df = pd.DataFrame({
                'Seafood (S)': np.array(nut['Seafood'], dtype="float64"),
                'Effort (E)': np.array(nut['Effort'], dtype="float64"),
                'Fraudsters (F)': np.array(nut['Fraudsters'], dtype="float64"),
                'Perception (FP)': np.array(nut['Perception of Fraud'], dtype="float64")
            })
            
            # Place the chart inside the specific column
            with col:
                st.markdown(f"**pw1 = {pw1_val}** ({pw1_label})")
                st.line_chart(df, height=300)




