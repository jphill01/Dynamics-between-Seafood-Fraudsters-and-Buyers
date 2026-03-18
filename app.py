import streamlit as st
import pandas as pd
import numpy as np
from System import DynamicalSystem
from text import INTRO
from numpy.random import default_rng as rng
import plotly.graph_objects as go

system = DynamicalSystem(
    params={
        'gamma_m': 10.0,
        'gamma_f': 1.0,
        'gamma_s': 1.0,
        'gamma_e': 1.0,
        'gamma_p': 1.0,
        'gamma_fp': 1.0,
        'e_d': 1.0,
        'e_sw': 1.0,
        'e_sm': 1.0,
        'K': 1.0,
        
        'F_threshold': 0.5,
        'q': 0.07,
        'r': 0.225,
        'pw0': 1.0,
        'c0': 0.9,
        
        'pw1': 0.81,
        'c1': 0.153
    },
    state={
        'S': 0.6,
        'E': 0.3,
        'F': 0.1,
        'FP': 0.1
    }
)

simulation = system.time_series_plot(time=500)

# Start of page
st.set_page_config(
    layout="wide",
    page_title="Dynamics of Seafood, Fraudsters, and Buyers",
    page_icon=":fish:"
)

st.write(INTRO)
    
st.header("Time Series")
st.write("Time series of the system with its default parameters")
df = pd.DataFrame(
    np.array(list(zip(*simulation.values())), dtype="float64"),
    columns=simulation.keys()
)
st.line_chart(
    df,
    x_label="Time",
    y_label="Levels"
)

col1, col2 = st.columns(2)
with col1:
    st.header("Scenario: Market price elasticty of demand represents buyers' dependency on resource")
    res = system.ed_fp_demand(low_harvest=0.1, high_harvest=0.5)

    fig = go.Figure(data=[
        go.Surface(z=res['Low Harvest']),
        go.Surface(z=res['High Harvest']),
        # go.Surface(z=z2, showscale=False, opacity=0.9),
        # go.Surface(z=z3, showscale=False, opacity=0.9)

    ])

    st.plotly_chart(fig)
    

