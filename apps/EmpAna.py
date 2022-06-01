from turtle import position
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go

def app():
    df = pd.read_csv(r"C:\Users\shidd\Downloads\HRDataset_v14.csv")

    # filtered = st.multiselect(
    #     "Filter columns:",
    #     options=list(df.columns),
    #     default=[]
    # )

    # st.write(df[filtered])

    PickEmp = df.PerfScoreID + df.SpecialProjectsCount
    df = df.loc[PickEmp.sort_values(ascending= False).index]

    Name = st.selectbox(
        "Select Employee:",
        options=df["Employee_Name"].unique()
    )
    
    df_selection = df.query(
        "Employee_Name == @Name"
    )

    st.dataframe(df_selection)

