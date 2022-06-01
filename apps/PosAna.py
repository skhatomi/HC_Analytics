from turtle import position
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from functools import reduce

def app():
    df = pd.read_csv(r"C:\Users\shidd\Downloads\HRDataset_v14.csv")

    Position = st.selectbox(
        "Select Position:",
        options=df["Position"].unique()
    )

    df_selection = df.query(
    "Position == @Position"
    )

    st.dataframe(df_selection)

    with st.container():
        st.write("Number of Employee: ", len(df_selection))
        st.write("Average Salary: ", round(sum(df_selection["Salary"])/len(df_selection), 2))
        st.write("Average Score: ", round(sum(df_selection["PerfScoreID"])/len(df_selection), 2))
        st.write("Average Satisfaction Score: ", round(sum(df_selection["EmpSatisfaction"])/len(df_selection), 2))

    #PerformanceScore
    #dfp2 = df["PerformanceScore"].value_counts().rename_axis('PerformanceScore').reset_index(name='Total')
    #labels2 = dfp2["PerformanceScore"]
    #values2 = dfp2["Total"]

    #fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)])
    #st.plotly_chart(fig2, use_container_width = True)

    
    #Sex
    #dfp3 = df["Sex"].value_counts().rename_axis('Sex').reset_index(name='Total')
    #labels3 = dfp3["Sex"]
    #values3 = dfp3["Total"]

    #fig3 = go.Figure(data=[go.Pie(labels=labels3, values=values3, hole=.3)])
    #st.plotly_chart(fig3, use_container_width = True)

    #CitizenDesc
    #dfp4 = df["CitizenDesc"].value_counts().rename_axis('CitizenDesc').reset_index(name='Total')
    #labels4 = dfp4["CitizenDesc"]
    #values4 = dfp4["Total"]

    #fig4 = go.Figure(data=[go.Pie(labels=labels4, values=values4, hole=.3)])
    #st.plotly_chart(fig4, use_container_width = True)

    #Department
    #dfp5 = df["Department"].value_counts().rename_axis('Department').reset_index(name='Total')
    #labels5 = dfp5["Department"]
    #values5 = dfp5["Total"]

    #fig5 = go.Figure(data=[go.Pie(labels=labels5, values=values5, hole=.3)])
    #st.plotly_chart(fig5, use_container_width = True)
