from turtle import position
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from functools import reduce
from plotly.graph_objs import *

def app():
    df = pd.read_csv(r"C:\Users\shidd\Downloads\HRDataset_v14.csv")

    #NumberOfEmployee
    df2 = df["Position"].value_counts().rename_axis('Position').reset_index(name='Number of Employees')

    #AverageSalary
    df3 = df[['Position', 'Salary']].groupby(['Position']).mean()["Salary"].rename_axis('Position').reset_index(name='Average of Salary')

    #AverageScore
    df4 = df[['Position', 'PerfScoreID']].groupby(['Position']).mean()["PerfScoreID"].rename_axis('Position').reset_index(name='Average of Score')

    #AverageSatisfactionScore
    df5 = df[['Position', 'EmpSatisfaction']].groupby(['Position']).mean()["EmpSatisfaction"].rename_axis('Position').reset_index(name='Average of Satisfaction Score')

    #Merge
    dfs = [df2, df3, df4, df5]
    dft = reduce(lambda df_left, df_right: pd.merge(df_left, df_right, on = "Position"), dfs)
    #dft = dft.style.set_properties(**{'background-color': 'black', 'color': 'green'})

    st.write(dft.style.set_precision(2))
    fig = ff.create_table(dft)
    fig.update_layout(width=1000, height=600)
    st.write(fig)

    #PerformanceScore
    dfp2 = df["PerformanceScore"].value_counts().rename_axis('PerformanceScore').reset_index(name='Total')
    labels2 = dfp2["PerformanceScore"]
    values2 = dfp2["Total"]

    layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

    fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)], layout=layout)
    st.plotly_chart(fig2, use_container_width = True)

    
    #Sex
    dfp3 = df["Sex"].value_counts().rename_axis('Sex').reset_index(name='Total')
    labels3 = dfp3["Sex"]
    values3 = dfp3["Total"]

    fig3 = go.Figure(data=[go.Pie(labels=labels3, values=values3, hole=.3)])
    st.plotly_chart(fig3, use_container_width = True)

    #CitizenDesc
    dfp4 = df["CitizenDesc"].value_counts().rename_axis('CitizenDesc').reset_index(name='Total')
    labels4 = dfp4["CitizenDesc"]
    values4 = dfp4["Total"]

    fig4 = go.Figure(data=[go.Pie(labels=labels4, values=values4, hole=.3)])
    st.plotly_chart(fig4, use_container_width = True)

    #Department
    dfp5 = df["Department"].value_counts().rename_axis('Department').reset_index(name='Total')
    labels5 = dfp5["Department"]
    values5 = dfp5["Total"]

    fig5 = go.Figure(data=[go.Pie(labels=labels5, values=values5, hole=.3)])
    st.plotly_chart(fig5, use_container_width = True)
