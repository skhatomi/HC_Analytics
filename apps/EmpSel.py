from turtle import position
import streamlit as st
import pandas as pd
import altair as alt

def app():
    uploaded_file = st.file_uploader(label="Upload", type=['xlsx'])

    df_test = pd.read_excel(uploaded_file)

    
