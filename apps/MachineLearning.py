import streamlit as st
from PIL import Image
import base64
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.graph_objs import Layout
import plotly.express as px
import geopandas as gpd
import gspread as gs
import seaborn as sns
import matplotlib.pyplot as plt

gc = gs.service_account(filename = './cred.json')

sh1 = gc.open('Akreditasi Universitas').worksheet("SCORING TINGKAT PENDIDIKAN")

df1 = pd.DataFrame(sh1.get_all_records())

sh2 = gc.open('Akreditasi Universitas').worksheet("SCORING UNIVERSITAS")

df2 = pd.DataFrame(sh2.get_all_records())

sh3 = gc.open('Akreditasi Universitas').worksheet("SCORING JURUSAN")

df3 = pd.DataFrame(sh3.get_all_records())

def app():

    global df_test
    uploaded_file = st.file_uploader(label = "Choose a file", type=['xlsx'])

    df_test = pd.read_excel(r'./db/DATA_1.xlsx', engine='openpyxl', index_col=0)
    df_s = df_test.copy()
    df_s = df_s.drop(['PERFORMANCE LEVEL'], axis = 1)

    df_s['USIA'] = (40 - df_s['USIA'])/20
    df_s['IPK'] = df_s['IPK']/4

    df_s['TINGKAT PENDIDIKAN'] = df_s['TINGKAT PENDIDIKAN'].replace(
        df1['TINGKAT PENDIDIKAN'].tolist(), df1['SCORE'].tolist())

    df_s['UNIVERSITAS'] = df_s['UNIVERSITAS'].replace(df2['UNIVERSITAS'].tolist(), df2['SCORE'].tolist())

    df_s['JURUSAN'] = df_s['JURUSAN'].replace(df3['JURUSAN'].tolist(), df3['SCORE'].tolist())

    df_s['TIER PREDICTION'] = df_test["PERFORMANCE LEVEL"].copy()

    df_s = df_s.reset_index(drop = True)

    from sklearn.model_selection import train_test_split

    X = df_s.drop(columns=["NAMA", 'TIER PREDICTION', 'DOMISILI'])
    y = df_s["TIER PREDICTION"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.3,random_state=2)

    from sklearn import ensemble
    gb_clf = ensemble.GradientBoostingClassifier(n_estimators=50)
    gb_clf.fit(X_train,y_train)

    if uploaded_file is not None:
        df_test = pd.read_excel(uploaded_file,engine='openpyxl', index_col=0)
        df_s = df_test.copy()

        df_s['USIA'] = (40 - df_s['USIA'])/20
        df_s['IPK'] = df_s['IPK']/4
        
        df_s['TINGKAT PENDIDIKAN'] = df_s['TINGKAT PENDIDIKAN'].replace(
            df1['TINGKAT PENDIDIKAN'].tolist(), df1['SCORE'].tolist())

        df_s['UNIVERSITAS'] = df_s['UNIVERSITAS'].replace(df2['UNIVERSITAS'].tolist(), df2['SCORE'].tolist())

        df_s['JURUSAN'] = df_s['JURUSAN'].replace(df3['JURUSAN'].tolist(), df3['SCORE'].tolist())

        X = df_s.drop(columns=["NAMA", "DOMISILI"])

    df_s['TIER PREDICTION'] = gb_clf.predict(X)

    df_s['TIER PREDICTION'] = df_s['TIER PREDICTION'].replace(['TIER 1', 'TIER 2', 'TIER 3', 'TIER 4', 'TIER 5'], [4,3,2,1,0])

    fig, ax = plt.subplots()

    sns.set(rc={'figure.facecolor':(0,0,0,0)})
    sns.heatmap(df_s.corr(), ax=ax)

    df_s['TIER PREDICTION'] = gb_clf.predict(X)

    df_test = df_test.reset_index(drop = True)

    df_s[['DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN', 'IPK', 'USIA']] = df_test[['DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN', 'IPK', 'USIA']]

    tier = st.sidebar.multiselect(
        'Filter Tier', df_s['TIER PREDICTION'].unique(), default=None)
    all_tier = st.sidebar.checkbox("Pilih Semua Prediksi Tier", value=True)
    if all_tier:
        tier = df_s['TIER PREDICTION'].unique()
    major = st.sidebar.multiselect(
        'Filter Jurusan', df_s['JURUSAN'].unique(), default=None)
    all_major = st.sidebar.checkbox("Pilih Semua Jurusan", value=True)
    if all_major:
        major = df_s['JURUSAN'].unique()
    univ = st.sidebar.multiselect(
        'Filter Universitas', df_s['UNIVERSITAS'].unique(), default=None)
    all_univ = st.sidebar.checkbox("Pilih Semua Universitas", value=True)
    if all_univ:
        univ = df_s['UNIVERSITAS'].unique()
    dom = st.sidebar.multiselect(
        'Filter Domisili', df_s['DOMISILI'].unique(), default=None)
    all_dom = st.sidebar.checkbox("Pilih Semua Domisili", value=True)
    if all_dom:
        dom = df_s['DOMISILI'].unique()
    slide = st.sidebar.slider('Filter IPK', 0.00, 4.00, (3.00, 4.00))
    age = st.sidebar.slider('Filter Usia', 20, 30, (20, 30))

    df_s = df_s[df_s['TIER PREDICTION'].isin(tier) &
                        df_s['JURUSAN'].isin(major) &
                        df_s['UNIVERSITAS'].isin(univ) &
                        df_s['DOMISILI'].isin(dom) &
                        df_s['IPK'].between(slide[0], slide[1]) &
                        df_s['USIA'].between(age[0], age[1])]

    st.write('Data Calon Pegawai')
    st.dataframe(df_s.sort_values('TIER PREDICTION', ascending = True).reset_index(drop=True))

    st.markdown("""---""")

    #############################################

    col1, col2 = st.columns(2)

    with col1:

        st.write("AKURASI DATA: ", str(gb_clf.score(X_test, y_test)))

        st.markdown("""---""")

        st.write("KORELASI DATA")

        st.pyplot(fig)

    with col2:
        st.write("JUMLAH PELAMAR BERDASARKAN PREDIKSI TIER")

        dfp2 = df_s["TIER PREDICTION"].value_counts().rename_axis('TIER PREDICTION').reset_index(name='Total')
        dfp2 = dfp2.sort_values('TIER PREDICTION',ascending = True).reset_index(drop=True)

        labels2 = dfp2["TIER PREDICTION"]
        values2 = dfp2["Total"]

        layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

        fig2 = go.Figure(data = [go.Pie(labels = labels2, values = values2, hole = .3, sort=False)], layout = layout)
        fig2.update_traces(textinfo='value')
        st.plotly_chart(fig2, use_container_width = True)

    ############################# UNIVERSITAS ######################################

    dfy = df_s.groupby(['UNIVERSITAS', 'TIER PREDICTION']
                            ).size().reset_index(name='TOTAL')

    dfy = dfy.pivot(index='UNIVERSITAS',
                    columns='TIER PREDICTION', values='TOTAL')
    dfy = dfy.fillna(0)
    dfy = dfy.rename_axis(None, axis=1).reset_index()

    st.write('SEBARAN UNIVERSITAS')

    dfy = dfy[dfy['UNIVERSITAS'].isin(univ)]

    dfy['total'] = dfy.sum(axis=1)

    fig = px.bar(dfy.sort_values('total').tail(10), y = "UNIVERSITAS", x = dfy.columns[1:-1], title="Universitas")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=1100,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4
        )
    )
    fig.update_xaxes(
        # gridcolor = '#000000',
        # linecolor = '#000000',
        title='Jumlah Pelamar')
    # fig.update_yaxes(
    #     linecolor = '#000000')

    st.plotly_chart(fig)

    ######################## JURUSAN ###############################

    dfy = df_s.groupby(['JURUSAN', 'TIER PREDICTION']
                            ).size().reset_index(name='TOTAL')
    dfy = dfy.pivot(index='JURUSAN',
                    columns='TIER PREDICTION', values='TOTAL')
    dfy = dfy.fillna(0)
    dfy = dfy.rename_axis(None, axis=1).reset_index()
    
    st.write('SEBARAN JURUSAN')
    dfy = dfy[dfy['JURUSAN'].isin(major)]

    dfy['total'] = dfy.sum(axis=1)

    fig = px.bar(dfy.sort_values('total').tail(10), y = "JURUSAN", x = dfy.columns[1:-1], title="Jurusan")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='#ffffff',
        autosize=False,
        width=1100,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4
        )
    )
    fig.update_xaxes(
        # gridcolor = '#000000',
        # linecolor = '#000000',
        title='Jumlah Pelamar')
    # fig.update_yaxes(
    #     linecolor = '#000000')
    st.plotly_chart(fig)
