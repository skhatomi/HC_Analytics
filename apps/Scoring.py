# from threading import Timer
# from tkinter import E
import streamlit as st
# from PIL import Image
# import base64
import pandas as pd
# import numpy as np
import plotly.graph_objects as go
# import plotly.figure_factory as ff
from plotly.graph_objs import Layout
import plotly.express as px
# import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import gspread as gs

gc = gs.service_account(filename = 'cred.json')

sh1 = gc.open('Akreditasi Universitas').worksheet("SCORING TINGKAT PENDIDIKAN")

df1 = pd.DataFrame(sh1.get_all_records())

sh2 = gc.open('Akreditasi Universitas').worksheet("SCORING UNIVERSITAS")

df2 = pd.DataFrame(sh2.get_all_records())

sh3 = gc.open('Akreditasi Universitas').worksheet("SCORING JURUSAN")

df3 = pd.DataFrame(sh3.get_all_records())

def app():

    uploaded_file = st.file_uploader(label = "Choose a file", type=['xlsx'])
    global df_test

    if uploaded_file is not None:
        df_test = pd.read_excel(uploaded_file,engine='openpyxl', index_col=0)
        df_s = df_test.copy()
    
    else:
        df_test = pd.read_excel(r'./db/DATA_1.xlsx', engine='openpyxl', index_col=0)
        df_k = df_test.copy()
        df_test = df_test.drop(['PERFORMANCE LEVEL'], axis = 1)
        df_s = df_test.copy()
    
    df_test.columns = df_test.columns.str.replace(
        'PERFORMANCE LEVEL', 'PREDIKSI PERFORMANCE LEVEL')

    df_s['USIA'] = (40 - df_s['USIA'])/20
    df_s['IPK'] = df_s['IPK']/4

    df_s['TINGKAT PENDIDIKAN'] = df_s['TINGKAT PENDIDIKAN'].replace(
        df1['TINGKAT PENDIDIKAN'].tolist(), df1['SCORE'].tolist())

    df_s['UNIVERSITAS'] = df_s['UNIVERSITAS'].replace(df2['UNIVERSITAS'].tolist(), df2['SCORE'].tolist())

    df_s['JURUSAN'] = df_s['JURUSAN'].replace(df3['JURUSAN'].tolist(), df3['SCORE'].tolist())

    col1, col2, col4, col5, col6 = st.columns(5)
    with col1:
        a = st.number_input("IPK", min_value = 0, max_value = 10, value = 1)
    with col2:
        b = st.number_input("USIA", min_value = 0, max_value = 10, value = 1)
    # with col3:
    #     c = st.number_input("DOMISILI", min_value = 0, max_value = 10, value = 1)
    with col4:
        d = st.number_input("TINGKAT PENDIDIKAN", min_value = 0, max_value = 10, value = 1)
    with col5:
        e = st.number_input("UNIVERSITAS", min_value = 0, max_value = 10, value = 1)
    with col6:
        f = st.number_input("JURUSAN", min_value = 0, max_value = 10, value = 1)
    
    df_s['SCORE'] = (a*df_s['IPK'] + b*df_s['USIA'] + d*df_s['TINGKAT PENDIDIKAN'] + e*df_s['UNIVERSITAS'] + f*df_s['JURUSAN'])/(a+b+d+e+f)
    df_s['TIER PREDICTION'] = df_s['SCORE']
    df_s = df_s.reset_index(drop=True)
    for i in range(len(df_s['TIER PREDICTION'])):
        if 0 <= df_s['TIER PREDICTION'][i] <= 0.2:
            df_s['TIER PREDICTION'][i] = 'TIER 5'
        elif 0.2 < df_s['TIER PREDICTION'][i] <= 0.4:
            df_s['TIER PREDICTION'][i] = 'TIER 4'
        elif 0.4 < df_s['TIER PREDICTION'][i] <= 0.6:
            df_s['TIER PREDICTION'][i] = 'TIER 3'
        elif 0.6 < df_s['TIER PREDICTION'][i] <= 0.8:
            df_s['TIER PREDICTION'][i] = 'TIER 2'
        else:
            df_s['TIER PREDICTION'][i] = 'TIER 1'
    df_s['SCORE'] = df_s['SCORE'] * 100

    fig, ax = plt.subplots()

    sns.set(rc={'figure.facecolor':(0,0,0,0)})
    sns.heatmap(df_s.corr(), ax=ax)

    df_test = df_test.reset_index(drop=True)

    df_s[['IPK', 'USIA', 'DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN']] = df_test[[
        'IPK', 'USIA', 'DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN']]
    
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
    st.dataframe(df_s.sort_values('SCORE', ascending = False).reset_index(drop=True))

    st.markdown("""---""")

    #################################

    col1, col2 = st.columns(2)

    with col1:
        
        try:
            df_k = df_k.reset_index(drop=True)
            a = 0
            for i in range (len(df_s)):
                if df_k['PERFORMANCE LEVEL'][i] == df_s['TIER PREDICTION'][i]:
                    a += 1
            st.write("Akurasi Data: ", str(a/(len(df_s))))
        except:
            pass
        st.pyplot(fig)

    with col2:
        dfp2 = df_s["TIER PREDICTION"].value_counts().rename_axis('TIER PREDICTION').reset_index(name='Total')
        dfp2 = dfp2.sort_values('TIER PREDICTION',ascending = True).reset_index(drop=True)

        labels2 = dfp2["TIER PREDICTION"]
        values2 = dfp2["Total"]

        layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

        fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3, sort=False)], layout =layout)
        fig2.update_traces(textinfo='value')
        st.plotly_chart(fig2, use_container_width = True)

    # ##################################################

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.write('Total Universitas')
    #     dfz = df_s.groupby(
    #         ['UNIVERSITAS']).size().reset_index(name='TOTAL')
    #     st.dataframe(dfz)

    # with col2:
    #     st.write('Total Domisili')
    #     dfzz = df_s.groupby(
    #         ['DOMISILI']).size().reset_index(name='TOTAL')
    #     st.dataframe(dfzz)

    # with col3:
    #     st.write('\n')
    #     st.write('\n')
    #     try:
    #         st.write("Rata-rata usia: ",
    #                  str(round(sum(df_s["USIA"])/len(df_s), 2)), "Tahun")
    #         st.write("Rata-rata IPK: ",
    #                  str(round(sum(df_s["IPK"])/len(df_s), 2)))
    #     except:
    #         st.write('Tidak ada data yang ditampilkan')
    
    ########################## DOMISILI ################################

    # dfy = df_s.groupby(['DOMISILI', 'TIER PREDICTION']
    #                       ).size().reset_index(name='TOTAL')
    # dfy = dfy.pivot(index='DOMISILI',
    #                 columns='TIER PREDICTION', values='TOTAL')
    # dfy = dfy.fillna(0)
    # dfy = dfy.rename_axis(None, axis=1).reset_index()
    # for i in tier:
    #     dfy = dfy.astype({i : 'int64'})
    # dfy = dfy[dfy['DOMISILI'].isin(dom)]
    # st.write('SEBARAN DOMISILI')
    # dfy['total'] = dfy.sum(axis=1)

    # fig = px.bar(dfy.sort_values('total', ascending=False), x = "DOMISILI", y = tier, title="Domisili")

    # fig.update_layout(
    #     autosize=False,
    #     width=1430,
    #     height=400,
    #     margin=dict(
    #         l=0,
    #         r=0,
    #         b=0,
    #         t=0,
    #         pad=4
    #     ),
    #     paper_bgcolor="LightSteelBlue",
    # )
    # st.plotly_chart(fig)

    ######################## MAP DOMISILI ##########################

    # dfg = pd.read_csv(r'D:/map.csv', index_col=0)

    # df_6 = df_s[["DOMISILI", "TIER PREDICTION"]]
    # scoring = {'JAKARTA': 'Jakarta Raya',
    #            'MAKASSAR': 'Sulawesi Selatan',
    #            'BANJARMASIN': 'Kalimantan Selatan',
    #            'MEDAN': 'Sumatera Utara',
    #            'PALEMBANG': 'Sumatera Selatan',
    #            'PADANG': 'Sumatera Barat',
    #            'SURABAYA': 'Jawa Timur',
    #            'DENPASAR': 'Bali',
    #            'MANADO': 'Sulawesi Utara',
    #            'YOGYAKARTA': 'Yogyakarta',
    #            'JAYAPURA': 'Papua',
    #            'BANDUNG': 'Jawa Barat',
    #            'SEMARANG': 'Jawa Tengah'
    #            }
    # df_6['DOMISILI'] = df_6['DOMISILI'].map(scoring)
    # dfz = df_6.groupby(['DOMISILI', 'TIER PREDICTION']).size(
    # ).reset_index(name='counts')
    # dfz = dfz.pivot(index='DOMISILI',
    #                 columns='TIER PREDICTION', values='counts')
    # dfz = dfz.fillna(0)
    # dfz = dfz.rename_axis(None, axis=1).reset_index()

    # df_join = dfg.merge(
    #     dfz, how='left', left_on="NAME_1", right_on="DOMISILI")
    # ap = ['DOMISILI', 'geometry']
    # ap1 = []
    # for i in tier:
    #     ap1.append(i)
    # df_join = df_join[ap+ap1]

    # df_join = df_join.fillna(0)

    # df_join.at[0, "DOMISILI"] = "Aceh"
    # df_join.at[2, "DOMISILI"] = "Bangka Belitung"
    # df_join.at[3, "DOMISILI"] = "Banten"
    # df_join.at[4, "DOMISILI"] = "Bengkulu"
    # df_join.at[5, "DOMISILI"] = "Gorontalo"
    # df_join.at[7, "DOMISILI"] = "Jambi"
    # df_join.at[11, "DOMISILI"] = "Kalimantan Barat"
    # df_join.at[13, "DOMISILI"] = "Kalimantan Tengah"
    # df_join.at[14, "DOMISILI"] = "Kalimantan Timur"
    # df_join.at[15, "DOMISILI"] = "Kepulauan Riau"
    # df_join.at[16, "DOMISILI"] = "Lampung"
    # df_join.at[17, "DOMISILI"] = "Maluku"
    # df_join.at[18, "DOMISILI"] = "Maluku Utara"
    # df_join.at[19, "DOMISILI"] = "Nusa Tenggara Barat"
    # df_join.at[20, "DOMISILI"] = "Nusa Tenggara Timur"
    # df_join.at[22, "DOMISILI"] = "Papua Barat"
    # df_join.at[23, "DOMISILI"] = "Riau"
    # df_join.at[24, "DOMISILI"] = "Sulawesi Barat"
    # df_join.at[26, "DOMISILI"] = "Sulawesi Tengah"
    # df_join.at[27, "DOMISILI"] = "Sulawesi Tenggara"

    # from geopandas import GeoDataFrame
    # df_join = GeoDataFrame(df_join)

    # from shapely import wkt
    # df_join['geometry'] = df_join['geometry'].apply(wkt.loads)

    # import numpy as np
    # import matplotlib.pyplot as plt

    # values = st.selectbox(
    #     "Pilih Tier", ap1)

    # dom = st.multiselect(
    #     'Filter Domisili', df_join['DOMISILI'], default=df_join['DOMISILI'])

    # df_join = df_join[df_join['DOMISILI'].isin(dom)]

    # vmin, vmax = df_join[values].min(), df_join[values].max()
    # # create figure and axes for Matplotlib
    # fig, ax = plt.subplots(1, figsize=(30, 10))
    # # remove the axis
    # ax.axis('off')
    # # add a title
    # title = 'Level Peformance : {}'.format(values)
    # ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
    # # create an annotation for the data source
    # ax.annotate('Source: HC BNI', xy=(0.1, .08),  xycoords='figure fraction',
    #             horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
    # # Create colorbar as a legend

    # sm = plt.cm.ScalarMappable(
    #     cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # # add the colorbar to the figure
    # cbar = fig.colorbar(sm)
    # # create map
    # df_join.plot(column=values, cmap='OrRd', linewidth=0.8, ax=ax,
    #              edgecolor='0.8', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # df_join['coords'] = df_join['geometry'].apply(
    #     lambda x: x.representative_point().coords[:])
    # df_join['coords'] = [coords[0] for coords in df_join['coords']]
    # for idx, row in df_join.iterrows():
    #     plt.annotate(s=row['DOMISILI'], xy=row['coords'],
    #                  horizontalalignment='center', fontsize=7)

    # st.pyplot(fig)

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

    ########################JURUSAN###############################

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
