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

#logo tab
image = Image.open('./image/logo-bni.png', mode='r')

st.set_page_config(
    page_title="BNI-HACTICS",
    page_icon=image,
    layout="wide",
    initial_sidebar_state="expanded",
 )

#hide streamlit label
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

container = st.container()

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type='password')
# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

#header
def header(url):
     st.markdown(f'<center><p style="background-color:#FC6608;color:#000000;font-size:30px;border-radius:30px 0px 30px 0px;">{url}</p></center>', unsafe_allow_html=True)

#background
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('./image/background1.jpg')

if st.sidebar.checkbox("Login"):
    # st.balloons()

    if username == 'user' and password == 'bni46':

        header("TALENT ACQUISITION")
          
        del container

        uploaded_file = st.file_uploader(label = "Choose a file", type=['xlsx'])
        global df_test
        if uploaded_file is not None:
            df_test = pd.read_excel(uploaded_file,engine='openpyxl')
            df_test = df_test.drop(df_test.columns[0], axis = 1)
            df_s = df_test.copy()
        else:
            df_test = pd.read_excel('./db/HASIL.xlsx', engine='openpyxl', index_col=0)
            df_test = df_test.drop(['PERFORMANCE LEVEL'], axis = 1)
            df_s = df_test.copy()

        df_test.columns = df_test.columns.str.replace(
            'PERFORMANCE LEVEL', 'PREDIKSI PERFORMANCE LEVEL')

        df_s['USIA'] = (40 - df_s['USIA'])/20
        df_s['IPK'] = df_s['IPK']/4

        d_1 = ['DENPASAR',
               'JAKARTA',
               'YOGYAKARTA',
               'SURABAYA',
               'PADANG',
               'MAKASSAR',
               'SEMARANG',
               'PALEMBANG',
               'JAYAPURA',
               'BANJARMASIN',
               'MEDAN',
               'MANADO',
               'BANDUNG']
        d_2 = [0.7586,
               0.8498,
               0.9012,
               0.6633,
               0.7006,
               0.6943,
               0.5990,
               0.6720,
               0.3295,
               0.6359,
               0.7281,
               0.6856,
               0.6489]
        df_s['DOMISILI'] = df_s['DOMISILI'].replace(d_1, d_2)

        t_1 = ['S1',
               'S2',
               'D4']
        t_2 = [0.9,
               1,
               0.85]
        df_s['TINGKAT PENDIDIKAN'] = df_s['TINGKAT PENDIDIKAN'].replace(
            t_1, t_2)

        u_1 = ['UNIVERSITAS MATARAM', 'UNIVERSITAS UDAYANA', 'UNIVERSITAS RIAU',
               'UNIVERSITAS PADJADJARAN', 'UNIVERSITAS INDONESIA',
               'INSTITUT TEKNOLOGI BANDUNG', 'UNIVERSITAS WIDYATAMA',
               'UNIVERSITAS AIRLANGGA', 'UNIVERSITAS BINA NUSANTARA',
               'UNIVERSITAS GADJAH MADA', 'UNIVERSITAS ANDALAS',
               'INSTITUT PERTANIAN BOGOR',
               'UNIVERSITAS ISLAM INDONESIA YOGYAKARTA', 'UNIVERSITAS BRAWIJAYA',
               'UNIVERSITAS NEGERI YOGYAKARTA', 'UNIVERSITAS HASANUDDIN',
               'UNIVERSITAS SEBELAS MARET', 'UNIVERSITAS SUMATERA UTARA',
               'UNIVERSITAS TELKOM', 'UNIVERSITAS PANCASILA', 'UNIVERSITAS JAMBI',
               'UNIVERSITAS PEMBANGUNAN NASIONAL VETERAN YOGYAKARTA',
               'UNIVERSITAS MUSLIM INDONESIA', 'UNIVERSITAS JENDERAL SOEDIRMAN',
               'UNIVERSITAS DIPONEGORO', 'UNIVERSITI UTARA MALAYSIA',
               'UNIVERSITAS LAMPUNG', 'INSTITUT TEKNOLOGI SEPULUH NOPEMBER',
               'UNIVERSITAS KRISTEN MARANATHA', 'UNIVERSITAS GUNADARMA',
               'UNIVERSITAS TRISAKTI', 'UNIVERSITAS BENGKULU',
               'UNIVERSITAS SRIWIJAYA', 'UNIVERSITAS CENDERAWASIH',
               'UNIVERSITAS MERCU BUANA', 'UNIVERSITAS LAMBUNG MANGKURAT',
               'INSTITUT MANAJEMEN TELKOM', 'UNIVERSITAS ARMAJAYA',
               'UNIVERSITAS TANJUNGPURA', 'UNIVERSITAS SULTAN AGENG TIRTAYASA',
               'UNIVERSITAS ISLAM INDONESIA', 'UNIVERSITAS BUDI LUHUR',
               'UNIVERSITY OF BRADFORD',
               'UNIVERSITAS KATOLIK INDONESIA ATMA JAYA',
               'UNIVERSITAS PASUNDAN UNPAS', 'FU JEN CATHOLIC UNIVERSITY',
               'SEKOLAH TINGGI MANAJEMEN PPM JAKARTA', 'UNIVERSITAS PARAMADINA',
               'UNIVERSITAS JEMBER', 'UNIVERSITAS SAM RATULANGI',
               'UNIVERSITAS NEGERI SEMARANG', 'TAIWAN TECH',
               'UNIVERSITAS MULAWARMAN', 'UNIVERSITAS NEGERI MEDAN',
               'NATIONAL CENTRAL UNIVERSITY', 'UNIVERSITAS ISLAM NEGERI ALAUDDIN',
               'SEKOLAH TINGGI ILMU EKONOMI HARAPAN',
               'UNIVERSITAS MUHAMMADIYAH MALANG', 'UNIVERSITAS DIAN NUSWANTORO',
               'UNIVERSITY OF BIRMINGHAM',
               'UNIVERSITAS ISLAM NEGERI SUNAN GUNUNG DJATI BANDUNG',
               'UNIVERSITAS PRESIDEN', 'POLITEKNIK NEGERI UJUNG PANDANG',
               'UNIVERSITAS NEGERI MALANG', 'UNIVERSITAS NEGERI JAKARTA',
               'UNIVERSITAS ISLAM NEGERI UIN SUNAN KALIJAGA',
               'UNIVERSITAS PEMBANGUNAN NASIONAL VETERAN',
               'UNIVERSITAS PENDIDIKAN INDONESIA', 'STIKOM LSPR JAKARTA',
               'UNIVERSITAS SURABAYA', 'UNIVERSITAS KATOLIK PARAHYANGAN',
               'UNIVERSITAS NEGERI PADANG',
               'UNIVERSITAS PEMBANGUNAN NASIONAL VETERAN JATIM',
               'UNIVERSITAS SYIAH KUALA', 'ZHEJIANG UNIVERSITY',
               'UNIVERSITAS SURYAKANCANA', 'UNIVERSITAS TARUMANAGARA',
               'SEKOLAH TINGGI ILMU EKONOMI PERBANAS SURABAYA',
               'UNIVERSITAS ISLAM SUMATERA UTARA',
               'UNIVERSITAS PERTAHANAN REPUBLIK INDONESIA',
               'SEKOLAH TINGGI TEKNIK PLN', 'WESTMINSTER INTERNATIONAL',
               'UNIVERSITAS KRISTEN DUTA WACANA',
               'STINSTITUT TEKNOLOGI TARPANGERAN DIPONEGORO',
               'UNIVERSITAS PATTIMURA', 'SHANGHAI INT. STUDIES UNIV.',
               'UNIVERSITAS PELITA HARAPAN',
               'UNIVERSITAS KRISTEN SATYA WACANA UKSW',
               'UNIVERSITAS ISLAM SULTAN AGUNG']
        u_2 = [0.6,
               0.9,
               0.5,
               1,
               1,
               1,
               0.2,
               1,
               0.9,
               1,
               0.8,
               1,
               0.8,
               1,
               0.8,
               0.9,
               0.9,
               0.8,
               0.8,
               0.3,
               0.4,
               0.6,
               0.4,
               0.8,
               0.9,
               1,
               0.8,
               1,
               0.4,
               0.8,
               0.6,
               0.7,
               0.7,
               0.3,
               0.5,
               0.4,
               0.2,
               0.6,
               0.6,
               0.6,
               0.8,
               0.3,
               1,
               0.7,
               0.6,
               1,
               0.8,
               0.3,
               0.7,
               0.5,
               0.7,
               1,
               0.6,
               0.6,
               1,
               0.6,
               0.3,
               0.7,
               0.5,
               1,
               0.6,
               0.4,
               0.4,
               0.8,
               0.7,
               0.7,
               0.7,
               0.8,
               0.4,
               0.6,
               0.5,
               0.7,
               0.4,
               0.7,
               1,
               0.3,
               0.3,
               0.3,
               0.3,
               0.1,
               0.3,
               1,
               0.3,
               0.1,
               0.2,
               1,
               0.4,
               0.5,
               0.5
               ]
        df_s['UNIVERSITAS'] = df_s['UNIVERSITAS'].replace(u_1, u_2)

        j_1 = ['ELEKTRO', 'HUKUM', 'AKUNTANSI', 'STATISTIKA', 'MIPA',
               'EKONOMI DAN BISNIS', 'MANAJEMEN',
               'TEKNOLOGI INFORMASI DAN INFORMATIKA', 'TEKNIK', 'KOMPUTER',
               'AKUNTASI', 'TEKNIK SIPIL', 'TEKNIK MATERIAL DAN METALURGI',
               'ADMINISTRASI', 'AGRIBISNIS', 'SOSIAL DAN POLITIK', 'MANAJAMEN',
               'KOMUNIKASI', 'ARSITEKTUR', 'AGROTEKNOLOGI', 'TEKNIK INDUSTRI',
               'PERTANIAN', 'TEKNIK PERTAMBANGAN DAN PERMINYAKAN',
               'PERAIRAN DAN KELAUTAN', 'PERENCANAAN WILAYAH DAN KOTA',
               'PSIKOLOGI', 'GEOLOGI DAN GEODESI', 'PETERNAKAN', 'PENDIDIKAN',
               'TEKNIK PERKAPALAN', 'TEKNIK MESIN', 'BUDIDAYA', 'DESAIN',
               'KESEHATAN', 'LINGKUNGAN', 'PERIKANAN', 'ANTROPOLOGI']
        j_2 = [0.7,
               0.9,
               1,
               1,
               0.6,
               1,
               1,
               0.9,
               0.8,
               0.9,
               1,
               0.7,
               0.6,
               0.9,
               0.6,
               0.6,
               1,
               0.9,
               0.7,
               0.6,
               0.7,
               0.6,
               0.7,
               0.6,
               0.6,
               0.9,
               0.6,
               0.6,
               0.3,
               0.7,
               0.7,
               0.4,
               0.6,
               0.3,
               0.6,
               0.6,
               0.6
               ]
        df_s['JURUSAN'] = df_s['JURUSAN'].replace(j_1, j_2)

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
        
        df_s['SCORE'] = (a*df_s['IPK'] + b*df_s['USIA'] +
                         d*df_s['TINGKAT PENDIDIKAN'] + e*df_s['UNIVERSITAS'] + f*df_s['JURUSAN'])/(a+b+d+e+f)
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

        for i in tier:
            dfy = dfy.astype({i : 'int64'})
        st.write('SEBARAN UNIVERSITAS')

        dfy = dfy[dfy['UNIVERSITAS'].isin(univ)]

        dfy['total'] = dfy.sum(axis=1)

        fig = px.bar(dfy.sort_values('total').tail(10), y = "UNIVERSITAS", x = tier, title="Universitas")
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
     
        st.markdown("""---""")

        ########################JURUSAN###############################

        dfy = df_s.groupby(['JURUSAN', 'TIER PREDICTION']
                              ).size().reset_index(name='TOTAL')
        dfy = dfy.pivot(index='JURUSAN',
                        columns='TIER PREDICTION', values='TOTAL')
        dfy = dfy.fillna(0)
        dfy = dfy.rename_axis(None, axis=1).reset_index()
        for i in tier:
            dfy = dfy.astype({i : 'int64'})
        
        st.write('SEBARAN JURUSAN')
        dfy = dfy[dfy['JURUSAN'].isin(major)]

        dfy['total'] = dfy.sum(axis=1)

        fig = px.bar(dfy.sort_values('total').tail(10), y = "JURUSAN", x = tier, title="Jurusan")
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

    else:
        st.sidebar.error("Invalid username or password")

try:
    container.write("")
    image = Image.open('./image/BNI_Hactics_Horizontal-removebg-preview.png')

    st.image(image)

except:
    pass
