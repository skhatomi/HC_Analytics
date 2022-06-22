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

# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

#header
def header(url):
     st.markdown(f'<center><p style="background-color:#FC6608;color:#000000;font-size:60px;border-radius:30px 0px 30px 0px;">{url}</p></center>', unsafe_allow_html=True)

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

header("TALENT ACQUISITION")

df_test = pd.read_excel('./db/HASIL.xlsx', engine='openpyxl', index_col=0)
df_rest = df_test

# df_rest = df_rest.reset_index(drop = True)

# df_test = df_test.drop(columns=["NAMA"])

# from sklearn.model_selection import train_test_split

# df = pd.read_excel('D:/data_train_1.xlsx', engine='openpyxl', index_col=0)

# X = df.drop(columns=["PERFORMANCE LEVEL"])

# y = df["PERFORMANCE LEVEL"]

# #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=1)

# from sklearn import preprocessing
    
# lab = preprocessing.LabelEncoder()
# y_transformed = lab.fit_transform(y)
# #y_test_tr = lab.fit_transform(y_test)

# from sklearn import ensemble
# gb_clf = ensemble.GradientBoostingClassifier(n_estimators=50)
# gb_clf.fit(X,y_transformed)

# #rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
# #rf_clf.fit(X, y_transformed)

# a1 = pd.DataFrame(X.columns, columns = ['colname'])

# dum1 = pd.get_dummies(df_test['DOMISILI'])
# dum2 = pd.get_dummies(df_test['TINGKAT PENDIDIKAN'])
# dum3 = pd.get_dummies(df_test['UNIVERSITAS'])
# dum4 = pd.get_dummies(df_test['JURUSAN'])

# df_test = pd.concat((df_test, dum1, dum2, dum3, dum4), axis=1)

# df_test = df_test.drop(['DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN'], axis = 1)

# b = pd.DataFrame(df_test.columns, columns = ['colname'])
    
# b2 = a1[~a1.colname.isin(b.colname)]
# b2 = b2.reset_index(drop = True)

# for i in range (len(b2)):
#     df_test[b2['colname'][i]] = 0

# df_test = df_test.reset_index(drop = True)

# result = gb_clf.predict(df_test)

# df_result = pd.DataFrame(result, columns = ['PERFORMANCE LEVEL'])

# df_result = df_result.replace({0: 'TIER 1', 1: 'TIER 2', 2: 'TIER 3', 3: 'TIER 4', 4: 'TIER 5'})

# df_rest = pd.concat((df_result, df_rest), axis=1)

# df_rest = df_rest.sort_values('PERFORMANCE LEVEL',ascending = True)

#df_rest.to_excel(r'D:/HASIL.xlsx')


# makes = df_rest['JURUSAN'].drop_duplicates()
# make_choice = st.sidebar.selectbox('Select your vehicle:', makes)

# df_rest.loc[df_rest[JURUSAN]=make_choice]

# st.dataframe(df_rest) 

tier = st.sidebar.multiselect('Filter Tier', df_rest['PERFORMANCE LEVEL'].unique(), default = None)
all_tier = st.sidebar.checkbox("Pilih Semua Tier", value = True)
if all_tier:
    tier = df_rest['PERFORMANCE LEVEL'].unique()
major = st.sidebar.multiselect('Filter Jurusan', df_rest['JURUSAN'].unique(), default = None)
all_major = st.sidebar.checkbox("Pilih Semua Jurusan", value = True)
if all_major:
    major = df_rest['JURUSAN'].unique()
univ = st.sidebar.multiselect('Filter Universitas', df_rest['UNIVERSITAS'].unique(), default = None)
all_univ = st.sidebar.checkbox("Pilih Semua Universitas", value = True)
if all_univ:
    univ = df_rest['UNIVERSITAS'].unique()
dom = st.sidebar.multiselect('Filter Domisili', df_rest['DOMISILI'].unique(), default = None)
all_dom = st.sidebar.checkbox("Pilih Semua Domisili", value = True)
if all_dom:
    dom = df_rest['DOMISILI'].unique()
slide = st.sidebar.slider('Filter IPK', 0.00, 4.00, (3.00, 4.00))
age = st.sidebar.slider('Filter Usia', 20, 30, (20, 30))

df_rest = df_rest[df_rest['PERFORMANCE LEVEL'].isin(tier) &
                  df_rest['JURUSAN'].isin(major) &
                  df_rest['UNIVERSITAS'].isin(univ) &
                  df_rest['DOMISILI'].isin(dom) &
                  df_rest['IPK'].between(slide[0],slide[1]) &
                  df_rest['USIA'].between(age[0],age[1])]

st.write('Data Calon Pegawai')
st.dataframe(df_rest)

col1, col2, col3 = st.columns(3)

with col1:
    st.write('total')
    dfz = df_rest.groupby(['UNIVERSITAS']).size().reset_index(name='TOTAL')
    st.dataframe(dfz)

with col2:
    st.write('total')
    dfzz = df_rest.groupby(['DOMISILI']).size().reset_index(name='TOTAL')
    st.dataframe(dfzz)

with col3:
    st.write('\n')
    st.write('\n')
    try:
        st.write("Rata-rata usia: ", str(round(sum(df_rest["USIA"])/len(df_rest), 2)), "Tahun")
        st.write("Rata-rata IPK: ", str(round(sum(df_rest["IPK"])/len(df_rest), 2)))
    except:
        st.write('Tidak ada data yang ditampilkan')

dfy = df_test.groupby(['DOMISILI','PERFORMANCE LEVEL']).size().reset_index(name='TOTAL')
dfy = dfy.pivot(index='DOMISILI', columns='PERFORMANCE LEVEL', values='TOTAL')
dfy = dfy.fillna(0)
dfy = dfy.rename_axis(None, axis=1).reset_index()
dfy = dfy.astype({'TIER 1': 'int64'})
dfy = dfy.astype({'TIER 2': 'int64'})
dfy = dfy.astype({'TIER 3': 'int64'})
dfy = dfy.astype({'TIER 4': 'int64'})
dfy = dfy.astype({'TIER 5': 'int64'})
# st.write('Domisili')
# st.dataframe(dfy)
dom = st.multiselect('filter data', dfy['DOMISILI'].unique(), default = dfy['DOMISILI'].unique())
dfy = dfy[dfy['DOMISILI'].isin(dom)]
fig = px.bar(dfy, x="DOMISILI", y=["TIER 1", "TIER 2", "TIER 3", "TIER 4", "TIER 5"], title="Domisili")
fig.update_layout(
    autosize=False,
    width=1000,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
st.plotly_chart(fig)

dfy = df_test.groupby(['UNIVERSITAS','PERFORMANCE LEVEL']).size().reset_index(name='TOTAL')
dfy = dfy.pivot(index='UNIVERSITAS', columns='PERFORMANCE LEVEL', values='TOTAL')
dfy = dfy.fillna(0)
dfy = dfy.rename_axis(None, axis=1).reset_index()
dfy = dfy.astype({'TIER 1': 'int64'})
dfy = dfy.astype({'TIER 2': 'int64'})
dfy = dfy.astype({'TIER 3': 'int64'})
dfy = dfy.astype({'TIER 4': 'int64'})
dfy = dfy.astype({'TIER 5': 'int64'})
st.write('Universitas')
dom = st.multiselect('filter data', dfy['UNIVERSITAS'].unique(), default = ['UNIVERSITAS TELKOM',
                                                                            'UNIVERSITAS SUMATERA UTARA',
                                                                            'UNIVERSITAS SRIWIJAYA',
                                                                            'UNIVERSITAS SEBELAS MARET',
                                                                            'UNIVERSITAS PADJADJARAN',
                                                                            'UNIVERSITAS INDONESIA',
                                                                            'UNIVERSITAS GADJAH MADA',
                                                                            'UNIVERSITAS DIPONEGORO',
                                                                            'UNIVERSITAS BRAWIJAYA',
                                                                            'UNIVERSITAS ANDALAS',
                                                                            'UNIVERSITAS AIRLANGGA',
                                                                            'INSTITUT TEKNOLOGI SEPULUH NOPEMBER',
                                                                            'INSTITUT TEKNOLOGI BANDUNG',
                                                                            'INSTITUT PERTANIAN BOGOR'])
all_u = st.checkbox("Pilih Semua Universitas")
if all_u:
    dom = dfy['UNIVERSITAS'].unique()
dfy = dfy[dfy['UNIVERSITAS'].isin(dom)]
fig = px.bar(dfy, y="UNIVERSITAS", x=["TIER 1", "TIER 2", "TIER 3", "TIER 4", "TIER 5"], title="Domisili")
fig.update_layout(
    autosize=False,
    width=1430,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
st.plotly_chart(fig)

dfy = df_test.groupby(['JURUSAN','PERFORMANCE LEVEL']).size().reset_index(name='TOTAL')
dfy = dfy.pivot(index='JURUSAN', columns='PERFORMANCE LEVEL', values='TOTAL')
dfy = dfy.fillna(0)
dfy = dfy.rename_axis(None, axis=1).reset_index()
dfy = dfy.astype({'TIER 1': 'int64'})
dfy = dfy.astype({'TIER 2': 'int64'})
dfy = dfy.astype({'TIER 3': 'int64'})
dfy = dfy.astype({'TIER 4': 'int64'})
dfy = dfy.astype({'TIER 5': 'int64'})
# st.write('Domisili')
# st.dataframe(dfy)
dom = st.multiselect('filter data', dfy['JURUSAN'].unique(), default = ['AKUNTANSI',
                                                                        'HUKUM',
                                                                        'MANAJEMEN',
                                                                        'TEKNOLOGI INFORMASI DAN INFORMATIKA'])
dfy = dfy[dfy['JURUSAN'].isin(dom)]
fig = px.bar(dfy, y="JURUSAN", x=["TIER 1", "TIER 2", "TIER 3", "TIER 4", "TIER 5"], title="Jurusan")
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='#ffffff',
    autosize=False,
    width=1430,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=4
    )
)
# fig.update_xaxes(
#     gridcolor = '#000000',
#     linecolor = '#000000')
# fig.update_yaxes(
#     linecolor = '#000000')
st.plotly_chart(fig)


# dom = st.multiselect('filter data', dfy['UNIVERSITAS'].unique(), default = None)
# dfy = dfy[dfy['UNIVERSITAS'].isin(dom)]
# fig = px.bar(dfy, x="UNIVERSITAS", y=["TIER 1", "TIER 2", "TIER 3", "TIER 4", "TIER 5"], title="Universitas")
# st.plotly_chart(fig)











#     else:
#         st.sidebar.error("Invalid username or password")

# try:
#     container.write("")
#     image = Image.open('D:\BNI_Hactics_Horizontal-removebg-preview.png')

#     st.image(image)

# except:
#     pass



