import streamlit as st
from PIL import Image
import base64
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.graph_objs import Layout

image = Image.open('D:\logo-bni.png')

st.set_page_config(
     page_title="BNI-HACTICS",
     page_icon=image,
     layout="wide",
     initial_sidebar_state="expanded",
 )

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def header(url):
     st.markdown(f'<p style="background-color:#FC6608;color:#000000;font-size:60px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

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

set_bg_hack('D:/background1.jpg')

#def load_lottieurl(url):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()
#
#st.markdown("<h1 style='text-align: center; color: white;'>HC Analytics</h1>", unsafe_allow_html=True)

    #lottie_analytic = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_dews3j6m.json")
    
container = st.container()

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type = 'password')

if st.sidebar.checkbox("Login"):
    #st.balloons()

    if username == 'user' and password == 'bni46':
        header("Talent Acquisition")

        del container
        st.sidebar.success("Logged in as {}".format(username))

        dfx = pd.read_excel('D:/data_train.xlsx', engine='openpyxl', index_col=0)

        st.header('Data ODP saat ini')
        
        #Universitas
        _, col2, _ = st.columns([2, 2, 1])

        with col2:
            st.write('Top 10 Univerisity')
        
        dfp2 = dfx["UNIVERSITAS"].value_counts().rename_axis('UNIVERSITAS').reset_index(name='Total')
        dfp2 = dfp2.sort_values('Total',ascending = False).head(10)
        labels2 = dfp2["UNIVERSITAS"]
        values2 = dfp2["Total"]

        layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

        fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)], layout =layout)
        fig2.update_traces(textinfo='value')
        st.plotly_chart(fig2, use_container_width = True)

        st.markdown("""---""")
        #Jurusan
        _, col3, _ = st.columns([2, 2, 1])
        with col3:
            st.write('Top 10 Major')
        dfp2 = dfx["JURUSAN"].value_counts().rename_axis('JURUSAN').reset_index(name='Total')
        dfp2 = dfp2.sort_values('Total',ascending = False).head(10)        
        labels2 = dfp2["JURUSAN"]
        values2 = dfp2["Total"]

        fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)], layout =layout)
        fig2.update_traces(textinfo='value')
        st.plotly_chart(fig2, use_container_width = True)

        st.markdown("""---""")
        #Usia
        st.write("Rata-rata usia: ", str(round(sum(dfx["USIA"])/len(dfx), 2)), "Tahun")

        #IPK
        st.write("Rata-rata IPK: ", str(round(sum(dfx["IPK"])/len(dfx), 2)))

        st.write(":heavy_minus_sign:" * 58)
        #upload
        st.header('Prediksi tingkat kinerja calon ODP')
        uploaded_file = st.file_uploader(label="Unggah file calon ODP disini", type=['xlsx'])


        if uploaded_file is not None:
            try:
                df_test = pd.read_excel(uploaded_file, engine='openpyxl', index_col=0)
                df_rest = df_test
                df_rest = df_rest.reset_index(drop = True)
            except:
                pass
            
            df_test = df_test.drop(columns=["NAMA"])

            from sklearn.model_selection import train_test_split

            df = pd.read_excel('D:/data_train_1.xlsx', engine='openpyxl', index_col=0)

            X = df.drop(columns=["PERFORMANCE LEVEL"])

            y = df["PERFORMANCE LEVEL"]

            #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=1)

            from sklearn import preprocessing
                
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            #y_test_tr = lab.fit_transform(y_test)

            from sklearn import ensemble
            gb_clf = ensemble.GradientBoostingClassifier(n_estimators=50)
            gb_clf.fit(X,y_transformed)

            #rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
            #rf_clf.fit(X, y_transformed)

            a1 = pd.DataFrame(X.columns, columns = ['colname'])

            dum1 = pd.get_dummies(df_test['DOMISILI'])
            dum2 = pd.get_dummies(df_test['TINGKAT PENDIDIKAN'])
            dum3 = pd.get_dummies(df_test['UNIVERSITAS'])
            dum4 = pd.get_dummies(df_test['JURUSAN'])

            df_test = pd.concat((df_test, dum1, dum2, dum3, dum4), axis=1)

            df_test = df_test.drop(['DOMISILI', 'TINGKAT PENDIDIKAN', 'UNIVERSITAS', 'JURUSAN'], axis = 1)

            b = pd.DataFrame(df_test.columns, columns = ['colname'])
               
            b2 = a1[~a1.colname.isin(b.colname)]
            b2 = b2.reset_index(drop = True)

            for i in range (len(b2)):
                df_test[b2['colname'][i]] = 0

            df_test = df_test.reset_index(drop = True)

            result = gb_clf.predict(df_test)

            df_result = pd.DataFrame(result, columns = ['PERFORMANCE LEVEL'])

            df_result = df_result.replace({0: 'TIER 1', 1: 'TIER 2', 2: 'TIER 3', 3: 'TIER 4', 4: 'TIER 5'})

            df_rest = pd.concat((df_result, df_rest), axis=1)
            
            df_rest = df_rest.sort_values('PERFORMANCE LEVEL',ascending = True)

            st.dataframe(df_rest)

    else:
        st.sidebar.error("Invalid username or password")

try:
    container.write("")
    image = Image.open('D:\BNI_Hactics_Horizontal-removebg-preview.png')

    st.image(image)
    #st_lottie(lottie_analytic, height=300, key="analytic")
except:
    pass
