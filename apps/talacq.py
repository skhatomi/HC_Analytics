import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

def app():
    
    dfx = pd.read_excel('D:/data_train.xlsx', index_col=0)

    #PerformanceScore
    dfp2 = dfx["UNIVERSITAS"].value_counts().rename_axis('UNIVERSITAS').reset_index(name='Total')
    labels2 = dfp2["UNIVERSITAS"]
    values2 = dfp2["Total"]

    fig2 = go.Figure(data=[go.Pie(labels=labels2, values=values2, hole=.3)])
    st.plotly_chart(fig2, use_container_width = True)




    #upload
    uploaded_file = st.file_uploader(label="Upload", type=['xlsx'])

    if uploaded_file is not None:
        try:
            df_test = pd.read_excel(uploaded_file, index_col=0)
            df_rest = df_test
            df_rest = df_rest.reset_index(drop = True)
        except:
            pass
    
        df_test = df_test.drop(columns=["NAMA"])

        from sklearn.model_selection import train_test_split

        df = pd.read_excel('D:/data_train_1.xlsx', index_col=0)

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

        st.dataframe(df_rest)


