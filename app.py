import streamlit as st
import requests
from multiapp import MultiApp
from apps import EmpSel, OveAna, PosAna, EmpAna, talacq # import your app modules here
from streamlit_lottie import st_lottie
from PIL import Image
import base64

st.set_page_config(
     page_title="HC-Analytics",
     page_icon=":bar_chart:",
     layout="wide",
     initial_sidebar_state="expanded",
 )

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

def main():
    #lottie_analytic = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_dews3j6m.json")
    
    container = st.container()

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type = 'password')

    if st.sidebar.checkbox("Login"):
        if password == '':
            del container
            st.sidebar.success("Logged in as {}".format(username))
            app = MultiApp()
            
            app.add_app("Overall Analytics", OveAna.app)
            app.add_app("Position Analytics", PosAna.app)
            app.add_app("Employee Analytics", EmpAna.app)
            app.add_app("Employee Selector", EmpSel.app)
            app.add_app("Talent Acquisition", talacq.app)

            app.run()
        else:
            st.sidebar.warning("Invalid username or password")

    try:
        container.write("")
        image = Image.open('D:\BNI_Hactics_Horizontal-removebg-preview.png')

        st.image(image)
        #st_lottie(lottie_analytic, height=300, key="analytic")
    except:
        pass

if __name__ == '__main__':
    main()



