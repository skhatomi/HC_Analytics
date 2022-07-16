import streamlit as st
from multiapp import MultiApp
from apps import MachineLearning, Scoring # import your app modules here
from PIL import Image
import base64

# logo tab
image = Image.open('./image/logo-bni.png', mode='r')

st.set_page_config(
     page_title="HC-Analytics",
     page_icon=image,
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

set_bg_hack('./image/background1.jpg')

# hide streamlit label
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

            # progress = st.progress(0)
            # from time import sleep
            # for i in range(100):
            #     progress.progress(i)
            #     sleep(0.1)
            # del progress
            del container
            st.sidebar.success("Logged in as {}".format(username))
            app = MultiApp()
            
            app.add_app("Machine Learning", MachineLearning.app)
            app.add_app("Scoring System", Scoring.app)

            app.run()
        else:
            st.sidebar.info("Invalid username or password")

    try:
        container.write("")
        image = Image.open('./image/BNI_Hactics_Horizontal-removebg-preview.png')

        st.image(image)
    except:
        pass

if __name__ == '__main__':
    main()

# [theme]
# base="light"
# primaryColor="#fc6608"
# secondaryBackgroundColor="#04649c"
# textColor="#ffffff"



