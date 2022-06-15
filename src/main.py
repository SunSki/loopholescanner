#####################
### Set constants ###
#####################
DIGITAL_0 = 'digital_scattered'
CAPTURED_0 = 'captured_scattered'
SAMPLE_PHOTO_PATH = '../data/img/test/18cards_scattered.jpg'
PAGE_ICON_PATH = '../data/img/loopholesLogo.jpg'
LOGO_PATH = '../data/img/loopholeScannerLogo.jpg'
HERO_IMG_PATH = '../data/img/mainHero.jpg'
LECTURE_IMG_PATH = '../data/img/goodBadPhoto.jpg'
IMG_SAVE_FOLDER = '../data/img/upload'

#####################
### Set streamlit ###
#####################
import streamlit as st

st.set_page_config(
     page_title="loopholeScanner",
     page_icon=PAGE_ICON_PATH,
     layout="wide",
     initial_sidebar_state="expanded",
 )
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    /*header {display: none !important;}*/
    footer {display: none !important;}
    .block-container {padding-bottom:2rem; padding-top:3rem;}
    .title h1 {margin-top: 0; padding-top:0; margin-bottom: 1rem;}
    .title p {margin-bottom: 0; font-family: "Source Code Pro", monospace;}
    div.stButton > button:first-child {height: 3em; width: 9em;}
    </style>
    """, 
    unsafe_allow_html=True
)


from PIL import Image
import numpy as np
from _utils import saveNpImg
from glob import glob

import loopholeScanner



def saveUploadImage(img_np):
    save_img_num = len(glob(f'{IMG_SAVE_FOLDER}/*.jpg'))
    file_name = f"{IMG_SAVE_FOLDER}/upload_{save_img_num}.jpg"
    saveNpImg(img_np, file_name)


def generateDigitalBoard(img_bytes, img_np):
    try:
        digital_boards = loopholeScanner.generate(img_bytes, img_np)
        saveUploadImage(img_np)
        st.session_state.digitalBoards[DIGITAL_0] = digital_boards[DIGITAL_0]
        st.session_state.digitalBoards[CAPTURED_0] = digital_boards[CAPTURED_0]
        st.session_state.upload_state = 1
        
    except Exception as e:
        print('=== Error ===')
        print(f'type: {type(e)}')
        print(f'args: {e.args}')
        print(f'erorr: {e}')

        if e.args[0] == "Board not found":
            st.session_state.upload_state = 3
        else:
            st.session_state.upload_state = 2


def main():
    #################
    ### Set State ###
    #################
    if 'upload_state' not in st.session_state:
        st.session_state.upload_state = 0

    if 'digitalBoards' not in st.session_state:
        st.session_state.digitalBoards = {
            DIGITAL_0:None, 
            CAPTURED_0:None
        }

    if 'uploaded_img_bytes' not in st.session_state:
        st.session_state.uploaded_img_bytes = None

    if 'uploaded_img_Image' not in st.session_state:
        st.session_state.uploaded_img_Image = None
        
    if 'uploaded_img_np' not in st.session_state:
        st.session_state.uploaded_img_np = None


    ############
    ### main ###
    ############

    # Hero
    st.markdown('''
        <div class="title">
            <p>🪄Captured Board into Digital</p>
            <h1>LoopholeScanner</h1>
        </div>
        ''', unsafe_allow_html=True
    )
    st.image(HERO_IMG_PATH)
    st.image(LOGO_PATH)
    st.subheader('About LoopholeScanner')
    st.markdown('''
        This app can convert a taken photo of LOOPHOLES board into digital after playing. <br>
        Save or share it!
        ''', unsafe_allow_html=True
    )


    # Capture
    st.header('1. 🤳Take a Photo or Upload')

    uploaded_file = st.file_uploader(
        "👇 Tap box to take a photo", 
        type=["png", "jpg", "jpeg"])
    
    st.image(LECTURE_IMG_PATH)

    st.text('There is not any board? Then,')
    with open(SAMPLE_PHOTO_PATH, "rb") as file:
        btn = st.download_button(
                label="👋 Download sample photo",
                data=file,
                file_name="loopholescanner_sample.jpg",
                mime="image/jpg"
            )

    if uploaded_file is not None:
        st.session_state.uploaded_img_bytes = uploaded_file.read()
        st.session_state.uploaded_img_Image = Image.open(uploaded_file)
        st.session_state.uploaded_img_np = np.array(st.session_state.uploaded_img_Image)

        #st.markdown('<h3 style="text-align: right;">Push "x" to retake ☝️ . </h3>', unsafe_allow_html=True)

        st.header('2. Check taken photo 👀')
        st.image(
            st.session_state.uploaded_img_Image,
            # use_column_width=True
        )
        st.text('Q. Is it taken from above?')
        st.text('Q. Is the whole board in?')

        st.header('3. 🪄Start to Scan')
        if st.button('✨Scan Photo'):
            with st.spinner('Wait for Scanning cards...'):
                generateDigitalBoard(
                st.session_state.uploaded_img_bytes,
                st.session_state.uploaded_img_np)

        # uploaded
        if st.session_state.upload_state == 1:
            st.success('Well done!')

            st.header('4. 👇 Choose type')

            result_type = st.radio(
                "Digital or Captured",
                ('Digital Cards', 'Captured Cards')
            )

            if result_type == 'Digital Cards':
                st.image(
                    st.session_state.digitalBoards[DIGITAL_0],
                    # use_column_width=True
                )
            elif result_type == 'Captured Cards':
                st.image(
                    st.session_state.digitalBoards[CAPTURED_0],
                    # use_column_width=True
                )
            st.subheader('☝️ Tap and hold to save or share')
            
        # Error: General
        if st.session_state.upload_state == 2:
            st.error('Scanning Error: Take another photo')
        
        # Error: Board not found
        if st.session_state.upload_state == 3:
            st.error('Scanning Error: Board was not detected. Board must be included.')
        
    else:
        st.session_state.upload_state = 0
        st.session_state.digitalBoards[DIGITAL_0] = None
        st.session_state.digitalBoards[CAPTURED_0] = None
    
    st.markdown('''
        <p style="text-align: center; padding-top: 4rem">Copyright (c) 2022 LOOPHOLES All rights reserved.</p>
        ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()