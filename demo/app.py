from http.client import UnknownTransferEncoding
import os
from os.path import join

import streamlit as st
import markdowns

import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
from scripts import run_image

path = os.path.dirname(__file__)
ROOT_DIR = path+'\images'

img_data = [
    join(ROOT_DIR, r'select_images\bus.jpg'),
    join(ROOT_DIR, r'select_images\couch.jpg'),
    join(ROOT_DIR, r'select_images\fruits.jpg'),
    join(ROOT_DIR, r'select_images\sheep.jpg'),
    join(ROOT_DIR, r'select_images\tennis.jpg'),
    join(ROOT_DIR, r'select_images\umbrella.jpg')
]

global imgs 
imgs = [open(i, 'rb').read() for i in img_data]
print("imgs type: ", type(imgs))

loaded_images = {
    "bus": imgs[0],
    "couch": imgs[1],
    "fruits": imgs[2],
    "sheep": imgs[3],
    "tennis": imgs[4],
    "umbrella": imgs[5],
}

show_rec = False

def main():
    st.set_page_config(page_title="UniT Model Demo",
                        page_icon=path+"/images/cvlab.png",
                        layout="wide")
    
    st.markdown(markdowns.reconstruction_style, unsafe_allow_html=True)

    #st.markdown(markdowns.remove_padding, unsafe_allow_html=True)

    st.markdown(markdowns.hide_decoration_bar_style, unsafe_allow_html=True)
    
    st.markdown(markdowns.hide_main_menu, unsafe_allow_html=True)

    st.markdown(markdowns.page_title, unsafe_allow_html=True)
    st.markdown(markdowns.cv_group_title, unsafe_allow_html=True)
    
    # background on image level vs instance level annotation
    # wouldn't it be nice to be able to transfer knowledge from base classes to novel!
    st.markdown(markdowns.motivation_title, unsafe_allow_html=True)
    with st.expander("See more"):
        st.markdown(markdowns.motivation_string, unsafe_allow_html=True)
        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        bcol2.image(path+r'\images\image_vs_instance.png', width=550)

    # intro to UBC computer vision lab


    # theory on UniT paper
    st.markdown(markdowns.unit_title, unsafe_allow_html=True)
    with st.expander("See more"):
        st.markdown(markdowns.unit_string, unsafe_allow_html=True)
        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        bcol2.image(path+r'\images\unit.png', width=550)

    # select image of provided images

    # st.markdown(markdowns.object_selection_header, unsafe_allow_html=True)
    # radio_selection_col, image_selection_col, col3 = st.columns([2, 1, 1])
    # col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
    # with radio_selection_col:
    #     st.write(markdowns.radio_selection_styles, unsafe_allow_html=True)
    #     selected_image = st.radio("", list(loaded_images.keys()))

    # with image_selection_col:
    #     st.image(loaded_images[selected_image],
    #             caption='Selected Object',
    #             width=350)

    #st.image(imgs, width=250, caption=[1,2,3,4,5,6])


    st.markdown(markdowns.object_selection_header, unsafe_allow_html=True)

    image_columns = st.columns([1,1,1,1,1,1])
    for i, col in enumerate(image_columns):
        col.image(imgs[i], width=200)

    selection_columns = st.columns([1,1,1,1,1,1])
    checkboxkey = ['1','2','3','4','5','6']
    selected = [False]*6
    for i, col in enumerate(selection_columns): 
        selected[i] = col.checkbox(checkboxkey[i])

    images_to_run = []
    if len(np.where(selected)[0]) != 0:
        print("selected stuff: ", np.where(selected)[0])
        for index in np.where(selected)[0]:
            images_to_run.append(img_data[index])
    
    st.markdown(markdowns.upload_file_header, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['png','jpeg','jpg'])
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        uploaded_file_path = os.path.join(ROOT_DIR,uploaded_file.name)
        with open(uploaded_file_path,"wb") as f: 
            f.write(uploaded_file.getbuffer()) 

        images_to_run.append(uploaded_file_path)

        st.image(bytes_data,
                 caption=uploaded_file.name,
                 width=350)

    st.markdown(markdowns.choose_model_header, unsafe_allow_html=True)
    model_options = ("None", "Pre-trained A", "Pre-trained B")
    st.write(markdowns.selectbox_styles, unsafe_allow_html=True)
    selected_model = st.selectbox("Model options", model_options)

    model_path = ""
    if selected_model == "Pre-trained A":
        model_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    elif selected_model == "Pre-trained B":
        model_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    # if len(selected_model) == 0 or selected_model == "None":
    #     return
        # background on model

    # RUN!
    col1, col2, col3 = st.columns([1, 1, 1])
    st.markdown(markdowns.button_style, unsafe_allow_html=True)


    global show_rec
    if col2.button("Run Model"):
        if len(selected_model) == 0 or selected_model == "None":
            st.markdown(markdowns.choose_model_again_line, unsafe_allow_html=True)
            return
        show_rec = True
        # run the model here.. use selected_model
        result_images= run_image.test_model(model_path, images_to_run)

    if show_rec:
        if len(np.where(selected)[0]) != 0:
            st.markdown(markdowns.results_line, unsafe_allow_html=True)
            for i in range(len(images_to_run)):
                i1, i2 = st.columns([1, 1])
                i1.image(open(images_to_run[i], 'rb').read(), width=400)
                i2.image(result_images[i], width=400)
        else:
            st.markdown(markdowns.try_again_line, unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(markdowns.button_style, unsafe_allow_html=True)
        st.download_button(
            label = 'Download results', 
            data = '''text_contents''', 
            file_name = "unit_demo_results.xml", 
            mime='application/xml')

    st.text("")

    st.markdown(markdowns.learn_more_header, unsafe_allow_html=True)
    with st.expander("See more"):
        st.write("Check out the UniT paper [here](https://arxiv.org/pdf/2006.07502.pdf), or try the [code](https://github.com/ubc-vision/UniT) yourself!")
        st.write("For more info on what the UBC Computer Vision group is working on, check out our [website](https://vision.cs.ubc.ca/).")

if __name__ == "__main__":
    main()