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
from scripts import visualize
import time

path = os.path.dirname(__file__)
ROOT_DIR = path+'/images'

img_data = [
    join(ROOT_DIR, 'select_images/S.jpg'),
    join(ROOT_DIR, 'select_images/E.jpg'),
    join(ROOT_DIR, 'select_images/F.jpg'),
    join(ROOT_DIR, 'select_images/K.jpg'),
    join(ROOT_DIR, 'select_images/L.jpg'),
    join(ROOT_DIR, 'select_images/R.jpg')
]

global imgs 
imgs = [open(i, 'rb').read() for i in img_data]
print("imgs type: ", type(imgs))

global model_paths
model_paths = ["../configs/VOC/demo_config_VOC.yaml", 
               "../configs/COCO/demo_config_COCO.yaml"]

show_rec = False

@st.cache(allow_output_mutation=True)
def setup_models(model_paths):
    configs = []
    for path in model_paths:
        configs.append(visualize.setup(path))
    return configs

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
        bcol2.image(path+"/images/image_vs_instance.png", width=550)


    # theory on UniT paper
    st.markdown(markdowns.unit_title, unsafe_allow_html=True)
    with st.expander("See more"):
        st.markdown(markdowns.unit_string, unsafe_allow_html=True)
        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        bcol2.image(path+"/images/unit.png", width=550)

    # select image of provided images

    #st.image(imgs, width=250, caption=[1,2,3,4,5,6])

    model_configs = setup_models(model_paths)
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
    model_options = ("None", "VOC", "COCO")
    st.write(markdowns.selectbox_styles, unsafe_allow_html=True)
    selected_model = st.selectbox("Model options", model_options)

    model_cfg = None
    
    if selected_model == "VOC":
        model_cfg = model_configs[0]

    elif selected_model == "COCO":
        model_cfg = model_configs[1]

    # RUN!
    _, col2, _ = st.columns([1, 1, 1])
    st.markdown(markdowns.button_style, unsafe_allow_html=True)

    global show_rec
    if col2.button("Run Model"):
        if len(selected_model) == 0 or selected_model == "None":
            st.markdown(markdowns.choose_model_again_line, unsafe_allow_html=True)
            return
        show_rec = True
        
        result_images = []
        xml_results = []

        with st.spinner(text="Hold tight! It's running..."):
            for image in images_to_run:
                result_image, xml_result = visualize.main(model_cfg, image)
                result_images.append(result_image)
                xml_results.append(xml_result)

    if show_rec:
        if len(np.where(selected)[0]) != 0:
            st.markdown(markdowns.results_line, unsafe_allow_html=True)
            for i in range(len(images_to_run)):
                i1, i2 = st.columns([1, 1])
                i1.image(open(images_to_run[i], 'rb').read(), width=400)
                i2.image(result_images[i], width=400)
        else:
            st.markdown(markdowns.try_again_line, unsafe_allow_html=True)


        _, col2, _ = st.columns([1, 1, 1])
        with col2:
            st.markdown(markdowns.button_style, unsafe_allow_html=True)
            st.download_button(
                label = 'Download results', 
                data = xml_result, 
                file_name = "unit_demo_results.xml", 
                mime='application/xml')

    st.text("")

    st.markdown(markdowns.learn_more_header, unsafe_allow_html=True)
    with st.expander("See more"):
        st.write("Check out the UniT paper [here](https://arxiv.org/pdf/2006.07502.pdf), or try the [code](https://github.com/ubc-vision/UniT) yourself!")
        st.write("For more info on what the UBC Computer Vision group is working on, check out our [website](https://vision.cs.ubc.ca/).")

if __name__ == "__main__":
    main()