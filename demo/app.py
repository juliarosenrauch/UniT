import os

import torch
import numpy as np
import streamlit as st

import markdowns
import visualize

filepath = os.path.dirname(__file__)
ROOT_DIR = os.path.join(filepath, 'images')

img_data = [
    os.path.join(ROOT_DIR, 'select_images', 'S.jpg'),
    os.path.join(ROOT_DIR, 'select_images', 'E.JPG'),
    os.path.join(ROOT_DIR, 'select_images', 'F.jpg'),
    os.path.join(ROOT_DIR, 'select_images', 'K.jpg'),
    os.path.join(ROOT_DIR, 'select_images', 'L.jpg'),
    os.path.join(ROOT_DIR, 'select_images', 'R.jpg')
]

global imgs
imgs = [open(i, 'rb').read() for i in img_data]

global model_paths
model_paths = [os.path.join(filepath, '..', 'configs', 'VOC', 'demo_config_VOC.yaml'),
                os.path.join(filepath, '..', 'configs', 'COCO', 'demo_config_COCO.yaml')]

show_rec = False


def print_resources(memo):
    print(memo)
    memory_free = os.system('cat /proc/meminfo | grep MemFree')
    print(memory_free)

dataAvailable = False
if not dataAvailable:
    print("unzipping data")
    os.system('cat best_model_final_weights.zip* > ~/best_model_final_weights.zip')
    os.system('unzip best_model_final_weights.zip')

    os.system('cat VOC_split1.zip* > ~/VOC_split1.zip')
    os.system('unzip VOC_split1.zip')
    dataAvailable = True
    print_resources("unzipping")


@st.cache(allow_output_mutation=True)
def setup_models(model_paths):
    print_resources("setup model configs")
    configs = []
    for path in model_paths:
        configs.append(visualize.setup(path))
    return configs

def main():
    st.set_page_config(page_title="UniT Model Demo",
                        page_icon=open(os.path.join(ROOT_DIR, 'select_images', 'cvlab.png'), 'rb').read(),
                        layout="wide")

    st.markdown(markdowns.reconstruction_style, unsafe_allow_html=True)
    st.markdown(markdowns.hide_decoration_bar_style, unsafe_allow_html=True)
    st.markdown(markdowns.hide_main_menu, unsafe_allow_html=True)
    st.markdown(markdowns.page_title, unsafe_allow_html=True)
    st.markdown(markdowns.cv_group_title, unsafe_allow_html=True)

    st.markdown(markdowns.motivation_title, unsafe_allow_html=True)
    with st.expander("See more"):
        st.markdown(markdowns.motivation_string, unsafe_allow_html=True)
        _, bcol2, _ = st.columns([1, 1, 1])
        bcol2.image(open(os.path.join(ROOT_DIR, 'select_images', 'image_vs_instance.png'), 'rb').read(), use_column_width=True)

    st.markdown(markdowns.unit_title, unsafe_allow_html=True)
    with st.expander("See more"):
        st.markdown(markdowns.unit_string, unsafe_allow_html=True)
        st.write("For more details, check out the paper [here](https://arxiv.org/pdf/2006.07502.pdf).")
        _, bcol2, _ = st.columns([1, 1, 1])
        bcol2.image(open(os.path.join(ROOT_DIR, 'select_images' , 'unit.png'), 'rb').read(), use_column_width=True)

    print_resources("pre setup models")
    model_configs = setup_models(model_paths)
    print_resources("post setup models")

    st.markdown(markdowns.object_selection_header, unsafe_allow_html=True)

    image_columns = st.columns([1,1,1,1,1,1])
    for i, col in enumerate(image_columns):
        col.image(imgs[i], use_column_width=True)

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
        uploaded_file_path = os.path.join(ROOT_DIR, uploaded_file.name)
        with open(uploaded_file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        images_to_run.append(uploaded_file_path)

        _, col_uploaded, _ = st.columns([2, 1, 2])
        col_uploaded.image(bytes_data,caption=uploaded_file.name,use_column_width=True)

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
    _, col_button, _ = st.columns([3, 1, 3])
    st.markdown(markdowns.button_style, unsafe_allow_html=True)

    global show_rec
    if col_button.button("Run Model"):
        if len(selected_model) == 0 or selected_model == "None":
            # st.markdown(markdowns.choose_model_again_line, unsafe_allow_html=True)
            st.error("Please choose a model and run again!")
            return

        if len(np.where(selected)[0]) != 0:
            show_rec = True
        else:
            # st.markdown(markdowns.try_again_line, unsafe_allow_html=True)
            st.error("Please select or upload an image above and run again!")

        result_images = []
        xml_results = []

        with st.spinner(text="Hold tight! It's running..."):
            for image in images_to_run:
                print_resources("pre visualize")
                result_image, xml_result = visualize.main(model_cfg, image)
                print_resources("post visualize")
                result_images.append(result_image)
                xml_results.append(xml_result)

    if show_rec:
        if len(np.where(selected)[0]) != 0:
            st.markdown(markdowns.results_line, unsafe_allow_html=True)
            for i in range(len(images_to_run)):
                i1, i2, i3, i4, i5 = st.columns([1,1,1,1,1])
                i2.image(open(images_to_run[i], 'rb').read(), use_column_width=True)
                i4.image(result_images[i], use_column_width=True)
        else:
            # st.markdown(markdowns.try_again_line, unsafe_allow_html=True)
            st.error("Please select or upload an image above and run again!")


        _, col_dl, _ = st.columns([4, 1, 4])
        with col_dl:
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
