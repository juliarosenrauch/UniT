reconstruction_style = """
<style>
    .css-rncmk8 {
        text-align: center;
    }
    .css-1kyxreq {
        display: flex;
        flex-flow: column !important;
        align-items: center;
    }
</style>
"""

hide_decoration_bar_style = """
  <style>
    header {visibility: hidden;}
  </style>
"""

remove_padding = """ <style>
        .reportview-container .main .block-container{{
            padding-top: 0rem;
            padding-bottom: 0rem;
    }} </style> """

hide_main_menu = """ <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style> """

page_title = """<h1 align="center">UniT Demo</h1>"""
cv_group_title = """<h2 align="center">UBC Computer Vision Group</h2>"""

motivation_title = """<h3 align="left">Motivation</h3>"""

motivation_string = """<p align="left">Few instance level annotations, abundant image level annotations.</p>"""


unit_title = """<h3 align="left">Unified Knowledge Transfer for Any-shot Object Detection and Segmentation</h3>"""

unit_string = """<p align="left">Methods for object detection and segmentation rely on large scale instance-level annotations for training, which 
        are difficult and time-consuming to collect. Efforts to alleviate this look at varying degrees and quality of supervision. 
        Weakly-supervised approaches draw on image-level labels to build detectors/segmentors, while zero/few-shot methods assume abundant instance-level data for a set of base classes,
        and none to a few examples for novel classes. This taxonomy has largely siloed algorithmic designs. In this work, we aim
        to bridge this divide by proposing an intuitive and unified semi-supervised model that is applicable to a range of supervision: from zero to a few instance-level samples per novel
        class. For base classes, our model learns a mapping from weakly-supervised to fully-supervised detectors/segmentors.
        By learning and leveraging visual and lingual similarities between the novel and base classes, we transfer those mappings
        to obtain detectors/segmentors for novel classes; refining them with a few novel class instance-level annotated samples, if available. The overall model is end-to-end trainable
        and highly flexible.</p>"""

object_selection_header = """<h3 align="left" style="margin-top: 3rem;">To begin identifying objects, select an image</h3>"""

radio_selection_styles = """
<style>
    div.row-widget.stRadio > div {
        height: 100px;  
    }
    div.row-widget.stRadio > div > label {
        margin-bottom: 6px;
    }
</style>"""

upload_file_header = """<h3 align="left" style="margin-top: 0rem;">Or upload your own</h3>"""

choose_model_header = """<h3 align="left">Choose a model</h3>"""

selectbox_styles = """
<style>
    div.row-widget.stSelectbox > div {
        margin-bottom: 25px;
    }
</style>
"""

button_style = """
<style>
    div.stButton > button {
        padding: 0.5rem 0.5rem;
        background-color: #ffffff;
        color: black;
        margin: 60px 0px;
        border: solid 1px gray;
        width: 200px;
        font-size: 22px;
        transition-duration: 0.2s
    }
    button:hover {
        font-weight: 500;
        box-shadow: rgba(0, 0, 0, 0.2) 0px 10px 30px -15px;
        border-color: #2214c7;
        color: white;
        background-color: #ffffff;
    }
    button:active {
        background-color: #2214c7;
        color: white;
    }
}
</style>
"""
choose_model_again_line = """<h4 align="center" style="color:blue">Please choose a model and run again!</h4>"""

results_line = """<h4 align="center">Below is your image and the results from the model.</h4>"""

try_again_line = """<h4 align="center" style="color:blue">Please select or upload an image above and run again!</h4>"""

learn_more_header = """<h3 align="center">Learn more</h3>"""


references = """## References
* Bergmann, Paul, et al. "MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
* Baur, Christoph, et al. "Autoencoders for unsupervised anomaly segmentation in brain mr images: A comparative study." Medical Image Analysis (2021): 101952.
"""

acknowledgements = """## Thanks to everyone who has made this demo possible

**Computer Vision Project Participants and Vector Sponsors**

**Vector Institute Team**

"""

