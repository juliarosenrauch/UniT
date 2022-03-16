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

motivation_string = """<p align="left">What is object detection? Object detection is a field of computer vision that deals with the localization and 
classification of objects contained in an image. Methods for object detection rely on a large amount of instance-level annotations for training, 
which are difficult and time-consuming to collect. These are annotations that define which pixels of an image belong to various classes of data 
(for example, sheep) and labels them as such. </p>

<p align="left">In the below image, we see two types of annotations. Image level annotations label an image simply as containing various classes of data and are 
much easier to obtain (think: Google search of a sheep = lots of image level sheep annotations). Instance level annotations, require much more 
time and effort to collect. In general, there are a lot of available image level annotations, and few instance level annotations.</p>

<h5 align="center">Wouldn't it be great if we could somehow use our abundance of image level annotations to train models for object detection?</h5>"""


unit_title = """<h3 align="left">Unified Knowledge Transfer for Any-shot Object Detection and Segmentation</h3>"""

unit_string = """<p align="left">This is an area that Dr. Leonid Sigal and his graduate students, Siddhesh Khandelwal and Raghav Goyal, at the University 
of British Columbia's Computer Vision Lab have been actively working on. In March 2021, they published "UniT: Unified Knowledge Transfer for Any-shot Object 
Detection and Segmentation", a paper describing a model that maps similarities from base classes - those with an abundance of both image and instance level data, 
to novel classes - those with image level data but only zero to a few samples of instance-level data.</p>

<p align="left">For base classes, the UniT model learns a mapping from weakly-supervised to fully-supervised detectors/segmentors. By learning and leveraging visual and lingual 
similarities between the novel and base classes, we transfer those mappings to obtain detectors/segmentors for novel classes; refining them with a few novel class 
instance-level annotated samples, if available. The overall model is end-to-end trainable and highly flexible.</p>

<p align="left">For example, we may have access to an abundance of instance-level segmentations (think: pixel labels) of sheep, a base class in this case, but few instance 
level segmentations of goats, a novel class. UniT would learn the visual and lingual similarities between goats and sheep and allow us to perform more accurate object detection.</p>"""

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
        margin: 0 auto;
        border: solid 1px gray;
        width: 200px;
        font-size: 22px;
        transition-duration: 0.2s
        position:relative;
        top:50%; 
        left:50%;
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

running_line = """<h5 align="center" Just a second! It's running..."""

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

