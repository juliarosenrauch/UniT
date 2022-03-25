import numpy as np
import cv2
import json
import sys
import os
import inspect
import torch
from typing import Any, Dict

# sys.path.insert(0, '../../')
# sys.path.insert(0, '../')

# hack for importing local modules from parent directory:
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from configs import add_config
from dict2xml import dict2xml
from collections import OrderedDict
from xml.etree import ElementTree as ET

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.utils.visualizer import ColorMode

def setup(config_file):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

class visualizer_batch:
    def __init__(self, img):
        self.image = img

class VizPredictor:
    def __init__(self, cfg):
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        self.cfg = cfg.clone()
        self.viz_folder = os.path.join(cfg.OUTPUT_DIR, 'viz')
        os.makedirs(self.viz_folder, exist_ok=True)

        labels = []
        type = MetadataCatalog.get(cfg.DATASETS.FEWSHOT.TYPE)
        if type.get("name") == 'VOC':
            labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        elif type.get("name") == 'COCO':
            labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            print("LENGTH: ", len(labels))
        MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes = labels
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)


    def __call__(self, image_path):
        with inference_context(self.model), torch.no_grad():

            dataset_dict = {"file_name" :  image_path,
            "image" : None}

            image = utils.read_image(dataset_dict["file_name"], format='BGR')
            dataset_dict["height"] = image.shape[0]
            dataset_dict["width"] = image.shape[1]

            auginput = T.AugInput(image)
            transform = T.Resize((800, 800))(auginput)
            image = transform.apply_image(image)
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

            # print (dataset_dict["image"].shape)
            # print("image: ", dataset_dict["image"])
            prediction = self.model([dataset_dict])[0]

            # print(prediction)

            scores = prediction['instances'].scores
            score_mask = scores >= 0.70
            prediction["instances"] = prediction["instances"][score_mask]

            # prediction_dict = prediction["instances"].get_fields()
            # prediction_json = json.dumps(prediction_dict, indent = 4)

            img = cv2.imread(image_path)
            # img = cv2.resize(img, (800,800))
            v = Visualizer(img[:, :, ::-1],
                        MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                        scale=1.0,
                        instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(prediction["instances"].to("cpu"))
            # test_img = v.get_image()[:, :, ::-1]
            test_img = v.get_image()[:, :, :]
            cv2.imshow("result", test_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # out_file_name = "000140_processed.png"
            # cv2.imwrite(out_file_name, test_img)
            return test_img, prediction

class ModifiedInstances:
    def __init__(self, prediction):
        self: Dict[str, Any] = {}
        self["num_instances"] = len(prediction)
        # self.image_height = prediction.image_height
        # self.image_width = prediction.image_width
        fields: Dict[str, Any] = {}
        # for k, v in prediction['instances'].get_fields().items():
        fields["pred_boxes"] = prediction['instances'].pred_boxes.tensor.tolist()
        fields["scores"] = prediction['instances'].scores.tolist()
        fields["pred_classes"] = prediction['instances'].pred_classes.tolist()
        # fields["pred_masks"] = prediction['instances'].pred_masks.tolist()
        self["fields"] = fields
        print(self)

    def convert_to_json(self):
        json_instances = json.dumps(self.__dict__)
        return json_instances

    def convert_to_xml(self):
        xml_instances = dict2xml(self)
        # print("CONVERT: ", xml_instances)
        return xml_instances

def convert_instances_to_xml(prediction):
    dict: OrderedDict[str, Any] = {}
    dict["num_instances"] = len(prediction)
    dict["image_height"] = prediction["instances"].image_size[0]
    dict["image_width"] = prediction["instances"].image_size[1]

    fields: OrderedDict[str, Any] = {}

    fields["pred_boxes"] = prediction['instances'].pred_boxes.tensor.tolist()
    fields["scores"] = prediction['instances'].scores.tolist()
    fields["pred_classes"] = prediction['instances'].pred_classes.tolist()
    fields["pred_masks"] = prediction['instances'].pred_masks.numpy().tolist()

    # print("SHAPE: ", prediction['instances'].pred_masks.numpy().shape)

    # print(len(fields["pred_masks"]))
    # print(len(fields["pred_masks"][0]))
    # print(len(fields["pred_masks"][0][0]))
    dict["fields"] = fields
    # print(dict)

    xml = dict2xml(dict)
    # print("CONVERT: ", xml)
    return xml

    # root = ET.Element("instance")
    # ET.SubElement(root, "num_instances").text = len(prediction)
    # fields = ET.SubElement(root, "fields")
    # ET.SubElement(fields, "pred_boxes").text = prediction['instances'].pred_boxes.tensor.tolist()
    # ET.SubElement(fields, "scores").text = prediction['instances'].scores.tolist()
    # ET.SubElement(fields, "pred_classes").text = prediction['instances'].pred_classes.tolist()
    # # ET.SubElement(fields, "pred_masks").text = prediction['instances'].pred_masks.tolist()
    # # xml = ET.ElementTree(root)
    # ET.indent(root)
    # pretty_xml = ET.tostring(root, encoding='unicode')
    # return pretty_xml

    ## ATTEMPT 70
    # root = etree.Element("instance")
    # etree.SubElement(root, "num_instances").text = len(prediction)
    # fields = etree.SubElement(root, "fields")
    # etree.SubElement(fields, "pred_boxes").text = prediction['instances'].pred_boxes.tensor.tolist()
    # etree.SubElement(fields, "scores").text = prediction['instances'].scores.tolist()
    # etree.SubElement(fields, "pred_classes").text = prediction['instances'].pred_classes.tolist()
    # # ET.SubElement(fields, "pred_masks").text = prediction['instances'].pred_masks.tolist()
    # return etree.tostring(root, pretty_print=True)

def combine_xmls(list_of_xmls):
    first = None
    for xml in list_of_xmls:
        data = ET.fromstring(xml).getroot()
        if first is None:
            first = data
        else:
            first.extend(data)
    if first is not None:
        return ET.tostring(first)

def main(cfg, image_path):
    predictor = VizPredictor(cfg)
    result_image, prediction= predictor(image_path)
    # mod_instances = ModifiedInstances(prediction)
    # xml_result = mod_instances.convert_to_xml()
    xml_result = convert_instances_to_xml(prediction)
    return result_image, xml_result

if __name__ == '__main__':
    config_file = "../configs/COCO/demo_config_COCO.yaml"
    # image_path = r'../data/datasets/VOCdevkit/VOC2007/JPEGImages/000140.jpg'
    # r'../demo/images/select_images/A.jpg', r'../demo/images/select_images/B.jpg', r'../demo/images/select_images/C.jpg', r'../demo/images/select_images/D.jpg',

    image_paths = [r'../demo/images/select_images/T.jpg']
    # image_path = r'C:/Users/Julia Rosenrauch/Desktop/ENPH479/UniT/demo/images/select_images/A.jpg'
    cfg = setup(config_file)
    predictor = VizPredictor(cfg)

    for image_path in image_paths:
        result_image, prediction= predictor(image_path)
        # mod_instances = ModifiedInstances(prediction)
        # json_instances = mod_instances.convert_to_json()
        # xml_instances = mod_instances.convert_to_xml()

        xml_result = convert_instances_to_xml(prediction)
        # print("MOD INSTANCE: ", json_instances)
        # print("XML: ", xml_result)
